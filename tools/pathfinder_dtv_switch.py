#!/usr/bin/env python3
"""Switch the DTV mask and recording on Pathfinder X-engine nodes at one shared time.

    pathfinder_dtv_switch.py --hosts cx19 cx27 --mask on --write on

Every node is checked before any POST, so a bad host changes nothing. Before switching the mask on, the
pilot_profiles.json of each node's DtvRfiMask stages is checked as DtvRfiMask checks it: Kotekan stops if a
mask-on update reaches a detect-only bundle (a calibrated channel without a mask_qualification record). The file
is read at the path in the node's /config, so this machine must see it, or --bundle-dir gives a copy. The mask switches
at the first frame at or after now + --lead seconds, read from this machine's clock, so the
clock must be synchronised. Recording switches at the next whole frame. The script then
reads /config back from each node. Standard library only.

On the live site, choco keeps these switches and pushes them back to the nodes, so switch them with
`choco set` from the choco host (pathfinder_mask_schedule.sh); this script is for nodes choco does not manage.
"""
import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINTS = {"mask": "/dtv_mask_enabled", "write": "/dtv_write_enabled"}
# what a request can raise besides a bad reply: an OSError, or an http.client error that is not one
ERRORS = (OSError, http.client.HTTPException)
HEX = set("0123456789abcdef")


def request(host, port, path, body=None, timeout=5.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://{host}:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        text = reply.read().decode()
    return json.loads(text) if text.strip() else None


def state(config):
    """Return the mask and write switches of a node's active config."""
    return {name: config.get(path.lstrip("/"), {}).get("enabled") for name, path in ENDPOINTS.items()}


def mask_profiles(node, inherited=None):
    """Yield the pilot_profiles_path of each DtvRfiMask stage that /dtv_mask_enabled switches.

    A stage takes a value it does not set from the blocks above it, as Kotekan's config lookup does.
    """
    inherited = dict(inherited or {})
    if isinstance(node, dict):
        inherited.update({k: node[k] for k in ("pilot_profiles_path", "enabled_updatable_config") if k in node})
        if node.get("kotekan_stage") == "DtvRfiMask" and inherited.get("enabled_updatable_config") == ENDPOINTS["mask"]:
            yield inherited.get("pilot_profiles_path")
        for value in node.values():
            yield from mask_profiles(value, inherited)
    elif isinstance(node, list):
        for value in node:
            yield from mask_profiles(value, inherited)


def unqualified_reason(text, path):
    """Why DtvRfiMask refuses to switch the mask on with this pilot_profiles.json, or "" (DtvRfiMask.cpp)."""
    try:
        bundle = json.loads(text)
    except ValueError as error:
        return f"{path} is not JSON: {error}"
    profiles = bundle.get("profiles") if isinstance(bundle, dict) else None
    if not isinstance(profiles, list):
        return f"{path} has no profiles"
    for row in profiles:
        channel = row.get("physical_channel", -1) if isinstance(row, dict) else -1
        calibration = row.get("fine_calibration") if isinstance(row, dict) else None
        if not isinstance(calibration, dict):
            return f"physical channel {channel} has no fine_calibration"
        if calibration.get("status", "") != "calibrated":
            continue
        provenance = calibration.get("provenance", {})
        record = provenance.get("mask_qualification") if isinstance(provenance, dict) else None
        record = record if isinstance(record, dict) else {}
        digest, reviewer = record.get("test_report_sha256", ""), record.get("reviewer", "")
        if not (isinstance(digest, str) and len(digest) == 64 and set(digest) <= HEX and reviewer):
            return f"physical channel {channel} is not qualified for masking"
    return ""


def mask_refusals(host, config, bundle_dir):
    """Return why switching the mask on would stop Kotekan on this host, one line per stage."""
    refusals = []
    for path in sorted({p or "" for p in mask_profiles(config)}):
        local = Path(bundle_dir) / "pilot_profiles.json" if bundle_dir else Path(path)
        try:
            text = local.read_text()
        except OSError as error:
            refusals.append(f"{host}: cannot read {local} here ({error.strerror}); pass --bundle-dir with a copy "
                            "of the node's bundle")
            continue
        reason = unqualified_reason(text, local)
        if reason:
            refusals.append(f"{host}: the bundle {local} is detect-only ({reason}); a mask-on update would stop "
                            "Kotekan")
    return refusals


def preflight(hosts, port, wanted, bundle_dir=None):
    """Return the switch state of each host, or exit if any host cannot be switched."""
    states, failures = {}, []
    for host in hosts:
        try:
            posts = request(host, port, "/endpoints")["POST"]
            missing = [ENDPOINTS[n] for n in wanted if ENDPOINTS[n] not in posts]
            if missing:
                failures.append(f"{host}: no {', '.join(missing)} (detector stages not running?)")
                continue
            config = request(host, port, "/config")
            if wanted.get("mask"):
                refusals = mask_refusals(host, config, bundle_dir)
                if refusals:
                    failures.extend(refusals)
                    continue
            states[host] = state(config)
        except (*ERRORS, ValueError, KeyError) as error:
            failures.append(f"{host}: {error}")
    if failures:
        sys.exit("No node was changed.\n" + "\n".join(failures))
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--port", type=int, default=12048)
    parser.add_argument("--mask", choices=("on", "off"))
    parser.add_argument("--write", choices=("on", "off"))
    parser.add_argument("--lead", type=float, default=20.0, help="Seconds until the mask switches.")
    parser.add_argument("--dry-run", action="store_true", help="Check the nodes and print the POSTs only.")
    parser.add_argument("--bundle-dir", type=Path,
                        help="A copy of the nodes' runtime bundle, read in place of the pilot_profiles_path in each "
                             "node's /config when this machine cannot see it.")
    args = parser.parse_args()
    wanted = {n: getattr(args, n) == "on" for n in ENDPOINTS if getattr(args, n)}

    before = preflight(args.hosts, args.port, wanted, args.bundle_dir)
    for host, switches in before.items():
        print(f"{host}: mask {switches['mask']}, write {switches['write']}")
    if not wanted:
        return

    at_ns = time.time_ns() + int(args.lead * 1e9)
    bodies = {"write": {"enabled": wanted.get("write")}, "mask": {"enabled": wanted.get("mask"), "valid_at_time_ns": at_ns}}
    posts = [(n, bodies[n]) for n in ("write", "mask") if n in wanted]
    if args.dry_run:
        for name, body in posts:
            print(f"would POST {ENDPOINTS[name]} {json.dumps(body)}")
        return

    failed = False
    for host in args.hosts:
        for name, body in posts:
            try:
                request(host, args.port, ENDPOINTS[name], body)
            except (*ERRORS, ValueError) as error:
                detail = error.read().decode() if isinstance(error, urllib.error.HTTPError) else error
                print(f"{host}: POST {ENDPOINTS[name]} failed: {detail}")
                failed = True
    if "mask" in wanted:
        print(f"mask {args.mask} at {time.strftime('%H:%M:%S', time.gmtime(at_ns / 1e9))} UTC ({at_ns} ns)")

    for host in args.hosts:
        try:
            now = state(request(host, args.port, "/config"))
        except (*ERRORS, ValueError) as error:
            print(f"{host}: cannot read /config: {error}")
            failed = True
            continue
        wrong = [n for n, v in wanted.items() if now[n] != v]
        print(f"{host}: mask {now['mask']}, write {now['write']}" + (f"  NOT SET: {', '.join(wrong)}" if wrong else ""))
        failed |= bool(wrong)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
