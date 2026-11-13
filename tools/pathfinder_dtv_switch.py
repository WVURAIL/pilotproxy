#!/usr/bin/env python3
"""Switch the DTV mask and recording on Pathfinder X-engine nodes at one shared time.

    pathfinder_dtv_switch.py --hosts cx19 cx27 --mask on --write on

Every node is checked before any POST, so a bad host changes nothing. The mask switches
at the first frame at or after now + --lead seconds, read from this machine's clock, so the
clock must be synchronised. Recording switches at the next whole frame. The script then
reads /config back from each node. Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ENDPOINTS = {"mask": "/dtv_mask_enabled", "write": "/dtv_write_enabled"}


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


def preflight(hosts, port, wanted):
    """Return the switch state of each host, or exit if any host cannot be switched."""
    states, failures = {}, []
    for host in hosts:
        try:
            posts = request(host, port, "/endpoints")["POST"]
            missing = [ENDPOINTS[n] for n in wanted if ENDPOINTS[n] not in posts]
            if missing:
                failures.append(f"{host}: no {', '.join(missing)} (detector stages not running?)")
                continue
            states[host] = state(request(host, port, "/config"))
        except (OSError, ValueError, KeyError) as error:
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
    args = parser.parse_args()
    wanted = {n: getattr(args, n) == "on" for n in ENDPOINTS if getattr(args, n)}

    before = preflight(args.hosts, args.port, wanted)
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
            except (OSError, ValueError) as error:
                detail = error.read().decode() if isinstance(error, urllib.error.HTTPError) else error
                print(f"{host}: POST {ENDPOINTS[name]} failed: {detail}")
                failed = True
    if "mask" in wanted:
        print(f"mask {args.mask} at {time.strftime('%H:%M:%S', time.gmtime(at_ns / 1e9))} UTC ({at_ns} ns)")

    for host in args.hosts:
        try:
            now = state(request(host, args.port, "/config"))
        except (OSError, ValueError) as error:
            print(f"{host}: cannot read /config: {error}")
            failed = True
            continue
        wrong = [n for n, v in wanted.items() if now[n] != v]
        print(f"{host}: mask {now['mask']}, write {now['write']}" + (f"  NOT SET: {', '.join(wrong)}" if wrong else ""))
        failed |= bool(wrong)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
