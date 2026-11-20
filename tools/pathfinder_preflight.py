#!/usr/bin/env python3
"""Check the Pathfinder nodes before a calibration-mode DTV run, without changing anything.

    pathfinder_preflight.py --nodes cx47 cx42 --recv recv1

Run it on a host that reaches the nodes' REST port (cx50). From each node's /config it checks that the
detector writes fine powers and the node sends the mask, powers, fine support and fine powers
(dtv_calibration_mode), and that no cudaDtvSubtract names a pilot frequency; it prints the mask and
recording switches. On recv it checks that the same four products are received and written. The
subtraction check reads the configuration only: a subtraction switched off another way still shows,
so confirm a failure there against the data (the pilot's autocorrelation rises once it is off).
Exit status 0 when every check passes. Standard library only.
"""
import argparse
import json
import sys
import urllib.request

PRODUCTS = ("mask", "powers", "fine_support", "fine_powers")


def get(host, port, path):
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=10) as reply:
        return json.loads(reply.read().decode())


def gpu_commands(config, stage):
    block = config.get(stage, {})
    return [command for gpu in ("gpu_0", "gpu_1") for command in block.get(gpu, {}).get("commands", [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", nargs="+", default=["cx47", "cx42"], help="X-engine nodes holding the pilots.")
    parser.add_argument("--recv", default="recv1")
    parser.add_argument("--pilots", type=int, nargs="+", default=[2531, 2715, 2869, 3115],
                        help="Pilot and control freq_ids that no subtraction may touch.")
    parser.add_argument("--port", type=int, default=12048)
    args = parser.parse_args()
    failed = []

    def check(ok, text):
        print(("PASS " if ok else "FAIL ") + text)
        if not ok:
            failed.append(text)

    for host in args.nodes:
        config = get(host, args.port, "/config")
        print(f"== {host} ({get(host, args.port, '/version').get('kotekan_version')})")
        detector = [c for c in gpu_commands(config, "run_dtv_detector") if c.get("name") == "cudaPilotProxyDetector"]
        check(bool(detector) and all(c.get("dtv_fine_powers_name") for c in detector),
              f"{host}: the detector writes fine powers")
        sent = sorted(config.get("send_dtv", {}))
        check(all(any(name.startswith(p) for name in sent) for p in PRODUCTS),
              f"{host}: sends {', '.join(PRODUCTS)} (send_dtv has {sent})")
        subtract = [c for c in gpu_commands(config, "run_send_voltage") if c.get("name") == "cudaDtvSubtract"]
        touched = sorted({f for c in subtract for f in c.get("dtv_freq_ids", [])} & set(args.pilots))
        check(not touched, f"{host}: no DTV subtraction configured on a pilot"
                           + (f" (configured on {touched})" if touched else ""))
        print(f"     switches: mask {config.get('dtv_mask_enabled', {}).get('enabled')}, "
              f"recording {config.get('dtv_write_enabled', {}).get('enabled')}")

    recv = get(args.recv, args.port, "/config")
    print(f"== {args.recv} ({get(args.recv, args.port, '/version').get('kotekan_version')})")
    received, written = sorted(recv.get("buffer_recv_dtv", {})), sorted(recv.get("write_dtv", {}))
    check(all(p in received for p in PRODUCTS), f"{args.recv}: receives {', '.join(PRODUCTS)} (has {received})")
    check(all(p in written for p in PRODUCTS), f"{args.recv}: writes {', '.join(PRODUCTS)} (has {written})")
    print("ready" if not failed else f"NOT ready: {len(failed)} check(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
