#!/usr/bin/env python3
"""Check the Pathfinder nodes before a calibration-mode DTV run, without changing anything.

    pathfinder_preflight.py --nodes cx47 cx42 --recv recv1

Run it on a host that reaches the nodes' REST port (cx50). On each X-engine it checks that Kotekan was
built from an allowed branch (develop by default, which has no DTV subtraction or GNSS correlator), that
no cudaDtvSubtract runs, that the detector writes fine powers and that the node sends the mask, powers,
fine support and fine powers (dtv_calibration_mode); it prints the mask and recording switches. On recv
it checks that the same four products are received and written. Exit status 0 when every check passes.
Standard library only.
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
    parser.add_argument("--branches", nargs="+", default=["develop"],
                        help="Kotekan branches allowed on the X-engines.")
    parser.add_argument("--port", type=int, default=12048)
    args = parser.parse_args()
    failed = []

    def check(ok, text):
        print(("PASS " if ok else "FAIL ") + text)
        if not ok:
            failed.append(text)

    for host in args.nodes:
        config, version = get(host, args.port, "/config"), get(host, args.port, "/version")
        print(f"== {host} ({version.get('kotekan_version')}, {version.get('git_commit_hash')})")
        check(version.get("branch") in args.branches,
              f"{host}: Kotekan built from {' or '.join(args.branches)} (is {version.get('branch')})")
        detector = [c for c in gpu_commands(config, "run_dtv_detector") if c.get("name") == "cudaPilotProxyDetector"]
        check(bool(detector) and all(c.get("dtv_fine_powers_name") for c in detector),
              f"{host}: the detector writes fine powers")
        sent = sorted(config.get("send_dtv", {}))
        check(all(any(name.startswith(p) for name in sent) for p in PRODUCTS),
              f"{host}: sends {', '.join(PRODUCTS)} (send_dtv has {sent})")
        subtract = [c for c in gpu_commands(config, "run_send_voltage") if c.get("name") == "cudaDtvSubtract"]
        check(not subtract, f"{host}: no DTV subtraction runs")
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
