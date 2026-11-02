#!/usr/bin/env python3
"""Launch or reuse a CANFAR CUDA session with the canfar client (>= 1.4).

The Science Platform's skaha API needs registry credentials to pull the
session image. Obtain your Harbor CLI secret at https://images.canfar.net
under your name, User Profile, then "CLI secret". Supply it through:

    export CANFAR_REGISTRY_USER=<your-cadc-username>
    export CANFAR_REGISTRY_SECRET=<your-cli-secret>

An active CANFAR authentication context from ``cadc-get-cert`` or
``canfar auth login`` is also required. This script uses that context.

Usage:
    python launch_gpu_session.py             # launch or reuse, print connect URL
    python launch_gpu_session.py --status    # show status + URL only
    python launch_gpu_session.py --destroy   # tear down the session
    python launch_gpu_session.py --cores 8 --ram 16 --gpu 2 --timeout 60
"""
import argparse
import os
import sys
import time

DEFAULT_IMAGE = "images.canfar.net/skaha/astroml-cuda:latest"
DEFAULT_NAME = "cupy-gpu"
POLL_INTERVAL_SECONDS = 10


def _positive_int(value: str) -> int:
    """Parse a positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _inject_registry_creds() -> bool:
    """Copy the registry env vars into the nested settings of the canfar client.

    With ``CANFAR_CONFIG__REGISTRY__{USERNAME,SECRET}`` set, a plain
    ``Session()`` loads the normal auth context and also picks up the Harbor
    credentials. The script then never builds its own Configuration, which
    could overwrite the loaded one. Returns True when both variables are set.
    """
    user = os.environ.get("CANFAR_REGISTRY_USER")
    secret = os.environ.get("CANFAR_REGISTRY_SECRET")
    if user and secret:
        os.environ.setdefault("CANFAR_CONFIG__REGISTRY__USERNAME", user)
        os.environ.setdefault("CANFAR_CONFIG__REGISTRY__SECRET", secret)
        return True
    sys.stderr.write(
        "WARNING: CANFAR_REGISTRY_USER / CANFAR_REGISTRY_SECRET not set.\n"
        "         The skaha API needs your Harbor CLI secret to pull the image\n"
        "         (https://images.canfar.net -> profile -> CLI secret), or the\n"
        "         session create will fail with HTTP 400 'No authentication\n"
        "         provided for unknown or private image'.\n")
    return False


def _find(session, name):
    """Return a Running/Pending session dict matching ``name``, else None."""
    for s in session.fetch(kind="notebook"):
        if s.get("name") == name and s.get("status") in ("Running", "Pending"):
            return s
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help=f"session name (default: {DEFAULT_NAME})")
    ap.add_argument("--image", default=DEFAULT_IMAGE,
                    help="container image (default: astroml-cuda:latest)")
    ap.add_argument("--cores", type=_positive_int, default=1)
    ap.add_argument("--ram", type=_positive_int, default=8, help="RAM in GB")
    ap.add_argument("--gpu", type=_positive_int, default=1, help="number of GPUs")
    ap.add_argument("--timeout", type=_positive_int, default=999,
                    help="seconds to wait for the session to reach Running")
    ap.add_argument("--status", action="store_true",
                    help="just report status + connect URL")
    ap.add_argument("--destroy", action="store_true",
                    help="tear down the named session")
    args = ap.parse_args()

    _inject_registry_creds()

    try:
        from canfar.sessions import Session
    except ImportError:
        sys.stderr.write(
            "Could not import the canfar client. Install it into THIS env "
            "(not --user, which gets hidden by PYTHONNOUSERSITE):\n"
            "    pip install canfar\n")
        return 1

    session = Session()

    if args.destroy:
        existing = _find(session, args.name)
        if existing:
            session.destroy(existing["id"])
            print(f"destroyed {args.name} ({existing['id']})")
        else:
            print(f"no Running/Pending session named {args.name!r}")
        return 0

    existing = _find(session, args.name)

    if args.status:
        if existing:
            print(f"{args.name} ({existing['id']}): {existing['status']}")
            print(existing.get("connectURL", "(no URL yet)"))
        else:
            print(f"no Running/Pending session named {args.name!r}")
        return 0

    if existing:
        print(f"reusing {args.name} ({existing['id']}), status {existing['status']}")
        print(existing.get("connectURL", "(no URL yet)"))
        return 0

    ids = session.create(
        name=args.name, image=args.image, kind="notebook",
        cores=args.cores, ram=args.ram, gpu=args.gpu,
    )
    sid = ids[0] if isinstance(ids, (list, tuple)) else ids
    print(f"created {args.name} ({sid}); waiting up to {args.timeout}s for Running ...")

    deadline = time.monotonic() + args.timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        info = next((s for s in session.fetch(kind="notebook")
                     if s.get("id") == sid), None)
        status = info.get("status") if info else "?"
        if status == "Running":
            print(f"\nRunning  |  {args.gpu} GPU, {args.cores} cores, {args.ram} GB RAM")
            print(info.get("connectURL", "(no URL reported)"))
            return 0
        print(f"  {status} ...", flush=True)
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    sys.stderr.write(
        f"session {args.name} not Running after {args.timeout}s; "
        f"re-run with --status to fetch the URL once it settles.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
