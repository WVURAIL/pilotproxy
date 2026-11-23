"""The Pathfinder preflight against fake Kotekan REST servers: a node on 127.0.0.1 and recv on 127.0.0.2."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = ["mask", "powers", "fine_support", "fine_powers"]


def node_config(stat):
    command = dict(name="cudaPilotProxyDetector", dtv_fine_powers_name="dtv_fine_powers",
                   **({"dtv_stat_name": "dtv_stat"} if stat else {}))
    sent = [f"{p}_{n}" for p in PRODUCTS + (["stat"] if stat else []) for n in (0, 1)]
    return {"run_dtv_detector": {"gpu_0": {"commands": [command]}, "gpu_1": {"commands": [command]}},
            "send_dtv": {name: {} for name in sent}, "dtv_mask_enabled": {"enabled": False},
            "dtv_write_enabled": {"enabled": True}}


def recv_config(products):
    return {"buffer_recv_dtv": {p: {} for p in products}, "write_dtv": {p: {} for p in products}}


def serve(address, config):
    replies = {"/config": config, "/version": {"branch": "develop", "kotekan_version": "test", "git_commit_hash": "0"}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(replies[self.path]).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(address, Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("stat, recv_products, ready", [
    (False, PRODUCTS, True),                       # a detector without dtv_stat (before kotekan#1924)
    (True, PRODUCTS + ["stat"], True),
    (True, PRODUCTS, False),                       # recv rendered without dtv_stat: a v3 run would record no decision
])
def test_preflight_requires_dtv_stat_where_the_detector_writes_it(stat, recv_products, ready):
    node = serve(("127.0.0.1", 0), node_config(stat))
    port = node.server_address[1]
    try:
        recv = serve(("127.0.0.2", port), recv_config(recv_products))
    except OSError:
        node.shutdown()
        pytest.skip("127.0.0.2 is not available as a loopback address")
    try:
        done = subprocess.run([sys.executable, str(ROOT / "tools" / "pathfinder_preflight.py"), "--nodes", "127.0.0.1",
                               "--recv", "127.0.0.2", "--port", str(port)], capture_output=True, text=True, timeout=60)
    finally:
        node.shutdown()
        recv.shutdown()
    assert (done.returncode == 0) == ready, done.stdout
    assert ("stat" in done.stdout) == stat
