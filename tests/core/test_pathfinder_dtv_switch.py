"""pathfinder_dtv_switch.py against a fake Kotekan node: a detect-only bundle is refused before any POST."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUALIFIED = {"test_report_sha256": "ab" * 32, "reviewer": "operator"}


def write_bundle(directory, qualification):
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {"mask_qualification": qualification} if qualification else {}
    profiles = [{"physical_channel": 29, "fine_calibration": {"status": "calibrated", "provenance": provenance}},
                {"physical_channel": 37, "fine_calibration": {"status": "pending"}}]
    (directory / "pilot_profiles.json").write_text(json.dumps({"profiles": profiles}))
    return directory / "pilot_profiles.json"


def serve(profiles_path):
    # the stage takes pilot_profiles_path from the block above it, as Kotekan's config lookup allows
    config = {"merge_dtv_mask": {"pilot_profiles_path": str(profiles_path),
                                 "half_0": {"kotekan_stage": "DtvRfiMask", "enabled_updatable_config": "/dtv_mask_enabled"}},
              "dtv_mask_enabled": {"enabled": False, "valid_at_time_ns": 0}, "dtv_write_enabled": {"enabled": False}}
    posts = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/endpoints":
                self.reply({"POST": ["/dtv_mask_enabled", "/dtv_write_enabled"], "GET": ["/config"]})
            else:
                self.reply(config)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append((self.path, body))
            config[self.path.lstrip("/")].update(body)
            self.reply({})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, posts


def switch(server, *extra):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "pathfinder_dtv_switch.py"), "--hosts", "127.0.0.1",
                           "--port", str(server.server_address[1]), "--lead", "0", *extra],
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("qualification, switched", [(QUALIFIED, True), (None, False),
                                                     ({"test_report_sha256": "ab" * 32}, False)])
def test_mask_on_needs_a_qualified_bundle(tmp_path, qualification, switched):
    server, posts = serve(write_bundle(tmp_path / "bundle", qualification))
    try:
        done = switch(server, "--mask", "on", "--write", "on")
    finally:
        server.shutdown()
    assert (done.returncode == 0) == switched, done.stdout + done.stderr
    if switched:
        assert [path for path, _ in posts] == ["/dtv_write_enabled", "/dtv_mask_enabled"]
    else:
        assert posts == [] and "No node was changed" in done.stderr and "detect-only" in done.stderr


def test_mask_off_and_write_need_no_bundle(tmp_path):
    server, posts = serve(tmp_path / "missing" / "pilot_profiles.json")
    try:
        done = switch(server, "--mask", "off", "--write", "on")
    finally:
        server.shutdown()
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(posts) == 2


def test_a_bundle_this_machine_cannot_see_needs_a_copy(tmp_path):
    server, posts = serve(tmp_path / "on-the-node" / "pilot_profiles.json")
    try:
        refused = switch(server, "--mask", "on")
        assert refused.returncode != 0 and posts == [] and "--bundle-dir" in refused.stderr
        write_bundle(tmp_path / "copy", QUALIFIED)
        done = switch(server, "--mask", "on", "--bundle-dir", str(tmp_path / "copy"))
    finally:
        server.shutdown()
    assert done.returncode == 0, done.stdout + done.stderr
    assert [path for path, _ in posts] == ["/dtv_mask_enabled"]
