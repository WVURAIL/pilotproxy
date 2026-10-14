"""Verify the packaged records without importing or rerunning a study."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, entry in manifest["files"].items():
    path = root / name
    data = path.read_bytes()
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise SystemExit(f"evidence bytes differ: {name}")
actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
if actual != set(manifest["files"]) | {"MANIFEST.json"}:
    raise SystemExit("evidence inventory differs")
print(f"verified {len(manifest['files'])} frozen files")
