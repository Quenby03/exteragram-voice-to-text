#!/usr/bin/env python3
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "groq_voice_to_text.py"
PLUGIN_FILE = ROOT / "plugins" / "groq_voice_to_text" / "groq_voice_to_text.plugin"
INDEX_FILE = ROOT / "extera-plugins.json"

REPOSITORY = {
    "schema": 1,
    "name": "Maduro exteraGram Plugins",
    "description": "Коллекция плагинов для exteraGram от Maduro.",
    "author": "@Quenby03",
    "homepage": "https://github.com/Quenby03/maduro-exteragram-plugins",
}

FIELDS = (
    "id", "name", "description", "author", "version", "icon",
    "app_version", "sdk_version", "requirements", "category",
)


def read_metadata(path: Path):
    metadata = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("__") and target.id.endswith("__"):
            key = target.id[2:-2]
            if key in FIELDS:
                metadata[key] = ast.literal_eval(node.value)
    return metadata


def main():
    if not SOURCE_FILE.is_file():
        sys.exit(f"Plugin source not found: {SOURCE_FILE}")

    PLUGIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = SOURCE_FILE.read_bytes()
    if not PLUGIN_FILE.exists() or PLUGIN_FILE.read_bytes() != data:
        PLUGIN_FILE.write_bytes(data)

    metadata = read_metadata(PLUGIN_FILE)
    missing = [key for key in ("id", "name", "version") if not metadata.get(key)]
    if missing:
        sys.exit(f"{PLUGIN_FILE}: missing {', '.join(missing)}")

    previous = {}
    if INDEX_FILE.exists():
        previous = {
            plugin["id"]: plugin
            for plugin in json.loads(INDEX_FILE.read_text(encoding="utf-8")).get("plugins", [])
        }

    digest = hashlib.sha256(data).hexdigest()
    entry = {
        **metadata,
        "file": PLUGIN_FILE.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "size": len(data),
    }
    old_entry = previous.get(metadata["id"], {})
    if old_entry.get("sha256") == digest and old_entry.get("updated_at"):
        entry["updated_at"] = old_entry["updated_at"]
    else:
        entry["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    index = {**REPOSITORY, "plugins": [entry]}
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{INDEX_FILE.name}: {len(index['plugins'])} plugin(s), file={entry['file']}")


if __name__ == "__main__":
    main()
