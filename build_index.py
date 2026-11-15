#!/usr/bin/env python3
"""Build the exteraGram plugin index from every .plugin in plugins/."""

import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLUGINS_DIR = ROOT / "plugins"
INDEX_FILE = ROOT / "extera-plugins.json"
LEGACY_SOURCE = ROOT / "groq_voice_to_text.py"
LEGACY_PLUGIN = PLUGINS_DIR / "groq_voice_to_text" / "groq_voice_to_text.plugin"

REPOSITORY_DEFAULTS = {
    "schema": 1,
    "name": "Maduro exteraGram Plugins",
    "description": "Коллекция плагинов для exteraGram от Maduro.",
    "author": "@Quenby03",
    "homepage": "https://github.com/Quenby03/exteragram-voice-to-text",
}

FIELDS = {
    "id", "name", "description", "author", "version", "icon",
    "app_version", "sdk_version", "requirements", "category",
}
REQUIRED = {"id", "name", "version"}


def read_metadata(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    metadata = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if name.startswith("__") and name.endswith("__") and name[2:-2] in FIELDS:
            metadata[name[2:-2]] = ast.literal_eval(node.value)
    return metadata


def sync_legacy_groq_plugin() -> None:
    if not LEGACY_SOURCE.is_file():
        return
    LEGACY_PLUGIN.parent.mkdir(parents=True, exist_ok=True)
    source = LEGACY_SOURCE.read_bytes()
    if not LEGACY_PLUGIN.exists() or LEGACY_PLUGIN.read_bytes() != source:
        LEGACY_PLUGIN.write_bytes(source)


def main() -> None:
    sync_legacy_groq_plugin()
    if not PLUGINS_DIR.is_dir():
        sys.exit(f"Plugin directory not found: {PLUGINS_DIR}")

    previous = {}
    repository = dict(REPOSITORY_DEFAULTS)
    if INDEX_FILE.is_file():
        old_index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        repository.update({key: value for key, value in old_index.items() if key != "plugins"})
        previous = {entry["id"]: entry for entry in old_index.get("plugins", []) if "id" in entry}

    entries = []
    seen_ids = set()
    for path in sorted(PLUGINS_DIR.rglob("*.plugin")):
        data = path.read_bytes()
        metadata = read_metadata(path)
        missing = sorted(key for key in REQUIRED if not metadata.get(key))
        if missing:
            sys.exit(f"{path}: missing metadata: {', '.join(missing)}")
        plugin_id = metadata["id"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,31}", plugin_id):
            sys.exit(f"{path}: invalid plugin id: {plugin_id!r}")
        if plugin_id in seen_ids:
            sys.exit(f"Duplicate plugin id: {plugin_id}")
        seen_ids.add(plugin_id)

        digest = hashlib.sha256(data).hexdigest()
        entry = dict(metadata)
        entry.update({
            "file": path.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "size": len(data),
        })
        old_entry = previous.get(plugin_id, {})
        if old_entry.get("sha256") == digest and old_entry.get("updated_at"):
            entry["updated_at"] = old_entry["updated_at"]
        else:
            entry["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries.append(entry)

    repository["plugins"] = entries
    INDEX_FILE.write_text(json.dumps(repository, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{INDEX_FILE.name}: {len(entries)} plugin(s)")
    for entry in entries:
        print(f"  {entry['id']}: {entry['file']}")


if __name__ == "__main__":
    main()
