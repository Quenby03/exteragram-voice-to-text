#!/usr/bin/env python3
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLUGIN_FILE = ROOT / "groq_voice_to_text.py"
INDEX_FILE = ROOT / "extera-plugins.json"

REPOSITORY = {
    "schema": 1,
    "name": "Groq Voice to Text",
    "description": "Плагин exteraGram для преобразования голосовых сообщений (ГС) в текст через Groq Whisper API",
    "author": "@Quenby03",
    "homepage": "https://github.com/Quenby03/exteragram-voice-to-text",
}

FIELDS = (
    "id", "name", "description", "author", "version", "icon",
    "app_version", "sdk_version", "requirements", "category",
)


def get_meta(path: Path):
    meta = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("__") and target.id.endswith("__"):
                k = target.id[2:-2]
                if k in FIELDS:
                    meta[k] = ast.literal_eval(node.value)
    return meta


def main():
    if not PLUGIN_FILE.exists():
        print(f"File {PLUGIN_FILE} not found!")
        return

    data = PLUGIN_FILE.read_bytes()
    meta = get_meta(PLUGIN_FILE)

    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data)

    plugin_entry = {
        **meta,
        "file": PLUGIN_FILE.name,
        "sha256": sha256,
        "size": size,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    index_data = {
        **REPOSITORY,
        "plugins": [plugin_entry],
    }

    INDEX_FILE.write_text(json.dumps(index_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"extera-plugins.json updated: sha256={sha256}, size={size}")


if __name__ == "__main__":
    main()
