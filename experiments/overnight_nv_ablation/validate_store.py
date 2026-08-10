#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("store")
args = parser.parse_args()
path = Path(args.store) / "principles" / "principles.json"
items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
bad = [item.get("memory_id", "<missing>") for item in items if item.get("status") not in {"active", "deprecated"}]
if bad:
    raise SystemExit(f"live Store contains non-active history: {bad[:10]}")
print(json.dumps({"store": args.store, "records": len(items), "active": sum(item.get('status') == 'active' for item in items), "deprecated": sum(item.get('status') == 'deprecated' for item in items)}))
