#!/usr/bin/env python3
"""Export the FastAPI OpenAPI spec to openapi.json at the repo root."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.main import create_app

app = create_app()
spec = app.openapi()
out = Path(__file__).parent.parent.parent / "openapi.json"
out.write_text(json.dumps(spec, indent=2))
print(f"Wrote {out}")
