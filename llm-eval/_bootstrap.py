"""Import shim so promptfoo's Python subprocesses can load the backend package.

promptfoo runs each prompt/assert file as a standalone script, so this module
does two things every entry point needs before importing ``backend.*``:

* puts the repository root (``openkoutsi-backend/``) on ``sys.path`` so
  ``backend.app.*`` and ``openkoutsi.*`` import the same code the app runs;
* provides a dummy ``SECRET_KEY``. ``backend.app.core.config.Settings()`` runs
  at import time and rejects a missing/weak key — the eval performs no auth, so
  any 32+ char value is fine.

Import it first from every entry point: ``import _bootstrap  # noqa``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # openkoutsi-backend/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Only needed to satisfy Settings() at import time; never used to sign anything.
os.environ.setdefault("SECRET_KEY", "llm-eval-dummy-secret-key-not-used-000000000000")
