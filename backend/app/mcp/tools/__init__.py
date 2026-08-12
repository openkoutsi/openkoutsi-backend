"""The published tools (issue #42).

Importing this package is what registers them: each module below calls
:func:`backend.app.mcp.registry.tool`, and :func:`~backend.app.mcp.registry.all_tools`
imports this package lazily the first time anyone asks what exists.

Ten tools, chosen to be **task-shaped** rather than table-shaped. There is no
``query_activities(sql)`` and no ``get_table``; each tool answers a question a
coach actually asks — "how is this athlete going?", "what did they do on
Tuesday?", "are they hitting the plan?" — and returns the aggregate that
answers it. A generic data-access tool would push the shaping work onto the
model, which is exactly the work it is worst at and the work that costs the most
context.

Nine of them are about training that happened; ``get_athlete_profile`` is about
the athlete those answers are read against — the zone boundaries the zone
figures are measured in, and the constraints no activity data reveals.
"""

from backend.app.mcp.tools import (  # noqa: F401  (import side effect is the point)
    activities,
    goals,
    intensity,
    plans,
    power,
    profile,
    training,
)
