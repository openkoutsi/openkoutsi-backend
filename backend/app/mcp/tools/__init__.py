"""The published tools (issue #42).

Importing this package is what registers them: each module below calls
:func:`backend.app.mcp.registry.tool`, and :func:`~backend.app.mcp.registry.all_tools`
imports this package lazily the first time anyone asks what exists.

Ten tools, **task-shaped** rather than table-shaped. There is no
``query_activities(sql)`` and no ``get_table``: each answers a question a coach
asks and returns the aggregate that answers it, since a generic data-access tool
would push the shaping work onto the model — what it is worst at, and what costs
the most context.

Nine are about training that happened; ``get_athlete_profile`` is about the
athlete those answers are read against.
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
