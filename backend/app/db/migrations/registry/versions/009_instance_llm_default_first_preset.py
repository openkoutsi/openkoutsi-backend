"""Collapse instance LLM single-config into presets; first preset is the default.

The instance LLM configuration becomes purely a list of presets
(``llm_models``) whose **first entry is the default** selection. Everything is
defined per-preset: the separate default-model field (``llm_model``), the
instance-level single connection (``llm_base_url`` / ``llm_api_key_enc``) and
the global ``llm_extra_headers`` are all removed — they no longer act as
fallbacks/globals.

To preserve existing behaviour the data is migrated forward before the columns
are dropped:

* the instance single ``base_url`` / ``api_key_enc`` are inlined into any preset
  that omitted them (they used to inherit from the instance-level values);
* the global ``llm_extra_headers`` are merged into every preset's own headers
  (preset headers win);
* if no presets existed, one is synthesised from the single config;
* the old default (``llm_model``) is moved to the front of the list (synthesised
  from the single config when it named a bare model id).

Nothing falls back to env vars any more — a preset must carry everything it
needs.

Revision ID: 009_instance_llm_default_first_preset
Revises: 008_instance_llm_models_headers
Create Date: 2026-07-09
"""
import json

import sqlalchemy as sa
from alembic import op

revision = "009_instance_llm_default_first_preset"
down_revision = "008_instance_llm_models_headers"
branch_labels = None
depends_on = None


def _load_presets(raw) -> list:
    """Coerce a stored JSON value into a list of preset dicts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return val if isinstance(val, list) else []


def _load_headers(raw) -> dict:
    """Coerce a stored JSON value into a headers dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return val if isinstance(val, dict) else {}


def upgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, llm_base_url, llm_model, llm_api_key_enc, llm_models, "
            "llm_extra_headers FROM instance_settings"
        )
    ).fetchall()

    for rid, base_url, model, api_key_enc, models_raw, headers_raw in rows:
        single_base = (base_url or "").strip()
        single_key = api_key_enc or None
        default_name = (model or "").strip()
        global_headers = headers_raw if isinstance(headers_raw, dict) else _load_headers(headers_raw)

        presets = [p for p in _load_presets(models_raw) if isinstance(p, dict)]

        # 1. Inline the instance single config + global headers into presets that
        #    omitted them (they used to inherit from the instance-level values).
        for p in presets:
            if single_base and not p.get("base_url"):
                p["base_url"] = single_base
            if single_key and not p.get("api_key_enc"):
                p["api_key_enc"] = single_key
            if global_headers:
                merged = {**global_headers, **(p.get("headers") or {})}
                if merged:
                    p["headers"] = merged

        # 2. No presets but a single connection existed → synthesise one.
        if not presets and (single_base or single_key or default_name or global_headers):
            entry: dict = {"name": default_name or "default"}
            if single_base:
                entry["base_url"] = single_base
            if single_key:
                entry["api_key_enc"] = single_key
            if global_headers:
                entry["headers"] = dict(global_headers)
            presets = [entry]

        # 3. Move the old default to the front (synthesising it from the single
        #    config when it named a bare model id not present as a preset).
        if default_name:
            idx = next(
                (i for i, p in enumerate(presets)
                 if str(p.get("name", "")).strip() == default_name),
                None,
            )
            if idx is not None:
                if idx != 0:
                    presets.insert(0, presets.pop(idx))
            else:
                entry = {"name": default_name}
                if single_base:
                    entry["base_url"] = single_base
                if single_key:
                    entry["api_key_enc"] = single_key
                if global_headers:
                    entry["headers"] = dict(global_headers)
                presets.insert(0, entry)

        bind.execute(
            sa.text("UPDATE instance_settings SET llm_models = :m WHERE id = :id"),
            {"m": json.dumps(presets) if presets else None, "id": rid},
        )

    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.drop_column("llm_model")
        batch_op.drop_column("llm_base_url")
        batch_op.drop_column("llm_api_key_enc")
        batch_op.drop_column("llm_extra_headers")


def downgrade() -> None:
    with op.batch_alter_table("instance_settings") as batch_op:
        batch_op.add_column(sa.Column("llm_extra_headers", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("llm_api_key_enc", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("llm_base_url", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("llm_model", sa.String(), nullable=True))
