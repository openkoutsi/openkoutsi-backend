"""Alembic env for per-user DBs.

Usage:
    USER_ID=<user-uuid> alembic -c backend/alembic-user.ini upgrade head

The USER_ID environment variable selects which user DB to migrate.
All user DBs share the same schema version.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from backend.app.core.config import settings
import backend.app.models.message_orm  # noqa: F401 — populate UserBase.metadata
import backend.app.models.user_orm  # noqa: F401 — populate UserBase.metadata
from backend.app.db.base import UserBase

config = context.config

user_id = os.environ.get("USER_ID") or config.get_main_option("user_id", "")
if not user_id:
    raise RuntimeError("Set USER_ID environment variable to specify which user DB to migrate")

config.set_main_option(
    "sqlalchemy.url", f"sqlite+aiosqlite:///{settings.user_db_path(user_id)}"
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = UserBase.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
