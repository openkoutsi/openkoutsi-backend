"""Upgrading every per-user database on deploy (issue #50).

This runs from the container entrypoint before uvicorn is exec'd, so its cost
is start-up latency for the whole instance and its failure is a container that
refuses to serve. Both of those are properties worth pinning.

The cost one is why this file exists. The script used to spawn a fresh Python
interpreter per user — measured at ~0.89 s each, almost none of it migration
work — so the loop was O(users) in process startups on every deploy that
recreated the container. Two things fixed that, and both are load-bearing
enough to break silently: running Alembic in-process, and skipping users whose
database already says it is at head.

The skip only works if databases actually carry a stamp, which is the third
test here: ``init_user_db`` builds the schema with ``create_all``, which writes
no ``alembic_version`` row at all unless something stamps it.

Every test drives the script the way the entrypoint does: **synchronously, with
no running event loop.** ``env.py`` ends in ``asyncio.run()``, so a migration
cannot be invoked from inside one — which is a real constraint on callers, not
merely a test detail.
"""
import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.core.config import settings
from backend.app.db.user_session import init_user_db
from backend.scripts import migrate_user_dbs as script

REPO_ROOT = Path(__file__).resolve().parents[2]


def _revision(user_id: str) -> str | None:
    return script.current_revision(settings.user_db_path(user_id))


def _unstamp(user_id: str) -> None:
    """Make a database look like one built before stamping existed."""
    con = sqlite3.connect(settings.user_db_path(user_id))
    con.execute("DELETE FROM alembic_version")
    con.commit()
    con.close()


class TestAFreshDatabaseIsStampedAtHead:
    """Otherwise the skip below is inert for exactly the newest accounts.

    `create_all` builds the current schema but records no revision, so without
    a stamp every new user's database claims to be at nothing and the next
    deploy replays all of them against it. That works — every migration in this
    tree is idempotent — but it is real work per new user, once, for nothing.
    """

    def test_init_user_db_records_the_revision(self, isolate_user_dbs):
        asyncio.run(init_user_db("fresh"))

        cfg = script.build_config()
        assert _revision("fresh") == script.head_revision(cfg)

    def test_a_stamped_database_needs_no_upgrade(self, isolate_user_dbs):
        asyncio.run(init_user_db("fresh"))

        cfg = script.build_config()
        head = script.head_revision(cfg)
        with patch.object(script.command, "upgrade") as upgrade:
            assert script.migrate_user(cfg, "fresh", head, dry_run=False) is True

        upgrade.assert_not_called()


class TestOnlyUsersBehindHeadAreUpgraded:
    def test_the_one_at_head_is_skipped_and_the_other_is_not(
        self, isolate_user_dbs
    ):
        asyncio.run(init_user_db("current"))
        asyncio.run(init_user_db("behind"))
        _unstamp("behind")

        cfg = script.build_config()
        head = script.head_revision(cfg)
        with patch.object(script.command, "upgrade") as upgrade:
            script.migrate_user(cfg, "current", head, dry_run=False)
            script.migrate_user(cfg, "behind", head, dry_run=False)

        assert upgrade.call_count == 1, "the up-to-date user was migrated anyway"

    def test_an_unstamped_database_really_is_brought_to_head(
        self, isolate_user_dbs
    ):
        """Not mocked: the migrations actually run, and the stamp lands."""
        asyncio.run(init_user_db("behind"))
        _unstamp("behind")
        assert _revision("behind") is None

        cfg = script.build_config()
        script.migrate_user(cfg, "behind", script.head_revision(cfg), dry_run=False)

        assert _revision("behind") == script.head_revision(script.build_config())

    def test_a_second_pass_does_nothing(self, isolate_user_dbs):
        asyncio.run(init_user_db("behind"))
        _unstamp("behind")

        cfg = script.build_config()
        head = script.head_revision(cfg)
        script.migrate_user(cfg, "behind", head, dry_run=False)

        with patch.object(script.command, "upgrade") as upgrade:
            script.migrate_user(cfg, "behind", head, dry_run=False)
        upgrade.assert_not_called()


class TestOneUsersFailureIsTheirsAlone:
    """The contract the subprocess version got for free from a return code.

    In-process, an exception would propagate and abandon everyone after it in
    the loop — so the isolation has to be explicit, and the non-zero exit that
    stops the container from serving has to survive it.
    """

    def test_the_loop_continues_past_a_failure(self, isolate_user_dbs):
        for user_id in ("a", "b", "c"):
            asyncio.run(init_user_db(user_id))
            _unstamp(user_id)

        cfg = script.build_config()
        head = script.head_revision(cfg)
        seen = []

        def _fail_on_b(_cfg, _rev):
            import os
            seen.append(os.environ["USER_ID"])
            if os.environ["USER_ID"] == "b":
                raise RuntimeError("disk is full")

        with patch.object(script.command, "upgrade", side_effect=_fail_on_b):
            results = [
                script.migrate_user(cfg, u, head, dry_run=False) for u in ("a", "b", "c")
            ]

        assert results == [True, False, True]
        assert seen == ["a", "b", "c"], "a failure stopped the users behind it"

    def test_the_script_exits_non_zero_when_a_user_fails(self, isolate_user_dbs):
        """What actually stops the container from serving."""
        asyncio.run(init_user_db("doomed"))
        _unstamp("doomed")

        with (
            patch.object(script.command, "upgrade", side_effect=RuntimeError("nope")),
            patch.object(sys, "argv", ["migrate_user_dbs.py"]),
            pytest.raises(SystemExit) as exit_info,
        ):
            script.main()

        assert exit_info.value.code == 1

    def test_a_clean_run_does_not_exit(self, isolate_user_dbs):
        asyncio.run(init_user_db("fine"))

        with patch.object(sys, "argv", ["migrate_user_dbs.py"]):
            script.main()  # must not raise SystemExit


class TestTheEnvironmentIsLeftAsItWasFound:
    """`USER_ID` selects the database. This process may not own it."""

    def test_it_is_cleared_when_it_was_unset(self, isolate_user_dbs, monkeypatch):
        monkeypatch.delenv("USER_ID", raising=False)
        asyncio.run(init_user_db("someone"))

        with patch.object(sys, "argv", ["migrate_user_dbs.py"]):
            script.main()

        import os
        assert "USER_ID" not in os.environ

    def test_it_is_restored_when_it_was_set(self, isolate_user_dbs, monkeypatch):
        monkeypatch.setenv("USER_ID", "the-callers-choice")
        asyncio.run(init_user_db("someone"))
        _unstamp("someone")

        with patch.object(sys, "argv", ["migrate_user_dbs.py"]):
            script.main()

        import os
        assert os.environ["USER_ID"] == "the-callers-choice"


class TestDryRun:
    def test_it_changes_nothing(self, isolate_user_dbs):
        asyncio.run(init_user_db("behind"))
        _unstamp("behind")

        with (
            patch.object(script.command, "upgrade") as upgrade,
            patch.object(sys, "argv", ["migrate_user_dbs.py", "--dry-run"]),
        ):
            script.main()

        upgrade.assert_not_called()
        assert _revision("behind") is None


class TestItStillRunsAsAScript:
    """The entrypoint invokes it by path, not by import."""

    def test_dry_run_from_the_command_line(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "backend/scripts/migrate_user_dbs.py",
             "--dry-run", "--data-dir", str(tmp_path)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "No user databases found" in result.stdout
