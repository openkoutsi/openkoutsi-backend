"""bcrypt runs off the event loop (issue #50).

``hash_password``/``verify_password`` are synchronous and deliberately slow. Used
directly from an async handler they hold the loop for the whole hash, so one
login stalls every other request in the process. The async wrappers exist to
move that CPU to a worker thread.

The property worth pinning is not "the wrapper returns the same answer" — that
would still pass if someone replaced ``asyncio.to_thread`` with a direct call —
but "the hashing happened on some *other* thread". So these tests look at the
thread identity from inside the hasher itself.
"""
import asyncio
import threading
from unittest.mock import patch

from backend.app.core.auth import (
    hash_password,
    hash_password_async,
    verify_password,
    verify_password_async,
)


class TestHashingRunsOffTheLoop:
    async def test_hash_password_async_runs_on_another_thread(self):
        loop_thread = threading.get_ident()
        seen: list[int] = []

        def _record(plain: str) -> str:
            seen.append(threading.get_ident())
            return hash_password(plain)

        with patch("backend.app.core.auth.hash_password", _record):
            await hash_password_async("Testpass1234")

        assert len(seen) == 1
        assert seen[0] != loop_thread

    async def test_verify_password_async_runs_on_another_thread(self):
        loop_thread = threading.get_ident()
        hashed = hash_password("Testpass1234")
        seen: list[int] = []

        def _record(plain: str, stored: str) -> bool:
            seen.append(threading.get_ident())
            return verify_password(plain, stored)

        with patch("backend.app.core.auth.verify_password", _record):
            assert await verify_password_async("Testpass1234", hashed) is True

        assert len(seen) == 1
        assert seen[0] != loop_thread

    async def test_the_loop_keeps_running_during_a_hash(self):
        """The point of the change: other coroutines make progress meanwhile.

        A blocking hash would let the ticker run exactly once — it gets one turn
        before the hash starts and none until it finishes. Off the loop, the
        ticker is limited only by its own sleep.
        """
        ticks = 0
        done = asyncio.Event()

        async def tick():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.001)

        def _slow_hash(plain: str) -> str:
            threading.Event().wait(0.15)
            return "hashed"

        ticker = asyncio.create_task(tick())
        try:
            with patch("backend.app.core.auth.hash_password", _slow_hash):
                assert await hash_password_async("Testpass1234") == "hashed"
        finally:
            done.set()
            await ticker

        assert ticks > 5


class TestAsyncAndSyncAgree:
    async def test_async_hash_verifies_synchronously(self):
        hashed = await hash_password_async("Testpass1234")
        assert verify_password("Testpass1234", hashed) is True
        assert verify_password("Wrongpass1234", hashed) is False

    async def test_sync_hash_verifies_asynchronously(self):
        hashed = hash_password("Testpass1234")
        assert await verify_password_async("Testpass1234", hashed) is True
        assert await verify_password_async("Wrongpass1234", hashed) is False
