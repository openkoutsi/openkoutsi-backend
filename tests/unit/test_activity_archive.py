"""Archive expansion and its guards (issue #36).

An import archive is a file the user downloaded from somewhere else, so every
test here is about what happens when it is not what it claims to be: an entry
that expands to a gigabyte, a name that points outside the directory, a zip
inside a zip inside a zip.

The budget tests use small explicit limits rather than the production constants
— building a real 4 GB zip bomb to prove the check fires is not a unit test.
"""
from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import pytest

from backend.app.services.activity_archive import (
    ArchiveTooLarge,
    Budget,
    ExpandedFile,
    expand,
    expand_all,
    is_safe_entry_name,
)

FIXTURES = Path(__file__).parent.parent.parent / "testdata" / "fixtures"
RIDE_GPX = FIXTURES / "synthetic_ride.gpx"

pytestmark = pytest.mark.skipif(
    not RIDE_GPX.exists(),
    reason="synthetic GPX/TCX fixtures missing — run scripts/generate_synthetic_activity_fixtures.py",
)


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def by_name(files: list[ExpandedFile]) -> dict[str, ExpandedFile]:
    return {f.name: f for f in files}


class TestEntryNames:
    @pytest.mark.parametrize(
        "name",
        [
            "../../etc/passwd",
            "activities/../../../root/.ssh/authorized_keys",
            "/etc/passwd",
            "\\windows\\system32",
            "C:/Windows/system.ini",
            "..",
            "",
        ],
    )
    def test_traversal_and_absolute_names_are_rejected(self, name):
        assert is_safe_entry_name(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "activities/12345.fit",
            "12345.fit.gz",
            "a/b/c/ride.gpx",
            "ride..gpx",  # two dots in a filename is not a parent reference
            "..hidden/ride.fit",
        ],
    )
    def test_ordinary_names_are_accepted(self, name):
        assert is_safe_entry_name(name) is True

    def test_a_traversing_entry_is_reported_not_extracted(self, tmp_path):
        archive = write_zip(
            tmp_path / "hostile.zip",
            {"../escape.gpx": RIDE_GPX.read_bytes(), "fine.gpx": RIDE_GPX.read_bytes()},
        )
        files = expand(archive, "hostile.zip", tmp_path / "work")

        results = by_name(files)
        assert results["../escape.gpx"].path is None
        assert "unsafe" in results["../escape.gpx"].error
        # ...and nothing was written outside the work directory.
        assert not (tmp_path / "escape.gpx").exists()
        # The rest of the archive still imports.
        assert results["fine.gpx"].path is not None


class TestExpansion:
    def test_a_plain_file_yields_itself(self, tmp_path):
        source = write(tmp_path / "ride.gpx", RIDE_GPX.read_bytes())
        files = expand(source, "ride.gpx", tmp_path / "work")

        assert len(files) == 1
        assert files[0].format == "gpx"
        assert files[0].error is None

    def test_a_gzip_yields_its_contents_under_the_inner_name(self, tmp_path):
        source = write(tmp_path / "ride.gpx.gz", gzip.compress(RIDE_GPX.read_bytes()))
        files = expand(source, "ride.gpx.gz", tmp_path / "work")

        assert len(files) == 1
        assert files[0].name == "ride.gpx"
        assert files[0].format == "gpx"
        assert files[0].path.read_bytes() == RIDE_GPX.read_bytes()

    def test_a_zip_of_gzips_is_the_shape_strava_exports(self, tmp_path):
        archive = write_zip(
            tmp_path / "export.zip",
            {
                "activities/1.gpx.gz": gzip.compress(RIDE_GPX.read_bytes()),
                "activities/2.gpx.gz": gzip.compress(RIDE_GPX.read_bytes()),
            },
        )
        files = expand(archive, "export.zip", tmp_path / "work")

        assert sorted(f.name for f in files) == ["activities/1.gpx", "activities/2.gpx"]
        assert all(f.format == "gpx" for f in files)

    def test_directories_and_archiver_junk_are_skipped(self, tmp_path):
        archive = tmp_path / "export.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("activities/", b"")
            zf.writestr("__MACOSX/._ride.gpx", b"junk")
            zf.writestr(".DS_Store", b"junk")
            zf.writestr("activities/ride.gpx", RIDE_GPX.read_bytes())

        files = expand(archive, "export.zip", tmp_path / "work")
        assert [f.name for f in files] == ["activities/ride.gpx"]

    def test_a_file_that_is_not_an_activity_is_reported(self, tmp_path):
        source = write(tmp_path / "notes.txt", b"just some notes")
        files = expand(source, "notes.txt", tmp_path / "work")

        assert files[0].path is None
        assert files[0].format is None
        assert "Not a FIT, GPX or TCX" in files[0].error

    def test_a_corrupt_gzip_is_reported(self, tmp_path):
        source = write(tmp_path / "ride.gpx.gz", b"\x1f\x8b" + b"not really gzip")
        files = expand(source, "ride.gpx.gz", tmp_path / "work")

        assert files[0].path is None
        assert "decompress" in files[0].error

    def test_a_corrupt_zip_is_reported(self, tmp_path):
        source = write(tmp_path / "export.zip", b"PK\x03\x04" + b"\x00" * 40)
        files = expand(source, "export.zip", tmp_path / "work")

        assert files[0].path is None
        assert "archive" in files[0].error.lower()

    def test_one_broken_entry_does_not_lose_the_others(self, tmp_path):
        archive = write_zip(
            tmp_path / "export.zip",
            {
                "good.gpx": RIDE_GPX.read_bytes(),
                "bad.gpx.gz": b"\x1f\x8bnot gzip",
                "also-good.gpx": RIDE_GPX.read_bytes(),
            },
        )
        files = by_name(expand(archive, "export.zip", tmp_path / "work"))

        assert files["good.gpx"].path is not None
        assert files["also-good.gpx"].path is not None
        assert files["bad.gpx.gz"].path is None

    def test_nesting_is_walked_then_stopped(self, tmp_path):
        inner = write_zip(tmp_path / "inner.zip", {"ride.gpx": RIDE_GPX.read_bytes()})
        middle = write_zip(tmp_path / "middle.zip", {"inner.zip": inner.read_bytes()})
        outer = write_zip(tmp_path / "outer.zip", {"middle.zip": middle.read_bytes()})

        files = expand(outer, "outer.zip", tmp_path / "work")
        assert [f.name for f in files] == ["ride.gpx"]
        assert files[0].format == "gpx"

    def test_nesting_deeper_than_the_limit_is_refused(self, tmp_path):
        payload = write_zip(tmp_path / "l0.zip", {"ride.gpx": RIDE_GPX.read_bytes()})
        for level in range(1, 5):
            payload = write_zip(
                tmp_path / f"l{level}.zip", {f"l{level - 1}.zip": payload.read_bytes()}
            )

        files = expand(payload, "bomb.zip", tmp_path / "work")
        assert any("nesting" in (f.error or "") for f in files)


class TestBudgets:
    def test_too_many_entries_fails_the_job(self, tmp_path):
        archive = write_zip(
            tmp_path / "export.zip",
            {f"ride-{i}.gpx": RIDE_GPX.read_bytes() for i in range(5)},
        )
        with pytest.raises(ArchiveTooLarge, match="more than"):
            expand(archive, "export.zip", tmp_path / "work", Budget(entries=3))

    def test_too_many_expanded_bytes_fails_the_job(self, tmp_path):
        # A classic bomb: a tiny zip of highly compressible data.
        archive = write_zip(tmp_path / "bomb.zip", {"big.gpx": b"a" * 10_000_000})
        assert archive.stat().st_size < 50_000

        with pytest.raises(ArchiveTooLarge, match="expands to more than"):
            expand(archive, "bomb.zip", tmp_path / "work", Budget(total_bytes=1_000_000))

    def test_an_oversized_entry_fails_alone(self, tmp_path, monkeypatch):
        from backend.app.services import activity_archive

        monkeypatch.setattr(activity_archive, "MAX_MEMBER_BYTES", 4096)
        archive = write_zip(
            tmp_path / "export.zip",
            {"huge.gpx": b"<gpx>" + b"x" * 100_000, "small.gpx": RIDE_GPX.read_bytes()[:2000]},
        )
        files = by_name(expand(archive, "export.zip", tmp_path / "work"))

        # The oversized entry is a failure with a reason; the job carries on.
        assert files["huge.gpx"].path is None
        assert "larger than" in files["huge.gpx"].error
        assert "small.gpx" in files

    def test_a_lying_size_header_does_not_get_through(self, tmp_path, monkeypatch):
        """The declared size is a pre-check; the enforced one is what comes out.

        A zip's central directory states each entry's uncompressed size, and an
        attacker writes that number. The limit therefore has to be applied to
        the bytes actually read, which is what this checks by making the header
        claim something small.
        """
        from backend.app.services import activity_archive

        monkeypatch.setattr(activity_archive, "MAX_MEMBER_BYTES", 4096)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("liar.gpx", b"<gpx>" + b"x" * 100_000)
        raw = bytearray(buffer.getvalue())
        # Rewrite every 4-byte occurrence of the true size with a small one.
        raw = raw.replace((100_005).to_bytes(4, "little"), (100).to_bytes(4, "little"))
        archive = write(tmp_path / "liar.zip", bytes(raw))

        files = expand(archive, "liar.zip", tmp_path / "work")
        assert all(f.path is None for f in files)

    def test_the_budget_is_shared_across_uploaded_parts(self, tmp_path):
        parts = [
            (write(tmp_path / f"ride-{i}.gpx", RIDE_GPX.read_bytes()), f"ride-{i}.gpx")
            for i in range(4)
        ]
        files = expand_all(parts, tmp_path / "work")
        assert len(files) == 4
        assert all(f.format == "gpx" for f in files)

    def test_budget_counts_every_entry_at_every_level(self, tmp_path):
        budget = Budget(entries=10)
        inner = write_zip(tmp_path / "inner.zip", {"a.gpx": RIDE_GPX.read_bytes()})
        outer = write_zip(
            tmp_path / "outer.zip",
            {"inner.zip": inner.read_bytes(), "b.gpx": RIDE_GPX.read_bytes()},
        )
        expand(outer, "outer.zip", tmp_path / "work", budget)

        # The upload itself, two members of the outer zip, one of the inner.
        assert budget.entries_seen == 4
