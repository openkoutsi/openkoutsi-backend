# Activity test fixtures

Every `*.fit` file in this directory is picked up automatically and used to
parametrize the FIT-processing integration tests (`test_activities.py`,
`test_power.py`, `test_distance.py`, `test_athlete.py`). Drop a new file in and
it gets exercised through the upload → parse → analyse pipeline — no test edits
needed. Tests gate producer-specific assertions on what each file actually
contains (see `tests/integration/_fit_fixtures.py`), so a run with no power or an
indoor ride with no speed is handled correctly rather than failing.

The `*.gpx` and `*.tcx` files are named explicitly by the import tests rather
than discovered, because those tests turn on which format a given file is.

This directory is the **one committed exception** to `testdata/` being
git-ignored, so anything placed here **will** be committed. Only add files that
are safe to publish.

## Synthetic fixtures

The `synthetic_*.fit` files contain entirely made-up data and are safe to
commit. Regenerate them with:

```console
uv run python scripts/generate_synthetic_fit_fixtures.py
```

They span the capability matrix the tests care about:

| File | Power | Speed | GPS | Notes |
| --- | --- | --- | --- | --- |
| `synthetic_bike_power_gps.fit` | ✓ | ✓ | ✓ | |
| `synthetic_run_no_power.fit` | – | ✓ | ✓ | |
| `synthetic_indoor_no_gps.fit` | ✓ | – | – | |
| `synthetic_bike_hr_dropout.fit` | ✓ | ✓ | – | gappy streams — see below |

### `synthetic_bike_hr_dropout.fit`

A 600-second ride carrying the two kinds of hole a real file has, which is what
the stream-alignment contract exists for (backend issue #76):

- **A heart-rate dropout**, seconds 120–240: the strap loses contact while every
  other channel keeps recording. Before #76 this shifted every later HR sample
  120 positions earlier relative to power instead of leaving a gap.
- **A device pause**, seconds 400–460: no `record` frame at all, so every channel
  has a hole at the same place, and the session's timer time (540 s) is genuinely
  shorter than the elapsed grid (600 s).

Tests import the window boundaries from
`scripts/generate_synthetic_fit_fixtures.py` rather than restating them.

## Synthetic GPX and TCX fixtures

The two XML formats a Strava bulk export contains (issue #36). Regenerate with:

```console
uv run python scripts/generate_synthetic_activity_fixtures.py
```

| File | Power | HR | Laps | Notes |
| --- | --- | --- | --- | --- |
| `synthetic_ride.gpx` | ✓ | ✓ | – | 1 Hz, Garmin `TrackPointExtension` + Strava-style `<power>` |
| `synthetic_hr_only.gpx` | – | ✓ | – | what most GPX in the wild actually is |
| `synthetic_ride.tcx` | ✓ | ✓ | 2 | device distance per point, two 5-minute laps |
| `synthetic_ride.gpx.gz` | ✓ | ✓ | – | the gzip form the export ships |

`synthetic_ride.gpx` and `synthetic_ride.tcx` describe **the same ride**: ten
minutes at a steady 8 m/s with a single 60 m climb. That is what lets the tests
assert two independent parsers agree on distance, ascent and the averages, which
is the property that makes importing a mixed archive trustworthy.

Their coordinates trace a line through open water in the Gulf of Bothnia, so —
unlike a real ride — they identify nobody.

## Adding a real ride

Real device files usually contain GPS traces that reveal where you live or
train. Strip location data before committing:

```console
uv run python scripts/strip_fit_location.py ~/my_ride.fit testdata/fixtures/my_ride.fit
```

The stripper removes latitude/longitude and GPS-accuracy fields but keeps power,
heart rate, cadence, speed, distance, altitude and timestamps. Note that even a
stripped file still carries timestamps, HR and power, which can be somewhat
identifying — only commit files you're comfortable publishing.
