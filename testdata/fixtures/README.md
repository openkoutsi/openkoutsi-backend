# FIT test fixtures

Every `*.fit` file in this directory is picked up automatically and used to
parametrize the FIT-processing integration tests (`test_activities.py`,
`test_power.py`, `test_distance.py`, `test_athlete.py`). Drop a new file in and
it gets exercised through the upload → parse → analyse pipeline — no test edits
needed. Tests gate producer-specific assertions on what each file actually
contains (see `tests/integration/_fit_fixtures.py`), so a run with no power or an
indoor ride with no speed is handled correctly rather than failing.

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

| File | Power | Speed | GPS |
| --- | --- | --- | --- |
| `synthetic_bike_power_gps.fit` | ✓ | ✓ | ✓ |
| `synthetic_run_no_power.fit` | – | ✓ | ✓ |
| `synthetic_indoor_no_gps.fit` | ✓ | – | – |

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
