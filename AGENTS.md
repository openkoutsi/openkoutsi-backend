# Agents.md

## Documentation

When adding or changing features, keep the README up to date (at minimum, add a one-liner summary of the change).

Make sure the deployment guide (DEPLOY.md) is kept up to date.

Keep openapi.json up to date by regenerating it after API changes.

## Testing

Always test the results.

Tests are executed by running:

```console
uv run python -m pytest tests/
```

in the project root directory.

For new features, you should add new tests and ensure the existing ones pass.

For bug fixes, you should ensure existing tests pass.
It might be a good idea to add a test to cover the bug as well, to catch regressions in the future.

When new mandatory environment variables are added, ensure that CI runs apply a suitable placeholder value so that the tests running in CI will work.

## Frontend

The web frontend lives in its own repository: [openkoutsi/openkoutsi-web](https://github.com/openkoutsi/openkoutsi-web).
Changes to the UI, translations, and anything under the old `frontend/` directory belong there.
