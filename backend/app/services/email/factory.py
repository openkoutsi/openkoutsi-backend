"""Provider selection.

``get_email_provider()`` resolves the ``email_provider`` setting to a concrete
:class:`~backend.app.services.email.base.EmailProvider`. To add a provider,
implement the interface in this package and register its class in ``_PROVIDERS``.
"""

from functools import lru_cache

from backend.app.core.config import Settings, settings
from backend.app.services.email.base import EmailProvider
from backend.app.services.email.lettermint import LettermintProvider

_PROVIDERS: dict[str, type[EmailProvider]] = {
    LettermintProvider.PROVIDER_NAME: LettermintProvider,
}


def build_email_provider(config: Settings) -> EmailProvider:
    """Construct the provider named by ``config.email_provider``."""
    name = config.email_provider
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown email_provider {name!r}. "
            f"Known providers: {', '.join(sorted(_PROVIDERS))}."
        )
    return provider_cls.from_settings(config)


@lru_cache(maxsize=1)
def get_email_provider() -> EmailProvider:
    """Return the configured provider (cached for the process lifetime)."""
    return build_email_provider(settings)
