"""
Provider registry.

To add a new provider:
1. Create a module in this package implementing BaseProviderClient
2. Import it here and add an entry to PROVIDERS
3. Add client_id / client_secret settings to config.py
4. Document the new env vars in TODO.md
"""

from backend.app.services.providers.base import BaseProviderClient
from backend.app.services.providers.strava import StravaProviderClient
from backend.app.services.providers.wahoo import WahooClient

PROVIDERS: dict[str, type[BaseProviderClient]] = {
    "strava": StravaProviderClient,
    "wahoo": WahooClient,
}
