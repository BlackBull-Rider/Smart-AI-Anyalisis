"""
GREEN BULL RIDER V6
Module: backend/providers/provider_manager.py

Provider Manager

Python 3.13 Compatible
"""

from __future__ import annotations

import logging

from backend.providers.base_provider import BaseProvider
from backend.providers.yahoo_provider import YahooProvider

logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Provider Registry.
    """

    def __init__(self) -> None:

        self._providers: dict[str, BaseProvider] = {}

        self.register(
            "yahoo",
            YahooProvider(),
        )

        self._default = "yahoo"

    # ==================================================================
    # Registry
    # ==================================================================

    def register(
        self,
        name: str,
        provider: BaseProvider,
    ) -> None:

        self._providers[name.lower()] = provider

        logger.info(
            "Provider registered: %s",
            name,
        )

    def unregister(
        self,
        name: str,
    ) -> None:

        self._providers.pop(
            name.lower(),
            None,
        )

    # ==================================================================
    # Get Provider
    # ==================================================================

    def get(
        self,
        name: str | None = None,
    ) -> BaseProvider:

        if name is None:
            name = self._default

        provider = self._providers.get(
            name.lower(),
        )

        if provider is None:

            raise ValueError(
                f"Unknown provider: {name}"
            )

        return provider

    def set_default(
        self,
        name: str,
    ) -> None:

        if name.lower() not in self._providers:

            raise ValueError(
                f"Provider not registered: {name}"
            )

        self._default = name.lower()

    def available(self) -> list[str]:

        return sorted(
            self._providers.keys()
        )


provider_manager = ProviderManager()

