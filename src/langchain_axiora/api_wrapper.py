"""Thin HTTP wrapper around the Axiora REST API."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

DEFAULT_BASE_URL = "https://api.axiora.dev/v1"


class AxioraAPIWrapper(BaseModel):
    """Shared HTTP client used by all Axiora LangChain tools.

    Reuses connections via persistent httpx clients for better performance.
    """

    api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "AXIORA_API_KEY",
            error_message=(
                "Axiora API key not found. Set the AXIORA_API_KEY environment variable "
                "or pass api_key= to the constructor."
            ),
        ),
    )
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=30.0)

    _sync_client: httpx.Client | None = PrivateAttr(default=None)
    _async_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    def _headers(self) -> dict[str, str]:
        from langchain_axiora import __version__

        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "User-Agent": f"langchain-axiora/{__version__}",
            "Accept": "application/json",
        }

    def _clean(self, params: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in params.items() if v is not None}

    @property
    def sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                headers=self._headers(), timeout=self.timeout
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                headers=self._headers(), timeout=self.timeout
            )
        return self._async_client

    def request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        """Synchronous API request. Returns the JSON body."""
        url = f"{self.base_url.rstrip('/')}{path}"
        resp = self.sync_client.request(method, url, params=self._clean(params or {}))
        resp.raise_for_status()
        return resp.json()

    async def arequest(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Async API request. Returns the JSON body."""
        url = f"{self.base_url.rstrip('/')}{path}"
        resp = await self.async_client.request(method, url, params=self._clean(params or {}))
        resp.raise_for_status()
        return resp.json()
