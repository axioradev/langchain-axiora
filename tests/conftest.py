"""Shared test fixtures."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from langchain_axiora.api_wrapper import AxioraAPIWrapper


@pytest.fixture
def api():
    """AxioraAPIWrapper with a test key (no real API calls)."""
    return AxioraAPIWrapper(api_key="ax_test_key")


@pytest.fixture
def mock_api(api: AxioraAPIWrapper):
    """Yield (api, mock_client) with a mock HTTP client attached."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": [], "meta": {}}

    mock_client = MagicMock()
    mock_client.request.return_value = mock_resp

    with patch.object(
        type(api), "sync_client", new_callable=PropertyMock, return_value=mock_client
    ):
        yield api, mock_client, mock_resp
