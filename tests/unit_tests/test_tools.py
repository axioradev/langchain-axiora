"""Unit tests for langchain-axiora tools."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest
from langchain_core.tools import ToolException

from langchain_axiora import AxioraToolkit
from langchain_axiora.api_wrapper import AxioraAPIWrapper
from langchain_axiora.tools import (
    ALL_TOOLS,
    GetCompanyTool,
    GetFinancialsTool,
    SearchCompaniesTool,
)


@pytest.fixture
def api():
    return AxioraAPIWrapper(api_key="ax_test_key")


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------


def test_toolkit_returns_all_tools():
    toolkit = AxioraToolkit(api_key="ax_test_key")
    tools = toolkit.get_tools()
    assert len(tools) == 18
    names = {t.name for t in tools}
    assert "axiora_search_companies" in names
    assert "axiora_get_financials" in names
    assert "axiora_get_health_score" in names


def test_toolkit_selected_tools():
    toolkit = AxioraToolkit(
        api_key="ax_test_key",
        selected_tools=["axiora_search_companies", "axiora_get_financials"],
    )
    tools = toolkit.get_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"axiora_search_companies", "axiora_get_financials"}


def test_toolkit_invalid_selected_tools():
    with pytest.raises(ValueError, match="Invalid tool names"):
        AxioraToolkit(
            api_key="ax_test_key",
            selected_tools=["axiora_search_companies", "nonexistent_tool"],
        )


def test_toolkit_reads_env_var():
    with patch.dict(os.environ, {"AXIORA_API_KEY": "ax_from_env"}):
        toolkit = AxioraToolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 18


# ---------------------------------------------------------------------------
# Tool init DX — all three patterns work
# ---------------------------------------------------------------------------


def test_tool_init_with_api_key_directly():
    """Users can pass api_key= directly without creating an AxioraAPIWrapper."""
    tool = GetFinancialsTool(api_key="ax_direct_key")
    assert tool.api.api_key.get_secret_value() == "ax_direct_key"


def test_tool_init_with_wrapper(api: AxioraAPIWrapper):
    """Users can pass a shared wrapper."""
    tool = GetFinancialsTool(api=api)
    assert tool.api is api


def test_tool_init_from_env():
    """Tools auto-read AXIORA_API_KEY from env."""
    with patch.dict(os.environ, {"AXIORA_API_KEY": "ax_env_key"}):
        tool = GetFinancialsTool()
        assert tool.api.api_key.get_secret_value() == "ax_env_key"


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_tool_has_correct_metadata():
    tool = SearchCompaniesTool(api_key="ax_test")
    assert tool.name == "axiora_search_companies"
    assert "Japanese" in tool.description
    assert tool.args_schema is not None
    assert tool.handle_tool_error is True


def test_all_tools_have_args_schema():
    """Every tool except GetCoverage has an args_schema."""
    from langchain_axiora.tools import GetCoverageTool

    for tool_cls in ALL_TOOLS:
        tool = tool_cls(api_key="ax_test")
        assert tool.name.startswith("axiora_"), f"{tool.name} missing prefix"
        assert tool.handle_tool_error is True, f"{tool.name} missing handle_tool_error"
        if tool_cls is not GetCoverageTool:
            assert tool.args_schema is not None, f"{tool.name} missing args_schema"


def test_all_tools_have_async():
    """Every tool implements both _run and _arun."""
    for tool_cls in ALL_TOOLS:
        assert hasattr(tool_cls, "_run"), f"{tool_cls.__name__} missing _run"
        assert hasattr(tool_cls, "_arun"), f"{tool_cls.__name__} missing _arun"


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _mock_sync_client(response_json: dict) -> MagicMock:
    """Create a mock httpx client that returns response_json."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = response_json

    mock_client = MagicMock()
    mock_client.request.return_value = mock_resp
    return mock_client


def test_search_companies_calls_api(api: AxioraAPIWrapper):
    mock_client = _mock_sync_client({"data": [{"name": "Toyota"}], "meta": {"total": 1}})
    with patch.object(
        type(api), "sync_client", new_callable=PropertyMock, return_value=mock_client
    ):
        tool = SearchCompaniesTool(api=api)
        result = tool._run(query="Toyota")
        assert "Toyota" in result


def test_get_company_calls_api(api: AxioraAPIWrapper):
    mock_client = _mock_sync_client(
        {"data": {"edinet_code": "E02144", "name_en": "Toyota"}, "meta": {}}
    )
    with patch.object(
        type(api), "sync_client", new_callable=PropertyMock, return_value=mock_client
    ):
        tool = GetCompanyTool(api=api)
        result = tool._run(code="7203")
        assert "E02144" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_tool_raises_tool_exception_on_http_error(api: AxioraAPIWrapper):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"detail": "Company not found"}
    mock_resp.text = "Not Found"

    mock_client = MagicMock()
    mock_client.request.side_effect = httpx.HTTPStatusError(
        "Not Found", request=MagicMock(), response=mock_resp
    )

    with patch.object(
        type(api), "sync_client", new_callable=PropertyMock, return_value=mock_client
    ):
        tool = GetCompanyTool(api=api)
        with pytest.raises(ToolException, match="404"):
            tool._run(code="INVALID")


def test_error_hint_for_401(api: AxioraAPIWrapper):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"detail": "Unauthorized"}
    mock_resp.text = "Unauthorized"

    mock_client = MagicMock()
    mock_client.request.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_resp
    )

    with patch.object(
        type(api), "sync_client", new_callable=PropertyMock, return_value=mock_client
    ):
        tool = GetCompanyTool(api=api)
        with pytest.raises(ToolException, match="AXIORA_API_KEY"):
            tool._run(code="7203")


# ---------------------------------------------------------------------------
# API wrapper
# ---------------------------------------------------------------------------


def test_api_wrapper_cleans_none_params():
    api = AxioraAPIWrapper(api_key="test")
    cleaned = api._clean({"a": 1, "b": None, "c": "hello"})
    assert cleaned == {"a": 1, "c": "hello"}


def test_api_wrapper_secret_str():
    api = AxioraAPIWrapper(api_key="ax_secret_key")
    assert "ax_secret_key" not in repr(api)
    assert api.api_key.get_secret_value() == "ax_secret_key"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def test_retriever_init():
    from langchain_axiora import AxioraRetriever

    retriever = AxioraRetriever(api_key="ax_test")
    assert retriever.k == 10
    assert retriever.section is None


def test_retriever_with_section():
    from langchain_axiora import AxioraRetriever

    retriever = AxioraRetriever(api_key="ax_test", section="risk_factors", k=5)
    assert retriever.section == "risk_factors"
    assert retriever.k == 5


def test_retriever_to_documents():
    from langchain_axiora.retriever import AxioraRetriever

    result = {
        "data": [
            {
                "content": "Semiconductor supply chain risks...",
                "company_name": "Toyota",
                "doc_id": "S100ABCD",
                "section": "risk_factors",
            },
            {
                "content": "Foreign exchange fluctuation...",
                "company_name": "Honda",
                "doc_id": "S100EFGH",
                "section": "risk_factors",
            },
        ],
        "meta": {"total": 2},
    }
    docs = AxioraRetriever._to_documents(result)
    assert len(docs) == 2
    assert docs[0].page_content == "Semiconductor supply chain risks..."
    assert docs[0].metadata["company_name"] == "Toyota"
    assert docs[0].metadata["doc_id"] == "S100ABCD"
    assert "content" not in docs[0].metadata
