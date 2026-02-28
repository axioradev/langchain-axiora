"""Integration tests that hit the live Axiora API.

Run with: pytest tests/integration_tests/ -v

Requires AXIORA_API_KEY environment variable to be set.
These tests are skipped in CI unless the env var is present.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AXIORA_API_KEY"),
    reason="AXIORA_API_KEY not set — skipping live API tests",
)


@pytest.fixture
def api_key():
    return os.environ["AXIORA_API_KEY"]


def test_search_companies_live(api_key: str):
    from langchain_axiora import SearchCompaniesTool

    tool = SearchCompaniesTool(api_key=api_key)
    result = tool.invoke({"query": "Toyota"})
    assert "Toyota" in result or "トヨタ" in result


def test_get_financials_live(api_key: str):
    from langchain_axiora import GetFinancialsTool

    tool = GetFinancialsTool(api_key=api_key)
    result = tool.invoke({"code": "7203", "years": 1})
    assert "revenue" in result


def test_get_health_score_live(api_key: str):
    from langchain_axiora import GetHealthScoreTool

    tool = GetHealthScoreTool(api_key=api_key)
    result = tool.invoke({"code": "7203"})
    assert "score" in result


def test_toolkit_live(api_key: str):
    from langchain_axiora import AxioraToolkit

    toolkit = AxioraToolkit(api_key=api_key)
    tools = toolkit.get_tools()
    assert len(tools) == 18


def test_retriever_live(api_key: str):
    from langchain_axiora import AxioraRetriever

    retriever = AxioraRetriever(api_key=api_key, k=3)
    docs = retriever.invoke("semiconductor")
    # May return 0 docs if no translations match, but shouldn't crash
    assert isinstance(docs, list)
