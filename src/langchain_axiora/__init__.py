"""LangChain integration for Axiora — Japanese financial data API."""

from importlib.metadata import version

from langchain_axiora.api_wrapper import AxioraAPIWrapper
from langchain_axiora.retriever import AxioraRetriever
from langchain_axiora.toolkit import AxioraToolkit
from langchain_axiora.tools import (
    ALL_TOOLS,
    CompareCompaniesTool,
    GetCompanyTool,
    GetCoverageTool,
    GetFilingCalendarTool,
    GetFinancialsTool,
    GetGrowthTool,
    GetHealthRankingTool,
    GetHealthScoreTool,
    GetPeersTool,
    GetRankingTool,
    GetSectorOverviewTool,
    GetTimeseriesTool,
    GetTranslationsTool,
    ListFilingsTool,
    ScreenCompaniesTool,
    SearchCompaniesBatchTool,
    SearchCompaniesTool,
    SearchTranslationsTool,
)

__version__ = version("langchain-axiora")

__all__ = [
    "ALL_TOOLS",
    "AxioraAPIWrapper",
    "AxioraRetriever",
    "AxioraToolkit",
    "CompareCompaniesTool",
    "GetCompanyTool",
    "GetCoverageTool",
    "GetFilingCalendarTool",
    "GetFinancialsTool",
    "GetGrowthTool",
    "GetHealthRankingTool",
    "GetHealthScoreTool",
    "GetPeersTool",
    "GetRankingTool",
    "GetSectorOverviewTool",
    "GetTimeseriesTool",
    "GetTranslationsTool",
    "ListFilingsTool",
    "ScreenCompaniesTool",
    "SearchCompaniesBatchTool",
    "SearchCompaniesTool",
    "SearchTranslationsTool",
]
