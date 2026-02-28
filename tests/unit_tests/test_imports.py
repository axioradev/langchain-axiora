"""Verify public API exports match expectations."""

from langchain_axiora import __all__

EXPECTED_ALL = [
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


def test_all_exports_match():
    assert sorted(__all__) == sorted(EXPECTED_ALL)


def test_all_importable():
    """Every name in __all__ is actually importable."""
    import langchain_axiora

    for name in __all__:
        assert hasattr(langchain_axiora, name), f"{name} listed in __all__ but not importable"
