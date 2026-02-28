"""LangChain tool definitions for the Axiora API."""

from __future__ import annotations

import json
from typing import Any, NoReturn, Optional

import httpx
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, model_validator

from langchain_axiora.api_wrapper import AxioraAPIWrapper


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class SearchCompaniesInput(BaseModel):
    query: str = Field(description="Company name (JP or EN), securities code, or EDINET code")
    sector: Optional[str] = Field(default=None, description="Sector filter (e.g. '電気機器')")
    limit: int = Field(default=10, description="Max results (max 50)")


class GetCompanyInput(BaseModel):
    code: str = Field(description="EDINET code (e.g. 'E02144') or securities code (e.g. '7203')")


class GetFinancialsInput(BaseModel):
    code: str = Field(description="EDINET code or securities code")
    years: int = Field(default=5, description="Number of fiscal years (max 20)")


class GetGrowthInput(BaseModel):
    code: str = Field(description="EDINET code or securities code")
    years: int = Field(default=5, description="Number of years (max 20)")


class GetRankingInput(BaseModel):
    metric: str = Field(
        default="revenue",
        description=(
            "Metric to rank by: revenue, net_income, operating_income, "
            "total_assets, roe, roa, operating_margin, net_margin, "
            "equity_ratio, eps, bps"
        ),
    )
    sector: Optional[str] = Field(default=None, description="Optional sector filter")
    order: str = Field(default="desc", description="'desc' for top, 'asc' for bottom")
    limit: int = Field(default=20, description="Number of results (max 100)")


class GetSectorOverviewInput(BaseModel):
    sector: Optional[str] = Field(
        default=None,
        description="Sector name for stats. Omit to list all sectors with company counts.",
    )


class CompareCompaniesInput(BaseModel):
    codes: list[str] = Field(description="List of 2-5 EDINET or securities codes")
    years: int = Field(default=3, description="Number of years (max 10)")


class ScreenCompaniesInput(BaseModel):
    sector: Optional[str] = Field(default=None, description="Sector filter")
    min_revenue: Optional[int] = Field(default=None, description="Minimum revenue in JPY")
    min_net_income: Optional[int] = Field(default=None, description="Minimum net income in JPY")
    min_roe: Optional[float] = Field(default=None, description="Minimum ROE % (e.g. 10.0)")
    max_pe_ratio: Optional[float] = Field(default=None, description="Maximum PE ratio")
    limit: int = Field(default=20, description="Max results (max 100)")


class GetHealthScoreInput(BaseModel):
    code: str = Field(description="EDINET code or securities code")


class GetHealthRankingInput(BaseModel):
    sector: Optional[str] = Field(default=None, description="Optional sector filter")
    order: str = Field(default="desc", description="'desc' for healthiest, 'asc' for weakest")
    limit: int = Field(default=20, description="Max results (max 100)")


class GetPeersInput(BaseModel):
    code: str = Field(description="EDINET code or securities code")
    limit: int = Field(default=10, description="Max results (max 50)")


class GetTimeseriesInput(BaseModel):
    codes: list[str] = Field(description="List of 1-5 EDINET or securities codes")
    metric: str = Field(
        default="revenue",
        description=(
            "Metric: revenue, net_income, operating_income, total_assets, "
            "total_equity, eps, bps, dividends_per_share, operating_cf, "
            "investing_cf, financing_cf, roe, pe_ratio, num_employees"
        ),
    )
    years: int = Field(default=10, description="Number of years (max 20)")


class ListFilingsInput(BaseModel):
    company_code: Optional[str] = Field(default=None, description="Filter by company code")
    doc_type: Optional[str] = Field(
        default=None,
        description="Document type: 120=annual, 130=semi-annual, 140=quarterly",
    )
    limit: int = Field(default=20, description="Max results (max 100)")


class GetTranslationsInput(BaseModel):
    doc_id: str = Field(description="EDINET document ID (e.g. 'S100ABCD')")
    section: Optional[str] = Field(
        default=None,
        description=(
            "Section filter: mda, risk_factors, business_overview, "
            "governance, financial_notes, accounting_policy"
        ),
    )


class SearchTranslationsInput(BaseModel):
    query: str = Field(description="Search terms (e.g. 'semiconductor', 'risk factors')")
    section: Optional[str] = Field(default=None, description="Section filter")
    limit: int = Field(default=10, description="Max results (max 50)")


class GetFilingCalendarInput(BaseModel):
    month: str = Field(description="Month in YYYY-MM format (e.g. '2025-06')")


class SearchCompaniesBatchInput(BaseModel):
    queries: list[str] = Field(
        description="List of up to 10 company identifiers (codes or name fragments)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(data: Any) -> str:
    """Format API response as a compact JSON string for the LLM."""
    return json.dumps(data, ensure_ascii=False, default=str)


_HTTP_ERROR_HINTS: dict[int, str] = {
    401: "Invalid or missing API key. Check your AXIORA_API_KEY.",
    403: "Access denied. Your plan may not include this endpoint.",
    404: "Not found. Use axiora_search_companies to find the correct code.",
    429: "Rate limit exceeded. Wait a moment before retrying.",
}


def _handle_http_error(exc: httpx.HTTPStatusError) -> NoReturn:
    """Convert an HTTP error into a helpful ToolException message."""
    status = exc.response.status_code
    hint = _HTTP_ERROR_HINTS.get(status, "")
    try:
        body = exc.response.json()
        detail = body.get("detail", body.get("error", ""))
    except Exception:
        detail = exc.response.text[:200]
    parts = [f"Axiora API error {status}"]
    if detail:
        parts.append(str(detail))
    if hint:
        parts.append(hint)
    raise ToolException(". ".join(parts)) from exc


# ---------------------------------------------------------------------------
# Base tool with convenient init
# ---------------------------------------------------------------------------


class _AxioraBaseTool(BaseTool):
    """Base class that lets tools accept ``api_key`` directly.

    Any of these work::

        tool = MyTool(api_key="ax_live_...")       # direct key
        tool = MyTool()                             # reads AXIORA_API_KEY env var
        tool = MyTool(api=existing_wrapper)         # shared wrapper
    """

    api: AxioraAPIWrapper = Field(default=None)  # type: ignore[assignment]
    handle_tool_error: bool = True

    @model_validator(mode="before")
    @classmethod
    def _build_api_wrapper(cls, values: Any) -> Any:
        if isinstance(values, dict) and values.get("api") is None:
            kwargs: dict[str, Any] = {}
            api_key = values.pop("api_key", None)
            base_url = values.pop("base_url", None)
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            values["api"] = AxioraAPIWrapper(**kwargs)
        return values


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class SearchCompaniesTool(_AxioraBaseTool):
    name: str = "axiora_search_companies"
    description: str = (
        "Search for Japanese listed companies by name, securities code, or EDINET code. "
        "Use this to find a company's code before calling other tools. "
        "For looking up multiple companies at once, use axiora_search_companies_batch instead."
    )
    args_schema: type[BaseModel] = SearchCompaniesInput

    def _run(self, query: str, sector: str | None = None, limit: int = 10) -> str:
        try:
            return _fmt(self.api.request(
                "GET", "/companies/search", {"query": query, "sector": sector, "limit": limit}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, query: str, sector: str | None = None, limit: int = 10) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET", "/companies/search", {"query": query, "sector": sector, "limit": limit}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetCompanyTool(_AxioraBaseTool):
    name: str = "axiora_get_company"
    description: str = (
        "Get detailed info for a single Japanese company including name, sector, listing, "
        "and fiscal year end. Requires a company code — use axiora_search_companies first "
        "if you only have a name."
    )
    args_schema: type[BaseModel] = GetCompanyInput

    def _run(self, code: str) -> str:
        try:
            return _fmt(self.api.request("GET", f"/companies/{code}"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, code: str) -> str:
        try:
            return _fmt(await self.api.arequest("GET", f"/companies/{code}"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetFinancialsTool(_AxioraBaseTool):
    name: str = "axiora_get_financials"
    description: str = (
        "Get financial time series for a Japanese company. Returns revenue, net income, "
        "total assets, equity, ROE, ROA, and margins per fiscal year. "
        "Use this for detailed financial data. For growth rates, use axiora_get_growth instead."
    )
    args_schema: type[BaseModel] = GetFinancialsInput

    def _run(self, code: str, years: int = 5) -> str:
        try:
            return _fmt(
                self.api.request("GET", f"/companies/{code}/financials", {"years": years})
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, code: str, years: int = 5) -> str:
        try:
            return _fmt(
                await self.api.arequest(
                    "GET", f"/companies/{code}/financials", {"years": years}
                )
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetGrowthTool(_AxioraBaseTool):
    name: str = "axiora_get_growth"
    description: str = (
        "Get year-over-year growth rates and CAGRs for a Japanese company's financials. "
        "Use this when the question is about growth trends. "
        "For raw financial numbers, use axiora_get_financials instead."
    )
    args_schema: type[BaseModel] = GetGrowthInput

    def _run(self, code: str, years: int = 5) -> str:
        try:
            return _fmt(
                self.api.request("GET", f"/companies/{code}/growth", {"years": years})
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, code: str, years: int = 5) -> str:
        try:
            return _fmt(
                await self.api.arequest("GET", f"/companies/{code}/growth", {"years": years})
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetRankingTool(_AxioraBaseTool):
    name: str = "axiora_get_ranking"
    description: str = (
        "Rank Japanese companies by a financial metric (revenue, ROE, net income, etc.). "
        "Use this to find top/bottom companies by any metric. "
        "Optionally filter by sector."
    )
    args_schema: type[BaseModel] = GetRankingInput

    def _run(
        self,
        metric: str = "revenue",
        sector: str | None = None,
        order: str = "desc",
        limit: int = 20,
    ) -> str:
        try:
            return _fmt(self.api.request(
                "GET",
                f"/rankings/{metric}",
                {"sector": sector, "order": order, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(
        self,
        metric: str = "revenue",
        sector: str | None = None,
        order: str = "desc",
        limit: int = 20,
    ) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET",
                f"/rankings/{metric}",
                {"sector": sector, "order": order, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetSectorOverviewTool(_AxioraBaseTool):
    name: str = "axiora_get_sector_overview"
    description: str = (
        "List all sectors with company counts, or get aggregate financial stats for a "
        "specific sector. Use this to understand the Japanese market structure."
    )
    args_schema: type[BaseModel] = GetSectorOverviewInput

    def _run(self, sector: str | None = None) -> str:
        try:
            if sector:
                return _fmt(self.api.request("GET", f"/sectors/{sector}"))
            return _fmt(self.api.request("GET", "/sectors"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, sector: str | None = None) -> str:
        try:
            if sector:
                return _fmt(await self.api.arequest("GET", f"/sectors/{sector}"))
            return _fmt(await self.api.arequest("GET", "/sectors"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class CompareCompaniesTool(_AxioraBaseTool):
    name: str = "axiora_compare_companies"
    description: str = (
        "Compare financials of 2-5 Japanese companies side by side. "
        "Use this when directly comparing specific companies. "
        "For finding similar companies, use axiora_get_peers instead."
    )
    args_schema: type[BaseModel] = CompareCompaniesInput

    def _run(self, codes: list[str], years: int = 3) -> str:
        try:
            return _fmt(self.api.request(
                "GET", "/compare", {"codes": ",".join(codes), "years": years}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, codes: list[str], years: int = 3) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET", "/compare", {"codes": ",".join(codes), "years": years}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class ScreenCompaniesTool(_AxioraBaseTool):
    name: str = "axiora_screen_companies"
    description: str = (
        "Screen Japanese companies by financial criteria (sector, min revenue, min ROE, "
        "max PE ratio). All filters are combined with AND logic. "
        "Use this to find companies matching specific financial thresholds."
    )
    args_schema: type[BaseModel] = ScreenCompaniesInput

    def _run(
        self,
        sector: str | None = None,
        min_revenue: int | None = None,
        min_net_income: int | None = None,
        min_roe: float | None = None,
        max_pe_ratio: float | None = None,
        limit: int = 20,
    ) -> str:
        try:
            return _fmt(self.api.request(
                "GET",
                "/screen",
                {
                    "sector": sector,
                    "min_revenue": min_revenue,
                    "min_net_income": min_net_income,
                    "min_roe": min_roe,
                    "max_pe_ratio": max_pe_ratio,
                    "limit": limit,
                },
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(
        self,
        sector: str | None = None,
        min_revenue: int | None = None,
        min_net_income: int | None = None,
        min_roe: float | None = None,
        max_pe_ratio: float | None = None,
        limit: int = 20,
    ) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET",
                "/screen",
                {
                    "sector": sector,
                    "min_revenue": min_revenue,
                    "min_net_income": min_net_income,
                    "min_roe": min_roe,
                    "max_pe_ratio": max_pe_ratio,
                    "limit": limit,
                },
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetHealthScoreTool(_AxioraBaseTool):
    name: str = "axiora_get_health_score"
    description: str = (
        "Get the financial health score (0-100) for a Japanese company with component "
        "breakdown (stability, profitability, cash flow) and risk flags. "
        "Use this to assess a single company's financial health."
    )
    args_schema: type[BaseModel] = GetHealthScoreInput

    def _run(self, code: str) -> str:
        try:
            return _fmt(self.api.request("GET", f"/companies/{code}/health"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, code: str) -> str:
        try:
            return _fmt(await self.api.arequest("GET", f"/companies/{code}/health"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetHealthRankingTool(_AxioraBaseTool):
    name: str = "axiora_get_health_ranking"
    description: str = (
        "Rank Japanese companies by financial health score. "
        "Use this to find the healthiest or weakest companies, optionally within a sector."
    )
    args_schema: type[BaseModel] = GetHealthRankingInput

    def _run(
        self, sector: str | None = None, order: str = "desc", limit: int = 20
    ) -> str:
        try:
            return _fmt(self.api.request(
                "GET",
                "/rankings/health",
                {"sector": sector, "order": order, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(
        self, sector: str | None = None, order: str = "desc", limit: int = 20
    ) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET",
                "/rankings/health",
                {"sector": sector, "order": order, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetPeersTool(_AxioraBaseTool):
    name: str = "axiora_get_peers"
    description: str = (
        "Find peer companies in the same sector with similar revenue. "
        "Use this to discover competitors or comparable companies. "
        "For direct side-by-side comparison, use axiora_compare_companies instead."
    )
    args_schema: type[BaseModel] = GetPeersInput

    def _run(self, code: str, limit: int = 10) -> str:
        try:
            return _fmt(
                self.api.request("GET", f"/companies/{code}/peers", {"limit": limit})
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, code: str, limit: int = 10) -> str:
        try:
            return _fmt(
                await self.api.arequest("GET", f"/companies/{code}/peers", {"limit": limit})
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetTimeseriesTool(_AxioraBaseTool):
    name: str = "axiora_get_timeseries"
    description: str = (
        "Get time-series data for a financial metric across 1-5 companies. "
        "Returns chart-friendly format with fiscal_year and value per company. "
        "Use this when you need to plot or compare a single metric over time."
    )
    args_schema: type[BaseModel] = GetTimeseriesInput

    def _run(
        self, codes: list[str], metric: str = "revenue", years: int = 10
    ) -> str:
        try:
            return _fmt(self.api.request(
                "GET",
                "/timeseries",
                {"codes": ",".join(codes), "metric": metric, "years": years},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(
        self, codes: list[str], metric: str = "revenue", years: int = 10
    ) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET",
                "/timeseries",
                {"codes": ",".join(codes), "metric": metric, "years": years},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class ListFilingsTool(_AxioraBaseTool):
    name: str = "axiora_list_filings"
    description: str = (
        "List filings (annual, semi-annual, quarterly reports) with optional filters. "
        "Use this to find filing document IDs needed for axiora_get_translations."
    )
    args_schema: type[BaseModel] = ListFilingsInput

    def _run(
        self,
        company_code: str | None = None,
        doc_type: str | None = None,
        limit: int = 20,
    ) -> str:
        try:
            return _fmt(self.api.request(
                "GET",
                "/filings",
                {"company_code": company_code, "doc_type": doc_type, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(
        self,
        company_code: str | None = None,
        doc_type: str | None = None,
        limit: int = 20,
    ) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET",
                "/filings",
                {"company_code": company_code, "doc_type": doc_type, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetTranslationsTool(_AxioraBaseTool):
    name: str = "axiora_get_translations"
    description: str = (
        "Get English translations of a Japanese filing by document ID. "
        "Sections: mda, risk_factors, business_overview, governance, financial_notes, "
        "accounting_policy. Use axiora_list_filings first to find the doc_id."
    )
    args_schema: type[BaseModel] = GetTranslationsInput

    def _run(self, doc_id: str, section: str | None = None) -> str:
        try:
            return _fmt(self.api.request(
                "GET", f"/translations/{doc_id}", {"section": section}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, doc_id: str, section: str | None = None) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET", f"/translations/{doc_id}", {"section": section}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class SearchTranslationsTool(_AxioraBaseTool):
    name: str = "axiora_search_translations"
    description: str = (
        "Full-text search across English translations of Japanese filings. "
        "Returns matching sections with highlighted snippets. "
        "Use this to find what companies say about a topic (e.g. 'semiconductor', 'ESG')."
    )
    args_schema: type[BaseModel] = SearchTranslationsInput

    def _run(self, query: str, section: str | None = None, limit: int = 10) -> str:
        try:
            return _fmt(self.api.request(
                "GET",
                "/translations/search",
                {"query": query, "section": section, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, query: str, section: str | None = None, limit: int = 10) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET",
                "/translations/search",
                {"query": query, "section": section, "limit": limit},
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetFilingCalendarTool(_AxioraBaseTool):
    name: str = "axiora_get_filing_calendar"
    description: str = (
        "Get filing calendar for a month — shows how many filings were submitted per day. "
        "Use this to understand filing seasonality or find busy filing periods."
    )
    args_schema: type[BaseModel] = GetFilingCalendarInput

    def _run(self, month: str) -> str:
        try:
            return _fmt(self.api.request("GET", "/filings/calendar", {"month": month}))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, month: str) -> str:
        try:
            return _fmt(
                await self.api.arequest("GET", "/filings/calendar", {"month": month})
            )
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class SearchCompaniesBatchTool(_AxioraBaseTool):
    name: str = "axiora_search_companies_batch"
    description: str = (
        "Look up multiple companies at once (max 10). Accepts a mix of EDINET codes, "
        "securities codes, and name fragments. "
        "Use this instead of calling axiora_search_companies multiple times."
    )
    args_schema: type[BaseModel] = SearchCompaniesBatchInput

    def _run(self, queries: list[str]) -> str:
        try:
            return _fmt(self.api.request(
                "GET", "/companies/search", {"queries": ",".join(queries)}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self, queries: list[str]) -> str:
        try:
            return _fmt(await self.api.arequest(
                "GET", "/companies/search", {"queries": ",".join(queries)}
            ))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


class GetCoverageTool(_AxioraBaseTool):
    name: str = "axiora_get_coverage"
    description: str = (
        "Get data coverage statistics — total companies, filings, metric availability, "
        "and data freshness. Use this to understand what data is available before querying."
    )

    def _run(self) -> str:
        try:
            return _fmt(self.api.request("GET", "/coverage"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)

    async def _arun(self) -> str:
        try:
            return _fmt(await self.api.arequest("GET", "/coverage"))
        except httpx.HTTPStatusError as exc:
            return _handle_http_error(exc)


ALL_TOOLS: list[type[BaseTool]] = [
    SearchCompaniesTool,
    GetCompanyTool,
    GetFinancialsTool,
    GetGrowthTool,
    GetRankingTool,
    GetSectorOverviewTool,
    CompareCompaniesTool,
    ScreenCompaniesTool,
    GetHealthScoreTool,
    GetHealthRankingTool,
    GetPeersTool,
    GetTimeseriesTool,
    ListFilingsTool,
    GetTranslationsTool,
    SearchTranslationsTool,
    GetFilingCalendarTool,
    SearchCompaniesBatchTool,
    GetCoverageTool,
]
