<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo-light.svg">
    <img src="docs/logo-dark.svg" alt="Axiora" width="200">
  </picture>
</p>

<h1 align="center">langchain-axiora</h1>

<p align="center">
  <a href="https://pypi.org/project/langchain-axiora/"><img src="https://img.shields.io/pypi/v/langchain-axiora" alt="PyPI"></a>
  <a href="https://pypi.org/project/langchain-axiora/"><img src="https://img.shields.io/pypi/pyversions/langchain-axiora" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://x.com/axioradev"><img src="https://img.shields.io/badge/@axioradev-black?logo=x&logoColor=white" alt="X (Twitter)"></a>
</p>

LangChain tools for Japanese financial data — EDINET filings, XBRL, and company analysis.

Give your LangChain / LangGraph agents access to 4,000+ Japanese listed companies: financials from EDINET filings, health scores, screening, and English translations of annual reports. Powered by the [Axiora API](https://axiora.dev).

## Install

```bash
uv add langchain-axiora
# or
pip install langchain-axiora
```

## Quick Start

```python
from langchain_axiora import AxioraToolkit
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

toolkit = AxioraToolkit(api_key="ax_live_...")
agent = create_react_agent(ChatAnthropic(model="claude-sonnet-4-20250514"), toolkit.get_tools())

result = agent.invoke({
    "messages": [{"role": "user", "content": "Compare Toyota and Honda financials"}]
})
```

## Configuration

The API key can be provided three ways (in priority order):

```python
# 1. Explicit
toolkit = AxioraToolkit(api_key="ax_live_...")

# 2. Environment variable
import os
os.environ["AXIORA_API_KEY"] = "ax_live_..."
toolkit = AxioraToolkit()

# 3. .env file (with python-dotenv)
from dotenv import load_dotenv
load_dotenv()
toolkit = AxioraToolkit()
```

## Using a Subset of Tools

18 tools can be noisy for simple agents. Pass `selected_tools` to limit:

```python
toolkit = AxioraToolkit(
    api_key="ax_live_...",
    selected_tools=[
        "axiora_search_companies",
        "axiora_get_financials",
        "axiora_compare_companies",
    ],
)
# Invalid names raise ValueError with the valid list
```

## Using Individual Tools

Tools accept `api_key` directly — no wrapper needed:

```python
from langchain_axiora import GetFinancialsTool

tool = GetFinancialsTool(api_key="ax_live_...")
result = tool.invoke({"code": "7203", "years": 3})

# Or use AXIORA_API_KEY env var
tool = GetFinancialsTool()

# Async
result = await tool.ainvoke({"code": "7203"})
```

## Available Tools (18)

| Tool | Description |
|------|-------------|
| `axiora_search_companies` | Search companies by name or code |
| `axiora_get_company` | Get detailed company info |
| `axiora_get_financials` | Revenue, income, assets, ROE over time |
| `axiora_get_growth` | YoY growth rates and CAGRs |
| `axiora_get_ranking` | Rank companies by any metric |
| `axiora_get_sector_overview` | Sector stats and company counts |
| `axiora_compare_companies` | Side-by-side comparison (2-5 companies) |
| `axiora_screen_companies` | Filter by revenue, ROE, PE ratio, sector |
| `axiora_get_health_score` | Financial health score (0-100) |
| `axiora_get_health_ranking` | Rank by health score |
| `axiora_get_peers` | Find similar companies in same sector |
| `axiora_get_timeseries` | Chart-ready metric time series |
| `axiora_list_filings` | List annual/quarterly filings |
| `axiora_get_translations` | English translations of filings |
| `axiora_search_translations` | Full-text search across translations |
| `axiora_get_filing_calendar` | Filing activity per day |
| `axiora_search_companies_batch` | Bulk company lookup |
| `axiora_get_coverage` | Data coverage statistics |

## Retriever

Search English translations of Japanese filings as LangChain Documents:

```python
from langchain_axiora import AxioraRetriever

retriever = AxioraRetriever(api_key="ax_live_...", section="risk_factors", k=5)
docs = retriever.invoke("semiconductor supply chain risk")

for doc in docs:
    print(doc.metadata["company_name"], doc.page_content[:100])
```

Works in any LangChain chain or RAG pipeline that accepts a retriever.

## Error Handling

All tools use `handle_tool_error=True`. When the API returns an error, the agent receives a helpful message instead of crashing:

```
Axiora API error 404. Company not found. Use axiora_search_companies to find the correct code.
```

This lets the agent self-correct (e.g., search for the right code, then retry).

## Get an API Key

Sign up at [axiora.dev](https://axiora.dev) — free, no credit card.

[Docs](https://axiora.dev/docs) · [PyPI](https://pypi.org/project/langchain-axiora/)
