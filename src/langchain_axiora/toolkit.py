"""Axiora toolkit — returns all Axiora tools configured with a shared API client."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr

from langchain_axiora.api_wrapper import AxioraAPIWrapper
from langchain_axiora.tools import ALL_TOOLS


class AxioraToolkit(BaseToolkit):
    """LangChain toolkit for Japanese financial data via Axiora.

    Usage::

        from langchain_axiora import AxioraToolkit

        toolkit = AxioraToolkit(api_key="ax_live_...")
        tools = toolkit.get_tools()

        # Pass to any LangChain agent
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(model, tools)

        # Use a subset of tools (18 total can be noisy for simple agents)
        toolkit = AxioraToolkit(api_key="ax_live_...", selected_tools=["axiora_search_companies", "axiora_get_financials"])
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
    base_url: str = Field(default="https://api.axiora.dev/v1")
    selected_tools: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of tool names to include. "
            "If None, all 18 tools are returned."
        ),
    )

    model_config = {"populate_by_name": True}

    def get_tools(self) -> list[BaseTool]:
        """Return Axiora tools configured with the shared API wrapper."""
        api = AxioraAPIWrapper(api_key=self.api_key, base_url=self.base_url)
        all_tools = [tool_cls(api=api) for tool_cls in ALL_TOOLS]
        if self.selected_tools is not None:
            return [t for t in all_tools if t.name in self.selected_tools]
        return all_tools
