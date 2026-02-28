"""Axiora toolkit — returns all Axiora tools configured with a shared API client."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr, model_validator

from langchain_axiora.api_wrapper import AxioraAPIWrapper
from langchain_axiora.tools import ALL_TOOLS

_VALID_TOOL_NAMES: frozenset[str] = frozenset(
    tool_cls.model_fields["name"].default for tool_cls in ALL_TOOLS
)


class AxioraToolkit(BaseToolkit):
    """LangChain toolkit for Japanese financial data via Axiora.

    Setup:
        Install ``langchain-axiora`` and set environment variable ``AXIORA_API_KEY``.

        .. code-block:: bash

            pip install langchain-axiora
            export AXIORA_API_KEY="ax_live_..."

    Key init args:
        api_key: str
            Axiora API key. Reads from ``AXIORA_API_KEY`` env var if not provided.
        selected_tools: list[str] | None
            Optional subset of tool names to include (default: all 18).

    Instantiate:
        .. code-block:: python

            from langchain_axiora import AxioraToolkit

            toolkit = AxioraToolkit()
            tools = toolkit.get_tools()

    Use within an agent:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            from langgraph.prebuilt import create_react_agent

            llm = ChatAnthropic(model="claude-sonnet-4-20250514")
            agent = create_react_agent(llm, tools)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Compare Toyota and Honda"}]}
            )

    Use a subset of tools:
        .. code-block:: python

            toolkit = AxioraToolkit(
                selected_tools=["axiora_search_companies", "axiora_get_financials"],
            )
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

    @model_validator(mode="after")
    def _validate_selected_tools(self) -> "AxioraToolkit":
        if self.selected_tools is not None:
            invalid = set(self.selected_tools) - _VALID_TOOL_NAMES
            if invalid:
                raise ValueError(
                    f"Invalid tool names: {sorted(invalid)}. "
                    f"Valid names: {sorted(_VALID_TOOL_NAMES)}"
                )
        return self

    def get_tools(self) -> list[BaseTool]:
        """Return Axiora tools configured with the shared API wrapper."""
        api = AxioraAPIWrapper(api_key=self.api_key, base_url=self.base_url)
        all_tools = [tool_cls(api=api) for tool_cls in ALL_TOOLS]
        if self.selected_tools is not None:
            selected = set(self.selected_tools)
            return [t for t in all_tools if t.name in selected]
        return all_tools
