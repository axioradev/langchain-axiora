"""LangChain retriever for searching English translations of Japanese filings."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr

from langchain_axiora.api_wrapper import AxioraAPIWrapper


class AxioraRetriever(BaseRetriever):
    """Retriever that searches English translations of Japanese EDINET filings.

    Setup:
        Install ``langchain-axiora`` and set environment variable ``AXIORA_API_KEY``.

        .. code-block:: bash

            pip install langchain-axiora
            export AXIORA_API_KEY="ax_live_..."

    Key init args:
        api_key: str
            Axiora API key. Reads from ``AXIORA_API_KEY`` env var if not provided.
        section: str | None
            Optional section filter: mda, risk_factors, business_overview,
            governance, financial_notes, accounting_policy.
        k: int
            Max results to return (default 10, max 50).

    Instantiate:
        .. code-block:: python

            from langchain_axiora import AxioraRetriever

            retriever = AxioraRetriever()

    Invoke:
        .. code-block:: python

            docs = retriever.invoke("semiconductor supply chain risk")
            for doc in docs:
                print(doc.metadata["company_name"], doc.page_content[:100])

    Use in a chain:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough

            llm = ChatAnthropic(model="claude-sonnet-4-20250514")
            prompt = ChatPromptTemplate.from_template(
                "Based on these Japanese filing excerpts:\\n{context}\\n\\nAnswer: {question}"
            )
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
            )
            result = chain.invoke("What are the main risks for Japanese semiconductor companies?")
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
    section: str | None = Field(
        default=None,
        description="Section filter: mda, risk_factors, business_overview, etc.",
    )
    k: int = Field(default=10, description="Max results (max 50)")

    _api: AxioraAPIWrapper | None = None

    model_config = {"populate_by_name": True}

    @property
    def _wrapper(self) -> AxioraAPIWrapper:
        if self._api is None:
            self._api = AxioraAPIWrapper(api_key=self.api_key, base_url=self.base_url)
        return self._api

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        params: dict[str, Any] = {"q": query, "limit": self.k}
        if self.section:
            params["section"] = self.section
        try:
            result = self._wrapper.request("GET", "/translations/search", params)
        except httpx.HTTPStatusError:
            return []
        return self._to_documents(result)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        params: dict[str, Any] = {"q": query, "limit": self.k}
        if self.section:
            params["section"] = self.section
        try:
            result = await self._wrapper.arequest("GET", "/translations/search", params)
        except httpx.HTTPStatusError:
            return []
        return self._to_documents(result)

    @staticmethod
    def _to_documents(result: Any) -> list[Document]:
        docs: list[Document] = []
        items = result.get("data", []) if isinstance(result, dict) else []
        for item in items:
            content = item.get("content", item.get("snippet", ""))
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ("content", "snippet") and v is not None
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
