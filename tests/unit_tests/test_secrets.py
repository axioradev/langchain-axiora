"""Verify API keys are never leaked in str/repr/serialization."""

from __future__ import annotations

import json

from pydantic import SecretStr

from langchain_axiora import AxioraToolkit
from langchain_axiora.api_wrapper import AxioraAPIWrapper
from langchain_axiora.tools import GetFinancialsTool, SearchCompaniesTool


SECRET = "ax_live_supersecretkey123"


def test_api_wrapper_uses_secret_str():
    api = AxioraAPIWrapper(api_key=SECRET)
    assert isinstance(api.api_key, SecretStr)


def test_api_wrapper_hides_key_in_repr():
    api = AxioraAPIWrapper(api_key=SECRET)
    assert SECRET not in repr(api)
    assert SECRET not in str(api)


def test_api_wrapper_hides_key_in_json():
    api = AxioraAPIWrapper(api_key=SECRET)
    dumped = api.model_dump_json()
    assert SECRET not in dumped


def test_api_wrapper_get_secret_value():
    api = AxioraAPIWrapper(api_key=SECRET)
    assert api.api_key.get_secret_value() == SECRET


def test_tool_hides_key_in_repr():
    tool = GetFinancialsTool(api_key=SECRET)
    assert SECRET not in repr(tool)
    assert SECRET not in str(tool)


def test_tool_hides_key_in_json():
    tool = SearchCompaniesTool(api_key=SECRET)
    dumped = json.dumps(tool.model_dump(), default=str)
    assert SECRET not in dumped


def test_toolkit_hides_key_in_repr():
    toolkit = AxioraToolkit(api_key=SECRET)
    assert SECRET not in repr(toolkit)
    assert SECRET not in str(toolkit)
