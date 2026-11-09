"""google-adk は一時追加依存。通常のpytestでは未導入ならスキップする。"""

import pytest

pytest.importorskip("google.adk")

from adk_agent.glossary_agent.agent import root_agent
from app.agent import INSTRUCTIONS
from app.config import MODEL
from app import tools


def test_adk_reuses_exact_instruction_model_and_functions():
    assert root_agent.instruction == INSTRUCTIONS
    assert root_agent.model == MODEL
    assert root_agent.name == "glossary_agent"
    functions = {
        getattr(tool, "__name__", None)
        for tool in root_agent.tools
    }
    assert functions == {"search_glossary", "get_lecture_summary", "make_quiz"}


def test_adk_tools_are_original_functions():
    assert tools.search_glossary in root_agent.tools
    assert tools.get_lecture_summary in root_agent.tools
    assert tools.make_quiz in root_agent.tools
