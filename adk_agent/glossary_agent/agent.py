"""Pydantic AI 版と指示・ツール・モデルを共有する ADK エージェント。

起動（リポジトリ直下）:
    PYTHONPATH=. uv run --with 'google-adk>=2,<3' adk run adk_agent/glossary_agent
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
