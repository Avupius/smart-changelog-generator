"""
summarizer.py — Async GPT-4o-mini summarizer.

Groups classified commits by category and generates human-readable
summaries concurrently for all non-empty categories.
"""

import asyncio
import os
from typing import Optional

from openai import AsyncOpenAI

SYSTEM_PROMPT = """You are a technical writer creating professional software changelogs.
Your task: summarize a list of related Git commits into 2–5 concise bullet points.

Rules:
- Write in the language specified by the user.
- Be specific and technical — mention function names, components, or modules if inferable.
- Start each bullet with a capital letter and a dash (- ).
- No filler phrases like "This commit...", "The developer...", "Changes were made...".
- Maximum 5 bullet points. Minimum 2.
- No trailing period after the last bullet.
"""

USER_TEMPLATE = """Category: {category}
Output language: {language}

Commit messages:
{messages}

Write 2–5 bullet points summarizing what changed in this category."""


async def _summarize_category(
    client: AsyncOpenAI,
    category: str,
    commits: list[str],
    language: str,
) -> str:
    messages_str = "\n".join(f"- {m}" for m in commits[:50])  # cap at 50 to avoid token overflow
    user_msg = USER_TEMPLATE.format(
        category=category,
        language=language,
        messages=messages_str,
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"- Summary unavailable ({e})"


async def summarize_all(
    categorized: dict[str, list[str]],
    language: str = "English",
    openai_api_key: Optional[str] = None,
) -> dict[str, str]:
    """
    Summarize all non-empty categories concurrently.

    Args:
        categorized: dict mapping category name → list of commit messages
        language: Output language for the summaries
        openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)

    Returns:
        dict mapping category name → markdown bullet-point summary string
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = AsyncOpenAI(api_key=api_key)
    non_empty = {cat: msgs for cat, msgs in categorized.items() if msgs}

    if not non_empty:
        return {}

    tasks = [
        _summarize_category(client, cat, msgs, language)
        for cat, msgs in non_empty.items()
    ]
    results = await asyncio.gather(*tasks)

    return dict(zip(non_empty.keys(), results))
