"""
summarizer.py — Asynchroner GPT-4o-mini basierter Text-Summarizer für Commit-Kategorien.
Nimmt klassifizierte und gruppierte Commit-Messages entgegen und generiert Bullet-Point-Zusammenfassungen für jede Kategorie.
"""

import asyncio
import os
from typing import Optional

from openai import AsyncOpenAI

# System-Prompt definiert die Rolle und Verhaltensregeln für das GPT-Modell.
# Wird bei jedem API-Aufruf mitgesendet und gilt für alle Kategorien.
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


async def _summarize_category(client: AsyncOpenAI, category: str, commits: list[str], language: str,) -> str:
    """
    Generiert eine Zusammenfassung für eine einzelne Kategorie via GPT-4o-mini.

    Wird von summarize_all() für alle Kategorien parallel aufgerufen. 
    Bei Fehlern wird eine Fehlermeldung zurückgegeben, damit die anderen Kategorien trotzdem zusammengefasst werden können.

    Args:
        client: Geteilte AsyncOpenAI-Instanz (wird für alle Kategorien wiederverwendet)
        category: Kategoriename (z.B. "Features", "Bugfixes")
        commits: Liste von Commit-Messages in dieser Kategorie
        language: Gewünschte Ausgabesprache

    Returns:
        Markdown-formatierte Bullet-Points als String
    """

    # Auf maximal 100 Commits pro Kategorie beschränken, um Token-Limit zu vermeiden
    messages_str = "\n".join(f"- {m}" for m in commits[:100])  
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
            temperature=0.3, # Niedrige Temperatur für konsistente, sachliche Zusammenfassungen
            max_tokens=500, # Tokens beschränken, um Kosten zu kontrollieren (ca. 2-5 Bullet Points)
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fehler abfangen, damit die Zusammenfassung der anderen Kategorien nicht verloren geht und eine verständliche Fehlermeldung zurückgeben
        return f"- Summary unavailable ({e})"


async def summarize_all(categorized: dict[str, list[str]], language: str = "English", openai_api_key: Optional[str] = None,) -> dict[str, str]:
    """
    Fasst alle nicht leeren Kategorien parallel zusammen und gibt ein Dictionary mit Kategorie -> Zusammenfassung zurück.
    Erwartet bereits klassifizierte Commit-Messages als Input.
    Parallelisierung erfolgt über asyncio.gather, um die Gesamtzeit bei mehreren Kategorien zu reduzieren.

    Args:
        categorized: Dict mit Kategorie-Namen als Keys und Listen von Commit-Messages als Values
        language: Gewünschte Ausgabesprache für die Zusammenfassungen
        openai_api_key: OpenAI API Key
    
    Returns: 
        Dict mit Kategorie-Namen als Keys und den generierten Bullet-Point-Zusammenfassungen als Values

    Raises: ValueError: Wenn kein API-Key bereitgestellt oder in der Umgebung gefunden wird.
    """

    # API-Key aus Argument oder Umgebungsvariable laden
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    # Gemeinsame AsyncOpenAI-Instanz für alle API-Aufrufe erstellen (wird in _summarize_category wiederverwendet)
    # AsyncOpenAI ermöglicht parallele API-Aufrufe, was die Gesamtzeit bei mehreren Kategorien deutlich reduziert.
    client = AsyncOpenAI(api_key=api_key)
    non_empty = {cat: msgs for cat, msgs in categorized.items() if msgs}

    if not non_empty:
        return {}

    # Für jede Kategorie wird _summarize_category parallel aufgerufen, Ergebnisse werden in einem Dictionary zurückgegeben
    tasks = [
        _summarize_category(client, cat, msgs, language)
        for cat, msgs in non_empty.items()
    ]

    # Alle Task gleichzeitig starten und auf alle Ergebnisse warten, Rückgabe in der gleichen Reihenfolge wie die Kategorien
    results = await asyncio.gather(*tasks)

    return dict(zip(non_empty.keys(), results))
