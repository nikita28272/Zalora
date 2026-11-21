"""Свежие футбольные новости из открытых RSS-лент (бесплатно)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import feedparser
import httpx

from .teams import normalize

log = logging.getLogger(__name__)

FEEDS = (
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.theguardian.com/football/rss",
    "https://www.skysports.com/rss/12040",
    "https://www.espn.com/espn/rss/soccer/news",
)


@dataclass
class Headline:
    title: str
    summary: str
    link: str
    source: str


async def _fetch_feed(client: httpx.AsyncClient, url: str) -> list[Headline]:
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.info("RSS %s недоступна: %s", url, type(exc).__name__)
        return []
    parsed = feedparser.parse(resp.text)
    source = parsed.feed.get("title", url)
    return [
        Headline(
            title=e.get("title", ""),
            summary=(e.get("summary", "") or "")[:300],
            link=e.get("link", ""),
            source=source,
        )
        for e in parsed.entries
    ]


async def fetch_all() -> list[Headline]:
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "Mozilla/5.0 (betting-signals-bot)"}) as client:
        chunks = await asyncio.gather(*(_fetch_feed(client, url) for url in FEEDS))
    return [h for chunk in chunks for h in chunk]


def _keywords(team: str) -> set[str]:
    # значимые слова названия: «manchester city» -> {"manchester city", "city"} не берём,
    # используем полное нормализованное имя и самое длинное слово
    norm = normalize(team)
    words = [w for w in norm.split() if len(w) >= 4]
    keys = {norm}
    if words:
        keys.add(max(words, key=len))
    return keys


def for_match(headlines: list[Headline], home: str, away: str, limit: int = 15) -> list[Headline]:
    """Заголовки, где упоминается хотя бы одна из команд."""
    keys = _keywords(home) | _keywords(away)
    picked = []
    for h in headlines:
        text = normalize(f"{h.title} {h.summary}")
        if any(k and k in text for k in keys):
            picked.append(h)
        if len(picked) >= limit:
            break
    return picked
