"""Сопоставление названий команд между разными источниками.

The Odds API пишет «Bayern Munich», football-data.org — «FC Bayern München».
Нормализуем названия и ищем лучшее совпадение.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

# Служебные слова, которые не помогают различать клубы.
_STOPWORDS = {
    "fc", "cf", "afc", "ac", "as", "ss", "sc", "ssc", "us", "ogc", "rc", "rcd", "ud", "cd", "sd",
    "vfb", "vfl", "tsg", "sv", "bv", "fk", "sk", "club", "de", "calcio", "football", "futbol",
    "hotspur", "1899", "1846", "1848", "1860", "1900", "1901", "1904", "1907", "1909", "1910",
    "1913", "olympique", "stade", "losc", "ca", "bsc", "and", "the",
}

# Перевод английских вариантов к «родным» названиям.
_WORD_ALIASES = {
    "munich": "munchen",
    "cologne": "koln",
    "nuremberg": "nurnberg",
    "inter": "internazionale",
    "atletico": "atletico",
    "psg": "paris saint germain",
    "wolves": "wolverhampton wanderers",
    "spurs": "tottenham",
    "utd": "united",
    "man": "manchester",
    "gladbach": "monchengladbach",
    "leverkusen": "leverkusen",
    "frankfurt": "frankfurt",
    "betis": "betis",
    "la": "los angeles",  # НБА: «LA Clippers» (ESPN) = «Los Angeles Clippers» (линии)
}

# Полные алиасы для особых случаев (после нормализации).
_FULL_ALIASES = {
    "milan": "milan",
    "inter milan": "internazionale",
    "internazionale milan": "internazionale",
    "internazionale milano": "internazionale",
    "paris saint germain": "paris saint germain",
    "bayern munchen": "bayern munchen",
    "borussia monchengladbach": "monchengladbach",
    "brighton hove albion": "brighton",
    "brighton": "brighton",
    "west ham united": "west ham",
    "athletic bilbao": "athletic",
    "athletic club": "athletic",
    "atletico madrid": "atletico madrid",
    "club atletico de madrid": "atletico madrid",
    "rb leipzig": "leipzig",
    "rasenballsport leipzig": "leipzig",
}


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = text.replace("&", " and ").replace("-", " ").replace(".", " ").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = []
    for w in text.split():
        w = _WORD_ALIASES.get(w, w)
        words.extend(w.split())
    words = [w for w in words if w not in _STOPWORDS and not w.isdigit()]
    norm = " ".join(words)
    return _FULL_ALIASES.get(norm, norm)


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    # одно название целиком содержится в другом: «tottenham» ⊂ «tottenham»
    if ta <= tb or tb <= ta:
        return 0.9
    jaccard = len(ta & tb) / len(ta | tb)
    ratio = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, ratio * 0.85)


def best_match(name: str, candidates: Iterable[str], threshold: float = 0.72) -> str | None:
    """Возвращает кандидата с наибольшим сходством или None, если уверенности нет."""
    best, best_score = None, 0.0
    for cand in candidates:
        score = similarity(name, cand)
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= threshold else None
