"""Extract player identities from OCR text already read for an Apex event."""
from __future__ import annotations

import re
import unicodedata


GENERIC_WORDS = frozenset({
    "ASISTENCIA", "DERRIBADO", "DERRIBASTE", "DOWNED", "KNOCKED",
    "ELIMINADO", "ELIMINADA", "ELIMINASTE", "ELIMINACION", "ELIMINATED",
    "ELIMINATION", "DESANGRADO", "DESANGRO", "BLEED", "OUT", "SQUAD",
    "ESCUADRON", "WIPE", "PLAYER", "KILLFEED",
})


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.upper())
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.sub(r"[^A-Z0-9]+", " ", plain).strip()


def readable_name(tokens: list[str]) -> str | None:
    """Return a plausible name, never a bare event word or score."""
    while tokens and tokens[-1].isdigit() and len(tokens[-1]) <= 4:
        tokens = tokens[:-1]
    if not tokens or any(token in GENERIC_WORDS or len(token) < 2 for token in tokens):
        return None
    if not any(any(character.isalpha() for character in token) for token in tokens):
        return None
    if sum(map(len, tokens)) < 4:
        return None
    return " ".join(tokens)


def center_victim(text: str, alias: str, alias_index: int) -> str | None:
    normalized = normalize_text(text)
    start = len(normalized[:alias_index].split()) + len(normalize_text(alias).split())
    return readable_name(normalized.split()[start:])


def killfeed_victim(text: str, actor_end: int, alias: str | None = None, alias_index: int = 0) -> str | None:
    normalized = normalize_text(text)
    if alias is None:
        suffix = normalized[actor_end:].split()
        # Standard rows have a weapon/icon between the actor and the victim.
        # OCR may omit the icon entirely, leaving just the victim name.
        if len(suffix) > 1 and not (suffix[-1].isdigit() and len(suffix[-1]) <= 4 and suffix[-2].isalpha()):
            candidate = suffix[1:]
        else:
            candidate = suffix
        if len(candidate) == 1 and re.fullmatch(r"R\d{2,3}", candidate[0]):
            return None
        # Killfeed names can end in digits; punctuation from OCR may split
        # those digits into a separate token, unlike a center score popup.
        if len(candidate) > 1 and candidate[-1].isdigit() and candidate[-2].isalpha():
            candidate = candidate[:-2] + [candidate[-2] + candidate[-1]]
    else:
        start = len(normalized[:alias_index].split()) + len(normalize_text(alias).split())
        candidate = normalized.split()[start:]
    return readable_name(candidate)
