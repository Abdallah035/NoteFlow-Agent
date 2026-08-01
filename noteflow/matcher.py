"""Find which note the user means.

Turns a phrase like "the standup note" into a real note id, and decides
whether we have one clear match, several possibilities, or nothing.
"""

import re
import unicodedata
from datetime import datetime, timezone

# Arabic diacritics (tashkeel) - the small marks above and below letters.
# ـ is tatweel, a stretching character used only for decoration.
_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def normalize_text(text: str) -> str:
    """Make text comparable: lowercase, strip accents and Arabic spelling variants.

    The same Arabic word can be typed several ways. "الإجتماع" and "الاجتماع"
    are the same word, but different strings. We map both to one form so they
    can be compared. Only copies are normalized - stored notes keep their
    original spelling.
    """
    if not text:
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)

    # Alef variants -> plain alef
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Alef maqsura -> ya
    text = text.replace("ى", "ي")
    # Ta marbuta -> ha
    text = text.replace("ة", "ه")
    # Waw and ya carrying hamza -> plain forms
    text = text.replace("ؤ", "و").replace("ئ", "ي")

    return text.strip()


# Filler words that appear in almost every note, so they tell us nothing.
_ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
    "my", "me", "i", "you", "it", "this", "that", "these", "those",
    "note", "notes",
}

# Already written in normalized form (ta marbuta -> ha), because tokenize
# normalizes the text before it filters.
_ARABIC_STOPWORDS = {
    "من", "في", "علي", "الي", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "ما", "هو", "هي", "كان", "قد", "لم", "لا", "و", "او",
    "ملاحظه", "ملاحظات",
}

_STOPWORDS = _ENGLISH_STOPWORDS | _ARABIC_STOPWORDS


def _strip_arabic_prefix(word: str) -> str:
    """Remove 'ال' (the) from the front of an Arabic word.

    Arabic glues 'the' onto the word, so "السداد" and "سداد" are the same
    word written two ways. English never has this problem because "the" is
    a separate word we can drop as a stopword.
    """
    if len(word) > 4 and word.startswith("ال"):
        return word[2:]
    return word


def tokenize(text: str) -> list[str]:
    """Split text into comparable words, dropping filler words.

    \\w+ with re.UNICODE keeps Arabic letters. Using [a-z]+ here would return
    an empty list for every Arabic note, so they would never match anything.
    """
    words = re.findall(r"\w+", normalize_text(text), re.UNICODE)
    words = [_strip_arabic_prefix(w) for w in words]
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _same_word(a: str, b: str) -> bool:
    """True if two words look like the same word.

    Catches "meeting" and "meetings". Arabic broken plurals like
    "فاتورة" and "فواتير" are left to the embeddings in embeddings.py.
    """
    if a == b:
        return True
    return len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))


def _overlap(query_tokens: set, text_tokens: set) -> float:
    """What fraction of the query words appear in the text."""
    if not query_tokens or not text_tokens:
        return 0.0

    found = 0
    for query_word in query_tokens:
        for text_word in text_tokens:
            if _same_word(query_word, text_word):
                found += 1
                break

    return found / len(query_tokens)


def title_score(query: str, title: str) -> float:
    """How well the query matches a title. 1.0 is an exact match."""
    q_norm = normalize_text(query)
    t_norm = normalize_text(title)

    if not q_norm or not t_norm:
        return 0.0

    if q_norm == t_norm:
        return 1.0

    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(title))
    if not q_tokens or not t_tokens:
        return 0.0

    # What fraction of the words the user asked for did we find?
    overlap = _overlap(q_tokens, t_tokens)

    # A title that is the query plus a little extra is still a very good
    # match: "API" against "API migration".
    if q_norm in t_norm:
        return min(1.0, 0.8 + 0.2 * overlap)

    return overlap


def body_score(query: str, body: str) -> float:
    """How much of the query appears in the body.

    Divided by the query length, not the body length, so a long note is not
    punished for being long.
    """
    return _overlap(set(tokenize(query)), set(tokenize(body)))


def recency_score(created_at: str, now: datetime | None = None) -> float:
    """Newer notes score higher. 1.0 today, fading to 0.0 after 30 days.

    'now' can be passed in so tests do not depend on the real clock.
    """
    if not created_at:
        return 0.0

    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        # A corrupted or hand-edited timestamp must not break the search.
        return 0.0

    if now is None:
        # Our timestamps are stored without timezone info, so drop it here
        # too - subtracting naive from aware datetimes raises TypeError.
        now = datetime.now(timezone.utc).replace(tzinfo=None)

    days_old = (now - created).days

    if days_old < 0:
        # Clock skew or a hand-edited date. Treat it as brand new.
        return 1.0
    if days_old >= 30:
        return 0.0

    return 1.0 - (days_old / 30)


# How much each signal counts. They add up to 1.0.
WEIGHTS = {
    "exact_title": 0.35,
    "title_keyword": 0.20,
    "body_keyword": 0.15,
    "semantic": 0.20,
    "recency": 0.10,
}


def score_note(query: str, note: dict, semantic: float | None = None,
               now: datetime | None = None) -> float:
    """Score how well one note matches a query. Returns 0.0 to 1.0.

    'semantic' comes from the embeddings in stage 7. Leave it None when
    embeddings are switched off.
    """
    t_score = title_score(query, note["title"])

    parts = {
        # Two different questions: "is it exact?" and "how much overlap?".
        # An exact title collects both, which is what makes it dominate.
        "exact_title": 1.0 if t_score == 1.0 else 0.0,
        "title_keyword": t_score,
        "body_keyword": body_score(query, note["body"]),
        "recency": recency_score(note.get("created_at", ""), now),
    }

    weights = dict(WEIGHTS)

    if semantic is None:
        # Embeddings are off. Without this, the best possible score would be
        # 0.80, so a perfect match would fall below the 0.75 "one match"
        # threshold and the agent would ask which note the user meant.
        freed = weights.pop("semantic")
        total = sum(weights.values())
        for key in weights:
            weights[key] += freed * (weights[key] / total)
    else:
        parts["semantic"] = semantic

    return sum(parts[key] * weights[key] for key in parts)


# The three things the matcher can conclude.
ONE_MATCH = "one"
MULTIPLE = "multiple"
NONE = "none"

STRONG = 0.75      # above this we are confident
WEAK = 0.35        # below this it is not a real match
TIE_GAP = 0.10     # two scores this close are a tie, not a winner

# When keywords find nothing, we fall back to meaning alone. Real matches
# score about 0.4 and above, unrelated notes about 0.1, so this sits between.
SEMANTIC_ONLY = 0.35


def match(note_ref: str, notes: list[dict], semantic_scores: dict | None = None,
          now: datetime | None = None) -> tuple[str, list[dict]]:
    """Work out which note the user meant.

    Returns (outcome, candidates) where outcome is "one", "multiple" or "none".
    Candidates carry a "score" key so the caller can show it to the user.
    """
    if not notes:
        return NONE, []

    # "delete note 3" - an exact id always wins, no guessing needed.
    if note_ref.strip().isdigit():
        wanted = int(note_ref.strip())
        for note in notes:
            if note["id"] == wanted:
                return ONE_MATCH, [note]
        # No note with that id, so fall through and treat "3" as a search word.

    # First try with words only. Keyword matches are precise, so when they
    # work we trust them.
    scored = []
    for note in notes:
        score = score_note(note_ref, note, now=now)
        scored.append({**note, "score": score})

    scored.sort(key=lambda n: n["score"], reverse=True)
    good = [n for n in scored if n["score"] >= WEAK]

    # The words found nothing. Now try meaning instead, which can connect
    # "فواتير" to "فاتورة" or "delayed deployment" to "release postponed".
    if not good and semantic_scores:
        by_meaning = []
        for note in notes:
            score = semantic_scores.get(note["id"], 0.0)
            if score >= SEMANTIC_ONLY:
                by_meaning.append({**note, "score": score})

        by_meaning.sort(key=lambda n: n["score"], reverse=True)
        good = by_meaning

    if not good:
        return NONE, []

    best = good[0]

    # Anything close to the winner is a rival, even when the winner looks
    # strong. Scores of 0.80 / 0.78 / 0.77 mean "we cannot tell", so we ask
    # instead of deleting the wrong note.
    rivals = [n for n in good if best["score"] - n["score"] <= TIE_GAP]

    if len(rivals) > 1:
        return MULTIPLE, rivals[:5]

    if best["score"] >= STRONG:
        return ONE_MATCH, [best]

    return MULTIPLE, good[:5]
