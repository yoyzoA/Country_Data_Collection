"""
Semantic-aware edit costs for Wikipedia country infobox trees.

The first implementation treated leaf values mostly as plain strings, so an
update such as 1943 -> 1946 cost the same as 1943 -> 1776.  This module keeps
TED as the core algorithm, but refines UPDATE costs so that the cost reflects
both:

1. structural compatibility: values are only compared inside compatible fields
   and subsections, e.g. Population/Total with Population/Total, not
   Area/Total with GDP/Total;
2. semantic closeness: compatible numbers/dates/percentages receive graded
   costs in [0, 1].

A cost of 2 is reserved for incompatible alignments. Since delete + insert also
costs 2, the backtracker will prefer representing those cases as a deletion and
an insertion instead of a misleading UPDATE operation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, Optional, Set


INCOMPATIBLE_UPDATE_COST = 2.0


@dataclass(frozen=True)
class ParsedNumber:
    value: float
    kind: str = "number"  # number | percentage | ordinal | magnitude_word


@dataclass(frozen=True)
class ParsedDatePart:
    value: int
    kind: str  # year | month | day | iso_date


_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MAGNITUDE_WORDS = {
    "thousand": 1_000.0,
    "k": 1_000.0,
    "million": 1_000_000.0,
    "m": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "tn": 1_000_000_000_000.0,
}

_STOP_LABEL_WORDS = {
    "and", "of", "the", "a", "an", "as", "per", "by", "in", "on", "to",
    "estimated", "estimate", "est", "current", "total", "all", "rank",
}

# Broad field classes. These are deliberately used for compatibility, not as
# external ground truth. The comparison remains generic TED; this only prevents
# meaningless updates across unrelated infobox fields/subsections.
_FIELD_CATEGORY_KEYWORDS = {
    "population": {"population", "census", "density", "inhabitants"},
    "area": {"area", "land", "water", "km", "sq", "metropolitan"},
    "gdp": {"gdp", "ppp", "nominal", "capita"},
    "gini": {"gini"},
    "hdi": {"hdi"},
    "currency": {"currency"},
    "timezone": {"time", "zone", "utc", "dst", "summer"},
    "code": {"code", "calling", "iso", "tld", "internet"},
    "language": {"language", "languages", "vernacular"},
    "capital": {"capital", "city"},
    "demonym": {"demonym"},
    "religion": {"religion", "christianity", "islam", "judaism", "druzism"},
    "ethnic": {"ethnic", "ethnicity", "race", "origin"},
    "driving": {"driving"},
    "legislature": {"legislature", "parliament", "congress", "senate", "house", "assembly", "council"},
    "government_official": {"president", "minister", "chancellor", "speaker", "monarch", "king", "queen", "emir", "premier"},
    "government_system": {"government", "republic", "monarchy", "democracy", "federal", "unitary", "confederation"},
    "date_history": {
        "date", "independence", "declaration", "declared", "recognition", "recognised",
        "constitution", "founded", "foundation", "established", "formation",
        "unification", "reunification", "restoration", "mandate", "withdrawal",
        "annexation", "federation", "confederation", "state", "treaty", "westphalia",
    },
}

_HISTORY_SUBCATEGORY_KEYWORDS = {
    "independence": {"independence", "declaration", "declared"},
    "constitution": {"constitution"},
    "recognition": {"recognition", "recognised", "recognized"},
    "formation": {"foundation", "founded", "formation", "established", "federal", "state", "federation", "confederation", "unification", "reunification"},
    "colonial_end": {"mandate", "withdrawal", "withdrawn", "troops", "forces", "ended"},
    "treaty": {"treaty", "westphalia", "restoration", "annexation"},
}

_NUMERIC_ALLOWED_CATEGORIES = {
    "population", "area", "gdp", "gini", "hdi", "ethnic", "religion", "date_history"
}

_NUMERIC_TEXT_ONLY_CATEGORIES = {"code"}


def _clean_text(text: Any) -> str:
    return str(text or "").strip()


@lru_cache(maxsize=None)
def normalize_label(label: Any) -> str:
    text = _clean_text(label).lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[^a-z0-9%+.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def label_tokens(label: Any) -> Set[str]:
    return {
        tok
        for tok in normalize_label(label).split()
        if tok and tok not in _STOP_LABEL_WORDS
    }


@lru_cache(maxsize=None)
def label_similarity(a: Any, b: Any) -> float:
    """Lexical similarity in [0, 1] for labels/text."""
    na = normalize_label(a)
    nb = normalize_label(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    ta = label_tokens(na)
    tb = label_tokens(nb)
    jaccard = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    sequence = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, sequence)


@lru_cache(maxsize=None)
def field_category(label_or_context: Any) -> Optional[str]:
    """Map a field label or full subsection context to a broad category."""
    tokens = label_tokens(label_or_context)
    if not tokens:
        return None
    # Specific high-value categories first so e.g. "GDP per capita" is GDP, not generic area/capita.
    for category, keywords in _FIELD_CATEGORY_KEYWORDS.items():
        if tokens & keywords:
            return category
    return None


@lru_cache(maxsize=None)
def history_subcategory(label_or_context: Any) -> Optional[str]:
    tokens = label_tokens(label_or_context)
    if not tokens:
        return None
    for category, keywords in _HISTORY_SUBCATEGORY_KEYWORDS.items():
        if tokens & keywords:
            return category
    return None


def _context(node: Dict[str, Any]) -> str:
    """
    Context used for structural compatibility.

    Examples:
      attribute Total under Area       -> "Area / Total"
      attribute Total under GDP (PPP)  -> "GDP ( PPP ) / Total"
      token under Population/Density   -> "Population / Density"
    """
    value = node.get("semantic_context")
    if value:
        return str(value)

    ancestors = list(node.get("ancestor_labels") or [])
    if node.get("node_type") != "token":
        ancestors.append(str(node.get("label", "")))
    return " / ".join(x for x in ancestors if x)


def _parent_context(node: Dict[str, Any]) -> str:
    value = node.get("parent_context")
    if value:
        return str(value)
    parent = node.get("parent_label")
    return str(parent or "")


def _contexts_compatible(ctx_a: Any, ctx_b: Any, *, require_history_subtype: bool = True) -> bool:
    """Return True only when two fields/subsections are safe to compare."""
    na = normalize_label(ctx_a)
    nb = normalize_label(ctx_b)
    if na == nb:
        return True

    cat_a = field_category(ctx_a)
    cat_b = field_category(ctx_b)

    if cat_a is not None and cat_b is not None:
        if cat_a != cat_b:
            return False
        if cat_a == "date_history" and require_history_subtype:
            sub_a = history_subcategory(ctx_a)
            sub_b = history_subcategory(ctx_b)
            if sub_a is not None and sub_b is not None and sub_a != sub_b:
                # Avoid misleading updates such as "French mandate ended" -> "Federal state".
                return False
        return True

    # If categories are unknown, only allow a label update when labels are clearly similar.
    return label_similarity(ctx_a, ctx_b) >= 0.72


@lru_cache(maxsize=None)
def _parse_date_part(text: Any) -> Optional[ParsedDatePart]:
    raw = _clean_text(text).lower().strip("()[]{}.,;:")
    raw = raw.replace("−", "-").replace("–", "-").replace("—", "-")

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", raw)
    if m:
        year, month, day = map(int, m.groups())
        return ParsedDatePart(year * 365 + month * 30 + day, "iso_date")

    if raw in _MONTHS:
        return ParsedDatePart(_MONTHS[raw], "month")

    m = re.fullmatch(r"(?:c\.?\s*)?(\d{3,4})", raw)
    if m:
        year = int(m.group(1))
        if 500 <= year <= 2100:
            return ParsedDatePart(year, "year")

    m = re.fullmatch(r"\d{1,2}", raw)
    if m:
        value = int(raw)
        if 1 <= value <= 31:
            return ParsedDatePart(value, "day")

    return None


@lru_cache(maxsize=None)
def _parse_number_basic(text: Any) -> Optional[ParsedNumber]:
    raw = _clean_text(text).lower().strip()
    if not raw:
        return None

    cleaned = raw.replace(",", "")
    cleaned = cleaned.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.strip("()[]{}")

    if cleaned in _MAGNITUDE_WORDS:
        return ParsedNumber(_MAGNITUDE_WORDS[cleaned], "magnitude_word")

    ordinal_match = re.fullmatch(r"(\d+)(st|nd|rd|th)", cleaned)
    if ordinal_match:
        return ParsedNumber(float(ordinal_match.group(1)), "ordinal")

    is_percentage = "%" in cleaned or "percent" in cleaned

    multiplier = 1.0
    for word, scale in _MAGNITUDE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", cleaned):
            multiplier = scale
            break

    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    value = float(match.group(0)) * multiplier
    return ParsedNumber(value, "percentage" if is_percentage else "number")


def _parse_number(text: Any, *, parent_value: Any = "", context: Any = "") -> Optional[ParsedNumber]:
    """
    Parse numbers while using the parent value/context to recover missing units.

    Example: if a parent value is "5.36 million" and the current token is "5.36",
    the token is interpreted as 5.36 million. This improves population/GDP
    comparisons when tokenization separates the magnitude word.
    """
    parsed = _parse_number_basic(text)
    if parsed is None:
        return None

    combined = f"{parent_value} {context}".lower()

    # Water (%) fields are often stored as plain numbers under a percentage label.
    if parsed.kind == "number" and ("%" in combined or "percent" in combined):
        return ParsedNumber(parsed.value, "percentage")

    if parsed.kind == "number":
        token_norm = normalize_label(text)
        # Only apply a scale from the parent if the token itself does not already carry one.
        has_scale_in_token = any(re.search(rf"\b{re.escape(word)}\b", token_norm) for word in _MAGNITUDE_WORDS)
        if not has_scale_in_token:
            for word, scale in _MAGNITUDE_WORDS.items():
                if re.search(rf"\b{re.escape(word)}\b", combined):
                    return ParsedNumber(parsed.value * scale, parsed.kind)

    return parsed


def _date_cost(a: ParsedDatePart, b: ParsedDatePart) -> float:
    if a.kind == "iso_date" and b.kind == "iso_date":
        return min(abs(a.value - b.value) / (365.0 * 100.0), 1.0)
    if a.kind == "year" and b.kind == "year":
        return min(abs(a.value - b.value) / 100.0, 1.0)
    if a.kind == "month" and b.kind == "month":
        diff = abs(a.value - b.value)
        return min(diff, 12 - diff) / 6.0
    if a.kind == "day" and b.kind == "day":
        return min(abs(a.value - b.value) / 31.0, 1.0)
    return 1.0


def _number_cost(a: ParsedNumber, b: ParsedNumber) -> float:
    if a.kind == "percentage" and b.kind == "percentage":
        return min(abs(a.value - b.value) / 100.0, 1.0)

    if a.kind == "ordinal" and b.kind == "ordinal":
        return min(abs(a.value - b.value) / 200.0, 1.0)

    if a.value == b.value:
        return 0.0

    if a.value == 0 or b.value == 0:
        return min(abs(a.value - b.value) / (abs(a.value) + abs(b.value) + 1.0), 1.0)

    return min(abs(math.log(abs(a.value) / abs(b.value))) / math.log(10.0), 1.0)


def _text_cost(a: Any, b: Any) -> float:
    return min(max(1.0 - label_similarity(a, b), 0.0), 1.0)


def semantic_token_update_cost(node_a: Dict[str, Any], node_b: Dict[str, Any]) -> float:
    """Graded update cost for leaf value/token nodes."""
    label_a = node_a.get("label")
    label_b = node_b.get("label")

    ctx_a = _parent_context(node_a)
    ctx_b = _parent_context(node_b)

    if _clean_text(label_a) == _clean_text(label_b):
        # Exact token matches are free only inside the same field/subsection.
        # If the local token is the same but the context differs (e.g. "Total"
        # values under GDP PPP vs GDP nominal), keep a small context cost.
        if normalize_label(ctx_a) == normalize_label(ctx_b):
            return 0.0
        if _contexts_compatible(ctx_a, ctx_b):
            return min(_text_cost(ctx_a, ctx_b), 0.45)
        return INCOMPATIBLE_UPDATE_COST
    if not _contexts_compatible(ctx_a, ctx_b):
        return INCOMPATIBLE_UPDATE_COST

    cat = field_category(f"{ctx_a} {ctx_b}")

    # For codes, numeric-looking tokens are identifiers, not quantities.
    # +961 and +41 should not become cheap just because they are numerically close.
    if cat in _NUMERIC_TEXT_ONLY_CATEGORIES:
        return 1.0

    if cat == "date_history":
        date_a = _parse_date_part(label_a)
        date_b = _parse_date_part(label_b)
        if date_a is not None and date_b is not None:
            return _date_cost(date_a, date_b)

    if cat in _NUMERIC_ALLOWED_CATEGORIES or cat is None:
        num_a = _parse_number(label_a, parent_value=node_a.get("parent_value", ""), context=ctx_a)
        num_b = _parse_number(label_b, parent_value=node_b.get("parent_value", ""), context=ctx_b)
        if num_a is not None and num_b is not None:
            return _number_cost(num_a, num_b)

    return _text_cost(label_a, label_b)


def semantic_label_update_cost(node_a: Dict[str, Any], node_b: Dict[str, Any]) -> float:
    """Graded update cost for element/attribute labels with subsection awareness."""
    label_a = node_a.get("label")
    label_b = node_b.get("label")
    ctx_a = _context(node_a)
    ctx_b = _context(node_b)

    if _clean_text(label_a) == _clean_text(label_b):
        # Same local label is not necessarily the same field. "Total" under
        # Area and "Total" under GDP are different subsections.
        if normalize_label(ctx_a) == normalize_label(ctx_b):
            return 0.0
        if _contexts_compatible(ctx_a, ctx_b):
            return min(_text_cost(ctx_a, ctx_b), 0.45)
        return INCOMPATIBLE_UPDATE_COST

    if not _contexts_compatible(ctx_a, ctx_b):
        return INCOMPATIBLE_UPDATE_COST

    cat = field_category(f"{ctx_a} {ctx_b}")

    if cat in {"population", "area", "gdp", "gini", "hdi"}:
        # Same numerical subsection/metric family, e.g. Population/2021 estimate
        # -> Population/2019 estimate, or GDP/Total -> GDP/Total.
        return min(_text_cost(ctx_a, ctx_b), 0.45)

    if cat in {"language", "capital", "demonym", "government_official", "government_system", "legislature", "date_history"}:
        return min(_text_cost(ctx_a, ctx_b), 0.65)

    return _text_cost(label_a, label_b)


def semantic_update_cost(node_a: Dict[str, Any], node_b: Dict[str, Any]) -> float:
    """
    Main entry point used by ted_diff.update_cost().

    Returns:
      - 0.0 for exact, structurally compatible matches
      - [0, 1] for meaningful same-type semantic updates
      - 2.0 for cross-type or context-incompatible updates
    """
    if node_a["node_type"] != node_b["node_type"]:
        return INCOMPATIBLE_UPDATE_COST

    node_type = node_a["node_type"]

    if node_type == "token":
        return semantic_token_update_cost(node_a, node_b)

    if node_type == "root":
        # The root represents the country entity. We keep a fixed unit update cost
        # instead of making country-name string similarity influence the document score.
        return 0.0 if _clean_text(node_a.get("label")) == _clean_text(node_b.get("label")) else 1.0

    return semantic_label_update_cost(node_a, node_b)
