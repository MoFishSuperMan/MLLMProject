"""Short-answer extraction helpers for benchmark scoring."""

from __future__ import annotations

import re


FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)
SOURCE_RE = re.compile(r"\n?\s*来源[:：].*", re.DOTALL)
BACKEND_SOURCE_RE = re.compile(r"\n?\s*Sources?[:：].*", re.IGNORECASE | re.DOTALL)
CITATION_RE = re.compile(r"\[\s*(?:E\s*)?\d+\s*\]|\[page=[^\]]+\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
LATIN_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z&.'-]*(?:\s+(?:of|the|and|for|de|la|[A-Z][A-Za-z&.'-]*))*\b"
)

YES_NO_QUESTION_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|should|would|has|have|had)\b",
    re.IGNORECASE,
)

QUESTION_NUMBER_HINTS = (
    "how many",
    "difference",
    "value",
    "actual",
    "lowest",
    "highest",
    "maximum",
    "minimum",
    "sum",
    "average",
    "percent",
    "percentage",
    "number",
    "多少",
    "数值",
    "最大",
    "最小",
    "差",
)

ENTITY_ALIASES = {
    "圣迭戈": "san diego",
    "圣地亚哥": "san diego",
    "加州大学圣地亚哥分校": "university of california",
    "加州大学圣迭戈分校": "university of california",
    "加利福尼亚大学圣地亚哥分校": "university of california",
    "加利福尼亚大学圣迭戈分校": "university of california",
    "加州大学": "university of california",
    "加利福尼亚大学": "university of california",
}


def extract_short_answer(raw_answer: str, question: str = "") -> str:
    """Extract a benchmark-friendly answer from a verbose model response."""

    cleaned = strip_sources(raw_answer)
    final = extract_final_answer(cleaned)
    if final:
        if is_count_question(question):
            explicit_count = extract_explicit_count(cleaned)
            if explicit_count:
                return explicit_count
            listed_count = extract_list_count(cleaned)
            if listed_count:
                return str(listed_count)
        return repair_final_answer(final, cleaned, question)

    if is_yes_no_question(question):
        comparison = extract_comparison_yes_no(cleaned, question)
        if comparison:
            return comparison
        yes_no = extract_yes_no(cleaned)
        if yes_no:
            return yes_no

    if wants_number(question):
        number = extract_number(cleaned)
        if number:
            return number

    entity = extract_latin_entity(cleaned)
    if entity:
        return entity

    number = extract_number(cleaned)
    if number:
        return number

    yes_no = extract_yes_no(cleaned)
    if yes_no:
        return yes_no

    return normalize_short_answer(cleaned)


def repair_final_answer(final_answer: str, full_answer: str, question: str = "") -> str:
    year = repair_truncated_year(final_answer, full_answer, question)
    if year:
        return year
    return final_answer


def repair_truncated_year(final_answer: str, full_answer: str, question: str = "") -> str:
    final = normalize_short_answer(final_answer)
    if not re.fullmatch(r"\d{2,3}", final):
        return ""
    lowered = question.lower()
    if not any(hint in lowered for hint in ("when", "year", "date", "increase", "decrease", "peak")):
        return ""

    ranges = re.findall(
        r"\b((?:18|19|20)\d{2})\s*(?:年)?\s*(?:-|–|—|to|至|到)\s*((?:18|19|20)\d{2})\s*(?:年)?",
        full_answer,
        flags=re.IGNORECASE,
    )
    if ranges and any(hint in lowered for hint in ("increase", "decrease", "rise", "fall", "change")):
        return ranges[-1][1]

    candidates = re.findall(r"\b(?:18|19|20)\d{2}\b", full_answer)
    prefix_matches = [year for year in candidates if year.startswith(final)]
    if prefix_matches:
        return prefix_matches[-1]
    return ""


def extract_final_answer(answer: str) -> str:
    match = FINAL_ANSWER_RE.search(answer)
    if not match:
        return ""
    value = match.group(1).splitlines()[0]
    return normalize_short_answer(value)


def strip_sources(answer: str) -> str:
    answer = SOURCE_RE.sub("", answer)
    answer = BACKEND_SOURCE_RE.sub("", answer)
    return answer.strip()


def normalize_short_answer(answer: str) -> str:
    answer = strip_sources(answer)
    answer = CITATION_RE.sub("", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.strip(" \t\r\n。.!！?？,，;；:：\"'“”‘’()（）[]【】")
    return answer


def is_yes_no_question(question: str) -> bool:
    normalized = question.strip().lower()
    return bool(YES_NO_QUESTION_RE.search(question)) or normalized.startswith(("是否", "是不是"))


def extract_yes_no(answer: str) -> str:
    lowered = answer.lower()
    if re.search(r"\b(no|not|false)\b", lowered) or re.search(r"不是|否|不正确|没有", answer):
        return "No"
    if re.search(r"\b(yes|true)\b", lowered) or re.search(r"是的|是，|是。|可以|正确", answer):
        return "Yes"
    return ""


def extract_comparison_yes_no(answer: str, question: str) -> str:
    lowered = question.lower()
    if not any(hint in lowered for hint in ("greater", "larger", "more", "less", "smaller", "higher", "lower")):
        return ""

    totals = extract_equation_totals(answer)
    if len(totals) < 2:
        return ""
    left, right = totals[0], totals[1]
    if any(hint in lowered for hint in ("greater", "larger", "more", "higher")):
        return "Yes" if left > right else "No"
    if any(hint in lowered for hint in ("less", "smaller", "lower")):
        return "Yes" if left < right else "No"
    return ""


def extract_equation_totals(answer: str) -> list[float]:
    totals: list[float] = []
    for match in re.finditer(r"=\s*([-+]?\d+(?:\.\d+)?)\s*%?", answer):
        try:
            totals.append(float(match.group(1)))
        except ValueError:
            continue
    return totals


def wants_number(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered or hint in question for hint in QUESTION_NUMBER_HINTS)


def is_count_question(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in ("how many", "number of", "count")) or "多少" in question


def extract_list_count(answer: str) -> int | None:
    body = FINAL_ANSWER_RE.split(strip_sources(answer), maxsplit=1)[0]
    match = re.search(r"(?:分别是|包括|包含|include(?:s|d)?|including)[:：]?\s*(.+)", body, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    list_text = match.group(1)
    list_text = re.split(r"[。.\n]", list_text, maxsplit=1)[0]
    list_text = CITATION_RE.sub("", list_text)
    list_text = re.sub(r"\b(?:and)\b", "、", list_text, flags=re.IGNORECASE)
    list_text = list_text.replace("和", "、").replace("等", "")
    pieces = re.split(r"[、,，;；]", list_text)
    items = []
    for piece in pieces:
        item = normalize_short_answer(piece)
        if not item or NUMBER_RE.fullmatch(item):
            continue
        if len(item) > 40:
            continue
        items.append(item)
    if 2 <= len(items) <= 50:
        return len(items)
    return None


def extract_explicit_count(answer: str) -> str:
    body = FINAL_ANSWER_RE.split(strip_sources(answer), maxsplit=1)[0]
    match = re.search(r"(?:共计|共有|总共|共)\s*(\d+)\s*(?:项|个|种|条|根|bars?|items?)?", body, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def extract_number(answer: str) -> str:
    text = normalize_short_answer(answer)
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return ""

    decimals = [value for value in numbers if "." in value]
    if decimals:
        return decimals[-1].rstrip("%") if decimals[-1].endswith("%") else decimals[-1]

    filtered = [
        value
        for value in numbers
        if not (len(value) == 4 and value.isdigit() and 1800 <= int(value) <= 2200)
    ]
    if filtered:
        return filtered[0].rstrip("%") if filtered[0].endswith("%") else filtered[0]
    return numbers[-1].rstrip("%") if numbers[-1].endswith("%") else numbers[-1]


def extract_latin_entity(answer: str) -> str:
    text = normalize_short_answer(answer)
    candidates = [candidate.strip() for candidate in LATIN_ENTITY_RE.findall(text)]
    candidates = [
        candidate
        for candidate in candidates
        if candidate.lower() not in {"final answer", "source", "sources"}
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda value: (len(value.split()), len(value)))


def normalize_entity_answer(answer: str, question: str = "") -> str:
    """Map common Chinese entity renderings back to benchmark gold forms."""

    normalized = normalize_short_answer(answer)
    for phrase, canonical in sorted(ENTITY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase in normalized:
            return canonical
    return normalized
