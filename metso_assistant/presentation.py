"""Readable citations at the user boundary; evidence identifiers stay in traces."""
from __future__ import annotations

from pathlib import PurePath
import re


def line_ranges(values) -> str:
    intervals = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, dict):
            continue
        start, end = value.get("line_start"), value.get("line_end")
        if type(start) is int and type(end) is int and 0 < start <= end:
            intervals.append((start, end))
    merged = []
    for start, end in sorted(set(intervals)):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in merged)


def source_label(source: dict, *, locations=True) -> str:
    name = source.get("report_id") or source.get("file") or source.get("source_file") or "Source"
    name = PurePath(str(name).replace("\\", "/")).name
    if source.get("report_id"):
        name = str(source["report_id"])
    elif name.lower().endswith(".pdf"):
        title = name[:-4].replace("-", " ")
        acronyms = {"pf", "ds", "ac", "gm", "md", "vsp"}
        words = [word.upper() if word.casefold() in acronyms else word for word in title.split()]
        name = " ".join(words)
        name = name[:1].upper() + name[1:]
        if "brochure" not in name.casefold():
            name += " brochure"
    if locations and source.get("page"):
        name += f", p. {source['page']}"
    ranges = line_ranges(source.get("lines", source.get("line_ranges")))
    if locations and ranges:
        name += ", lines " + ranges
    return name


def render_sources(sources: list[dict]) -> str:
    labels = list(dict.fromkeys(source_label(source) for source in sources))
    return "\n".join("- " + label for label in labels)


def readable_citations(answer: str, sources: list[dict]) -> str:
    # Resolve only actual source identifiers. Never remove an unknown citation,
    # rewrite a number, or change evidence/claims in the structured result.
    labels = {source["id"]: source_label(source, locations=not bool(source.get("report_id")))
              for source in sources if source.get("id")}
    def replace(match):
        identifiers = [part.strip() for part in match[1].split(",")]
        if not identifiers or any(part not in labels for part in identifiers):
            return match[0]
        return "[" + "; ".join(dict.fromkeys(labels[part] for part in identifiers)) + "]"
    return re.sub(r"\[([^\[\]\n]+)\]", replace, answer)
