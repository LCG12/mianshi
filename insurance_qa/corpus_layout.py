"""Parse the reviewed terms layout; raw spans and search text stay separate."""

from __future__ import annotations

import re


HEADING = re.compile(r"(?m)^[ \t]*(\d+\.\d+(?:\.\d+)*|\d+[．.])[ \t]+")


def _clean(text: str, ranges: list[tuple[int, int]], title: str,
           omit_lines: tuple[str, ...] = ()) -> tuple[str, list[int]]:
    """Remove the numbered heading and its wrapped left column, retaining offsets."""
    chars, positions = [], []
    remaining = title
    first = True
    for start, end in ranges:
        offset = start
        for line in text[start:end].splitlines(keepends=True):
            if line.strip() in omit_lines:
                offset += len(line)
                continue
            left = len(line) - len(line.lstrip())
            if first:
                match = HEADING.match(line)
                if match:
                    left = match.end()
                first = False
            if remaining:
                fragment = re.split(r"\s{2,}", line[left:].strip(), maxsplit=1)[0]
                if fragment and remaining.startswith(fragment):
                    left += line[left:].index(fragment) + len(fragment)
                    remaining = remaining[len(fragment):]
            for index in range(left, len(line)):
                if not line[index].isspace():
                    chars.append("×" if line[index] == "\uf0cd" else line[index])
                    positions.append(offset + index)
            offset += len(line)
    if remaining:
        raise ValueError(f"Unrecognized wrapped heading: {title!r}, remaining {remaining!r}")
    return "".join(chars), positions


def terms_units(pages: list[str], config: dict) -> list[dict]:
    """Return every body character in source spans, split only at reviewed anchors.

    Page numbers are physical, one-based. The caller verifies input file hashes.
    A reference is not a requirement: its target's gate must not be inherited.
    """
    text = "\n".join(pages)
    offsets = []
    cursor = 0
    for page in pages:
        offsets.append(cursor)
        cursor += len(page) + 1
    body_start = offsets[config["body_start_page"] - 1]
    headings = [(m.group(1).rstrip("．."), m.start()) for m in HEADING.finditer(text, body_start)]
    if [key for key, _ in headings] != list(config["titles"]):
        raise ValueError("Terms headings differ from the reviewed configuration")

    footnotes = []
    for item in config.get("footnotes", []):
        page = pages[item["page"] - 1]
        if page.count(item["text"]) != 1:
            raise ValueError("Footnote anchor is missing or ambiguous")
        start = offsets[item["page"] - 1] + page.index(item["text"])
        footnotes.append((start, start + len(item["text"]), item["section"]))

    def source_spans(ranges):
        spans = []
        for start, end in ranges:
            for number, (base, page) in enumerate(zip(offsets, pages), 1):
                left = max(start, base)
                right = min(end, base + len(page))
                if left < right:
                    spans.append([number, left - base, right - base])
        return spans

    def without_footnotes(start, end):
        ranges = [(start, end)]
        for first, last, _ in footnotes:
            remaining = []
            for left, right in ranges:
                before_end = min(right, first)
                after_start = max(left, last)
                if left < before_end:
                    remaining.append((left, before_end))
                if after_start < right:
                    remaining.append((after_start, right))
            ranges = remaining
        return ranges

    units = [{"section": "introduction", "title": "条款名称与称谓说明", "unit_kind": "metadata",
              "spans": source_spans([(body_start, headings[0][1])]),
              "retrieval_text": re.sub(r"\s+", "", text[body_start:headings[0][1]]),
              "requires": [], "references": []}]
    for index, (section, start) in enumerate(headings):
        end = headings[index + 1][1] if index + 1 < len(headings) else len(text)
        ranges = without_footnotes(start, end)
        title = config["titles"][section]
        cleaned, positions = _clean(text, ranges, title, tuple(
            item["marker"] for item in config.get("footnotes", [])
            if item["section"] == section and "marker" in item))
        pieces = config.get("splits", {}).get(section, [{"key": section, "title": title}])
        boundaries = [0]
        for piece in pieces[1:]:
            anchor = piece["start"]
            if cleaned.count(anchor) != 1:
                raise ValueError(f"Missing or ambiguous anchor in {section}: {anchor}")
            boundaries.append(cleaned.index(anchor))
        if boundaries != sorted(set(boundaries)):
            raise ValueError(f"Out-of-order split anchors in {section}")
        boundaries.append(len(cleaned))
        for number, piece in enumerate(pieces):
            first, last = boundaries[number:number + 2]
            raw_start = start if number == 0 else positions[first]
            raw_end = end if number == len(pieces) - 1 else positions[last]
            part_ranges = [(max(a, raw_start), min(b, raw_end)) for a, b in ranges
                           if max(a, raw_start) < min(b, raw_end)]
            extra = [(a, b) for a, b, owner in footnotes if owner == section]
            if extra and len(pieces) != 1:
                raise ValueError("Assign a footnote to a single unsplit section")
            part_ranges.extend(extra)
            default_kind = "heading" if "." not in section or not cleaned else "clause"
            if section.startswith("12."):
                default_kind = "definition"
            links = config.get("relations", {}).get(piece["key"], {})
            units.append({"section": piece["key"], "title": piece.get("title", title),
                          "unit_kind": piece.get("unit_kind", default_kind),
                          "spans": source_spans(part_ranges),
                          "retrieval_text": cleaned[first:last] + "".join(
                              re.sub(r"\s+", "", text[a:b]) for a, b in extra),
                          "requires": piece.get("requires", links.get("requires", [])),
                          "references": piece.get("references", links.get("references", []))})
    keys = {unit["section"] for unit in units}
    for unit in units:
        if not set(unit["requires"] + unit["references"]) <= keys:
            raise ValueError(f"Dangling relation from {unit['section']}")
    return units
