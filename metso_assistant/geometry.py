"""Optional PDF-vector table evidence, never a factual authority or approval.

Outputs contain source text and belong in the private ingestion cache. All
detected tables are retained. ``coherent`` is a conservative shape/numeric-text
screen, not proof that printed models, units or values have correct associations.
Use original-page inspection and content-bound review before verifying facts.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.metadata
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping


BOUNDARY_TOLERANCE = 0.01
MIN_NUMERIC_CELL_RATIO = 0.20
MIN_NUMERIC_CELLS = 3
TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
NUMERIC_TEXT = re.compile(r"(?<!\w)[+\-−]?\d+(?:[.,]\d+)?")


def _white(color: Any) -> bool:
    if isinstance(color, (int, float)) and not isinstance(color, bool):
        return math.isfinite(color) and abs(color - 1) <= 1e-6
    if isinstance(color, (tuple, list)) and len(color) in {1, 3, 4}:
        target = 0 if len(color) == 4 else 1
        return all(isinstance(component, (int, float)) and not isinstance(component, bool)
                   and math.isfinite(component) and abs(component - target) <= 1e-6
                   for component in color)
    return False


def keep_pdf_object(obj: Mapping) -> bool:
    """Ignore invisible white fill edges; retain strokes and nonwhite fills."""
    return not (obj.get("object_type") == "rect" and obj.get("stroke") is False
                and obj.get("fill") is True and _white(obj.get("non_stroking_color")))


def _bbox(value: Iterable[float]) -> tuple[float, float, float, float]:
    numbers = tuple(value)
    if len(numbers) != 4 or any(isinstance(number, bool) or not isinstance(number, (int, float))
                                or not math.isfinite(number) for number in numbers):
        raise ValueError("bbox requires four finite PDF coordinates")
    x0, top, x1, bottom = map(float, numbers)
    if x1 <= x0 or bottom <= top:
        raise ValueError("bbox must have positive width and height")
    return x0, top, x1, bottom


def _boundaries(values: Iterable[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(set(values)):
        # Compare with the first coordinate to avoid transitive cluster drift.
        if groups and value - groups[-1][0] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def cells_from_bboxes(raw_cells: Iterable[Iterable[float]],
                      extract_text: Callable[[tuple], str | None],
                      tolerance: float = BOUNDARY_TOLERANCE) -> dict:
    """Derive logical spans from all boundaries, keeping each native bbox/text."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    bboxes = [_bbox(cell) for cell in raw_cells]
    xs = _boundaries([coordinate for box in bboxes for coordinate in (box[0], box[2])], tolerance)
    ys = _boundaries([coordinate for box in bboxes for coordinate in (box[1], box[3])], tolerance)
    cells, issues = [], []
    for box in sorted(bboxes, key=lambda item: (item[1], item[0], item[3], item[2])):
        left, right = (min(range(len(xs)), key=lambda index: abs(xs[index] - coordinate))
                       for coordinate in (box[0], box[2]))
        top, bottom = (min(range(len(ys)), key=lambda index: abs(ys[index] - coordinate))
                       for coordinate in (box[1], box[3]))
        text = extract_text(box)
        if text is not None and not isinstance(text, str):
            raise ValueError("Cell text must be a string or null")
        cells.append({"text": text, "row_index": top, "col_index": left,
                      "row_span": bottom - top, "col_span": right - left,
                      "bbox": list(box)})
        if bottom <= top or right <= left:
            issues.append({"code": "collapsed_cell_boundaries", "severity": "blocker", "bbox": list(box)})
    return {"cells": cells, "row_count": max(0, len(ys) - 1),
            "column_count": max(0, len(xs) - 1), "x_boundaries": xs,
            "y_boundaries": ys, "issues": issues}


def assess_geometry_table(table: Mapping) -> dict:
    """Describe suspicious geometry without repairing cells or reading facts."""
    result = deepcopy(dict(table))
    issues = list(result.get("issues", []))
    cells = result["cells"]
    rows, columns = result["row_count"], result["column_count"]
    if rows < 3 or columns < 2:
        issues.append({"code": "insufficient_grid_dimensions", "severity": "blocker"})
    nonempty = [cell for cell in cells if cell["text"] and cell["text"].strip()]
    numeric = [cell for cell in nonempty if NUMERIC_TEXT.search(cell["text"])]
    ratio = len(numeric) / len(nonempty) if nonempty else 0
    result["numeric_cell_ratio"] = round(ratio, 6)
    result["numeric_cell_count"] = len(numeric)
    if len(numeric) < MIN_NUMERIC_CELLS or ratio < MIN_NUMERIC_CELL_RATIO:
        issues.append({"code": "insufficient_numeric_structure", "severity": "blocker"})
    bbox = _bbox(result["bbox"])
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    occupied = Counter()
    for cell in cells:
        for row in range(cell["row_index"], cell["row_index"] + cell["row_span"]):
            for column in range(cell["col_index"], cell["col_index"] + cell["col_span"]):
                occupied[(row, column)] += 1
        lines = [line for line in (cell["text"] or "").splitlines() if line.strip()]
        box = _bbox(cell["bbox"])
        cell_area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / area
        if len(lines) >= 5 and (cell["row_span"] >= 3 or cell_area_ratio >= 0.45):
            issues.append({"code": "giant_multiline_cell", "severity": "blocker", "bbox": list(box)})
    coverage = len(occupied) / (rows * columns) if rows and columns else 0
    result["occupied_grid_ratio"] = round(coverage, 6)
    if any(count > 1 for count in occupied.values()):
        issues.append({"code": "overlapping_cells", "severity": "blocker"})
    if coverage < 0.50:
        issues.append({"code": "fragmented_grid", "severity": "blocker"})
    result["issues"] = issues
    result["coherent"] = not any(issue.get("severity") == "blocker" for issue in issues)
    return result


def extract_geometry_pages(pdf_path: str | Path, pages: Iterable[int] | None = None) -> dict[int, list[dict]]:
    """Extract all raw vector tables on selected physical source pages."""
    import pdfplumber

    pdf_path = Path(pdf_path)
    result = {}
    with pdfplumber.open(pdf_path) as pdf:
        selected = list(pages) if pages is not None else list(range(1, len(pdf.pages) + 1))
        if any(type(page) is not int or not 1 <= page <= len(pdf.pages) for page in selected):
            raise ValueError("Pages must be valid one-based physical PDF page numbers")
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate physical pages requested")
        for number in selected:
            source = pdf.pages[number - 1]
            removed = sum(not keep_pdf_object(rect) for rect in source.rects)
            page = source.filter(keep_pdf_object)
            tables = sorted(page.find_tables(table_settings=TABLE_SETTINGS), key=lambda table: (table.bbox[1], table.bbox[0]))
            result[number] = []
            for ordinal, native in enumerate(tables, 1):
                raw_rows = native.extract()
                text_by_bbox = {tuple(box): text for row, texts in zip(native.rows, raw_rows)
                                for box, text in zip(row.cells, texts) if box is not None}
                converted = cells_from_bboxes(native.cells, lambda box: text_by_bbox.get(box))
                table = {"id": f"{pdf_path.stem}:page-{number}:geometry-{ordinal}",
                         "page": number, "bbox": list(native.bbox),
                         "coordinate_system": "pdf_points_top_left", **converted,
                         "raw_rows": raw_rows, "raw_cell_bboxes": [list(box) for box in native.cells],
                         "filtered_white_rectangle_count": removed,
                         "parser": {"name": "pdfplumber", "version": importlib.metadata.version("pdfplumber")},
                         "settings": {"table_settings": dict(TABLE_SETTINGS),
                                      "boundary_tolerance": BOUNDARY_TOLERANCE,
                                      "white_unstroked_fill_filter": True,
                                      "minimum_numeric_cell_ratio": MIN_NUMERIC_CELL_RATIO,
                                      "minimum_numeric_cells": MIN_NUMERIC_CELLS}}
                result[number].append(assess_geometry_table(table))
    return result


def match_geometry_table(candidate_bbox: Iterable[float], tables: Iterable[Mapping],
                         min_iou: float = 0.25, min_candidate_coverage: float = 0.60) -> dict | None:
    """Match a coherent full table; fragmented/tied candidates remain unmatched."""
    if not 0 <= min_iou <= 1 or not 0 <= min_candidate_coverage <= 1:
        raise ValueError("Match thresholds must be between zero and one")
    candidate = _bbox(candidate_bbox)
    candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    matches = []
    for table in tables:
        if not table.get("coherent") or table.get("coordinate_system") != "pdf_points_top_left":
            continue
        box = _bbox(table["bbox"])
        width = max(0, min(candidate[2], box[2]) - max(candidate[0], box[0]))
        height = max(0, min(candidate[3], box[3]) - max(candidate[1], box[1]))
        intersection = width * height
        table_area = (box[2] - box[0]) * (box[3] - box[1])
        iou = intersection / (candidate_area + table_area - intersection)
        coverage = intersection / candidate_area
        if iou >= min_iou and coverage >= min_candidate_coverage:
            matches.append((iou, coverage, table))
    matches.sort(key=lambda item: item[0], reverse=True)
    if not matches or len(matches) > 1 and abs(matches[0][0] - matches[1][0]) <= 0.01:
        return None
    iou, coverage, table = matches[0]
    result = deepcopy(dict(table))
    result["match"] = {"iou": round(iou, 6), "candidate_coverage": round(coverage, 6)}
    return result
