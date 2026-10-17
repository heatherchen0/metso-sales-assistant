from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from metso_assistant.geometry import (
    assess_geometry_table, cells_from_bboxes, extract_geometry_pages,
    keep_pdf_object, match_geometry_table,
)


def table_from_cells(texts, boxes, bbox, identifier="synthetic"):
    converted = cells_from_bboxes(boxes, lambda box: texts[tuple(box)])
    return assess_geometry_table({"id": identifier, "bbox": bbox,
                                  "coordinate_system": "pdf_points_top_left", **converted})


def structured_table(identifier="synthetic"):
    values = [["Parameter", "Value"], ["Width", "10"], ["Height", "20"], ["Depth", "30"]]
    boxes = [(column * 20, row * 10, (column + 1) * 20, (row + 1) * 10)
             for row in range(4) for column in range(2)]
    texts = {box: values[index // 2][index % 2] for index, box in enumerate(boxes)}
    return table_from_cells(texts, boxes, [0, 0, 40, 40], identifier)


class WhiteFillTests(unittest.TestCase):
    def rect(self, color, stroke=False):
        return {"object_type": "rect", "stroke": stroke, "fill": True, "non_stroking_color": color}

    def test_only_white_unstroked_fill_rectangles_are_excluded(self):
        for color in (1, (1,), (1, 1, 1), (0, 0, 0, 0)):
            with self.subTest(color=color):
                self.assertFalse(keep_pdf_object(self.rect(color)))
        for color in (None, (0.999, 0.999, 0.999), (0.8,), (0, 0, 0), (0, 0, 0, 1)):
            with self.subTest(color=color):
                self.assertTrue(keep_pdf_object(self.rect(color)))

    def test_actual_strokes_and_unknown_graphics_state_remain(self):
        self.assertTrue(keep_pdf_object(self.rect((1, 1, 1), stroke=True)))
        self.assertTrue(keep_pdf_object(self.rect((1, 1, 1), stroke=None)))
        line = dict(self.rect((1, 1, 1)), object_type="line")
        self.assertTrue(keep_pdf_object(line))
        self.assertTrue(keep_pdf_object({"object_type": "char", "text": "10"}))


class SpanTests(unittest.TestCase):
    def test_spanning_cells_keep_native_bbox_and_nulls(self):
        boxes = [(0, 0, 80, 20), (0, 20, 40, 60), (40, 20, 80, 40), (40, 40, 80, 60)]
        texts = {boxes[0]: "Merged heading", boxes[1]: "Parameter", boxes[2]: "10", boxes[3]: None}
        result = cells_from_bboxes(boxes, lambda box: texts[box])
        self.assertEqual((result["row_count"], result["column_count"]), (3, 2))
        self.assertEqual((result["cells"][0]["row_span"], result["cells"][0]["col_span"]), (1, 2))
        self.assertEqual((result["cells"][1]["row_span"], result["cells"][1]["col_span"]), (2, 1))
        self.assertEqual(result["cells"][1]["bbox"], [0, 20, 40, 60])
        self.assertIsNone(result["cells"][3]["text"])

    def test_boundaries_tolerate_small_rounding_without_inventing_text(self):
        boxes = [(0, 0, 20, 10), (20.000001, 0, 40, 10)]
        result = cells_from_bboxes(boxes, lambda box: "raw")
        self.assertEqual(result["column_count"], 2)
        self.assertEqual(result["cells"][1]["bbox"][0], 20.000001)
        self.assertEqual([cell["text"] for cell in result["cells"]], ["raw", "raw"])


class CoherenceTests(unittest.TestCase):
    def test_structured_grid_is_candidate_evidence_not_verified_facts(self):
        table = structured_table()
        self.assertTrue(table["coherent"])
        self.assertEqual(table["numeric_cell_count"], 3)
        self.assertNotIn("facts", table)
        self.assertNotIn("review_status", table)

    def test_giant_multiline_cell_swallowing_many_rows_is_retained_but_rejected(self):
        boxes = [(0, 0, 20, 60)] + [(20, row * 10, 40, (row + 1) * 10) for row in range(6)]
        texts = {box: str(index) for index, box in enumerate(boxes)}
        texts[boxes[0]] = "plot\naxis\ncaption\nlegend\nannotation\nbody"
        table = table_from_cells(texts, boxes, [0, 0, 40, 60])
        self.assertFalse(table["coherent"])
        self.assertTrue(any(issue["code"] == "giant_multiline_cell" for issue in table["issues"]))
        self.assertEqual(table["cells"][0]["text"], texts[boxes[0]])
        self.assertIsNone(match_geometry_table(table["bbox"], [table]))

    def test_plot_decorations_or_text_only_boxes_are_not_table_candidates(self):
        boxes = [(0, 0, 20, 10), (20, 0, 40, 10), (0, 10, 20, 20), (20, 10, 40, 20)]
        texts = {box: "decorative caption" for box in boxes}
        table = table_from_cells(texts, boxes, [0, 0, 40, 20])
        self.assertFalse(table["coherent"])
        self.assertIsNone(match_geometry_table(table["bbox"], [table]))

    def test_overlapping_cells_are_not_approved_as_a_valid_partition(self):
        table = structured_table()
        table["cells"].append(deepcopy(table["cells"][2]))
        table["issues"] = []
        result = assess_geometry_table(table)
        self.assertFalse(result["coherent"])
        self.assertTrue(any(issue["code"] == "overlapping_cells" for issue in result["issues"]))


class MatchTests(unittest.TestCase):
    def test_best_coherent_iou_match_is_returned_without_mutating_raw_table(self):
        table = structured_table()
        before = deepcopy(table)
        result = match_geometry_table([1, 1, 39, 39], [table])
        self.assertEqual(result["id"], table["id"])
        self.assertGreater(result["match"]["iou"], 0.9)
        self.assertEqual(table, before)

    def test_partial_fragment_does_not_match_the_whole_table(self):
        table = structured_table()
        self.assertIsNone(match_geometry_table([0, 0, 80, 80], [table]))

    def test_tied_candidates_and_wrong_coordinate_system_remain_unmatched(self):
        first = structured_table("one")
        second = structured_table("two")
        self.assertIsNone(match_geometry_table(first["bbox"], [first, second]))
        first["coordinate_system"] = "pdf_bottom_left"
        self.assertIsNone(match_geometry_table(first["bbox"], [first]))


class ExtractionTests(unittest.TestCase):
    def test_all_raw_tables_survive_inventory_with_native_rows(self):
        native_boxes = [(column * 20, row * 10, (column + 1) * 20, (row + 1) * 10)
                        for row in range(4) for column in range(2)]
        raw_rows = [["Parameter", "Value"], ["Width", "10"], ["Height", "20"], ["Depth", "30"]]
        native = SimpleNamespace(bbox=(0, 0, 40, 40), cells=native_boxes,
                                 rows=[SimpleNamespace(cells=native_boxes[row * 2:row * 2 + 2]) for row in range(4)],
                                 extract=lambda: raw_rows)
        filtered = MagicMock()
        filtered.find_tables.return_value = [native]
        page = MagicMock()
        page.rects = [{"object_type": "rect", "stroke": False, "fill": True, "non_stroking_color": (1, 1, 1)}]
        page.filter.return_value = filtered
        context = MagicMock()
        context.__enter__.return_value.pages = [page]
        with patch("pdfplumber.open", return_value=context):
            result = extract_geometry_pages(Path("synthetic.pdf"))
        self.assertEqual(list(result), [1])
        self.assertEqual(result[1][0]["raw_rows"], raw_rows)
        self.assertEqual(result[1][0]["raw_cell_bboxes"], [list(box) for box in native_boxes])
        self.assertEqual(result[1][0]["filtered_white_rectangle_count"], 1)
        self.assertEqual(result[1][0]["coordinate_system"], "pdf_points_top_left")
        page.filter.assert_called_once_with(keep_pdf_object)


if __name__ == "__main__":
    unittest.main()
