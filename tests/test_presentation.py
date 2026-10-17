from copy import deepcopy
import unittest

from metso_assistant.presentation import line_ranges, readable_citations, render_sources


class PresentationTests(unittest.TestCase):
    def test_named_citations_preserve_numbers_source_identity_and_raw_evidence(self):
        sources = [{"id": "Babc", "file": "filter-optimizer.pdf", "page": 2},
                   {"id": "S01", "report_id": "FSR-0999", "lines": [{"line_start": 4, "line_end": 5}]},
                   {"id": "S02", "report_id": "FSR-0998"}]
        before = deepcopy(sources)
        result = readable_citations("12 cloths: XCU 123.40 [S01, S02]. Adjustments [Babc]. [unknown]", sources)
        self.assertIn("[FSR-0999; FSR-0998]", result)
        self.assertIn("[Filter optimizer brochure, p. 2]", result)
        self.assertIn("12 cloths: XCU 123.40", result)
        self.assertIn("[unknown]", result)
        self.assertEqual(before, sources)

    def test_source_lines_merge_only_supported_adjacent_intervals(self):
        values = [{"line_start": 13, "line_end": 14}, {"line_start": 15, "line_end": 15},
                  {"line_start": 20, "line_end": 21}, {"line_start": 13, "line_end": 13}]
        self.assertEqual(line_ranges(values), "13–15, 20–21")
        result = render_sources([{"id": "S01", "report_id": "FSR-0999", "lines": values}])
        self.assertEqual(result, "- FSR-0999, lines 13–15, 20–21")
        self.assertNotIn("line_start", result)

    def test_same_pdf_page_is_not_repeated_for_two_chunks(self):
        rows = [{"id": "Ba", "file": "larox-pf-ds.pdf", "page": 5},
                {"id": "Bb", "file": "larox-pf-ds.pdf", "page": 5}]
        self.assertEqual(len(render_sources(rows).splitlines()), 1)
        self.assertEqual(readable_citations("See [Ba, Bb]", rows), "See [Larox PF DS brochure, p. 5]")


if __name__ == "__main__":
    unittest.main()
