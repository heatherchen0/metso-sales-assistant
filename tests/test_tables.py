import unittest

from metso_assistant.tables import grid_from_cells, normalize_table, numeric


def cell(row, column, value, span=1):
    return {"row_index": row, "col_index": column, "row_span": 1, "col_span": span, "text": value, "bbox": None}


class TableNormalizationTests(unittest.TestCase):
    def normalize(self, cells, rows, columns):
        return normalize_table("synthetic", cells, rows, columns,
                               {"file": "synthetic.pdf", "sha": "a" * 64, "page": 1},
                               "synthetic-parser", "b" * 64)

    def test_group_crossing_a_configuration_keeps_that_mapping_ambiguous(self):
        cells = [cell(0, 0, "Series"), cell(0, 1, ""), cell(0, 2, "10"), cell(0, 3, "20", 2), cell(0, 5, "30"),
                 cell(1, 0, "Filtration area"), cell(1, 1, "m2"), cell(1, 2, "10"), cell(1, 3, "20", 2), cell(1, 5, "30"),
                 cell(2, 0, "Cloth length"), cell(2, 1, "m"), cell(2, 2, "7", 2), cell(2, 4, "9", 2)]
        table = self.normalize(cells, 3, 6)
        facts = {f["configuration"]: f for f in table["facts"] if f["parameter"] == "Cloth length"}
        self.assertEqual(facts["10"]["value"], 7)
        self.assertEqual(facts["30"]["value"], 9)
        self.assertIsNone(facts["20"]["value"])
        self.assertEqual(facts["20"]["status"], "ambiguous")
        self.assertEqual(len(facts["20"]["candidate_cells"]), 2)

    def test_blank_is_missing_and_never_forward_filled(self):
        cells = [cell(0, 0, "Series"), cell(0, 1, ""), cell(0, 2, "10"), cell(0, 3, "20"), cell(0, 4, "30"),
                 cell(1, 0, "Filtration area"), cell(1, 1, "m2"), cell(1, 2, "10"), cell(1, 3, "20"), cell(1, 4, "30"),
                 cell(2, 0, "Height"), cell(2, 1, "mm"), cell(2, 2, "500"), cell(2, 3, ""), cell(2, 4, "-")]
        facts = [f for f in self.normalize(cells, 3, 5)["facts"] if f["parameter"] == "Height"]
        self.assertEqual([f["value"] for f in facts], [500, None, None])
        self.assertEqual([f["status"] for f in facts], ["reported", "missing", "missing"])

    def test_printed_volume_area_unit_is_preserved_with_a_warning(self):
        cells = [cell(0, 0, "Filtration area"), cell(0, 1, "m2"), cell(0, 2, "10"), cell(0, 3, "20"), cell(0, 4, "30"),
                 cell(1, 0, "Chamber volume"), cell(1, 1, "m2"), cell(1, 2, "0.2"), cell(1, 3, "0.4"), cell(1, 4, "0.6")]
        fact = next(f for f in self.normalize(cells, 2, 5)["facts"] if f["parameter"] == "Chamber volume")
        self.assertEqual(fact["unit"], "m²")
        self.assertEqual(fact["value"], .2)
        self.assertIn("source_dimension_anomaly", [i["code"] for i in fact["issues"]])

    def test_explicit_compound_quantities_keep_distinct_units_and_scope(self):
        cells = [cell(0, 0, "Chambers"), cell(0, 1, ""), cell(0, 2, "10"), cell(0, 3, "20"), cell(0, 4, "30"),
                 cell(1, 0, "Pressing pump"), cell(1, 1, "IEC"), cell(1, 2, "5 kW, 900 rpm, 40 Hz", 3),
                 cell(2, 0, "Height"), cell(2, 1, "mm"), cell(2, 2, "500", 3)]
        facts = [f for f in self.normalize(cells, 3, 5)["facts"] if f["configuration"] == "10" and "Pressing" in f["parameter"]]
        self.assertEqual([(f["value"], f["unit"]) for f in facts], [(5, "kW"), (900, "rpm"), (40, "Hz")])
        self.assertTrue(all(f["parameter"].endswith("IEC") for f in facts))

    def test_overlapping_cells_fail_instead_of_selecting_one(self):
        with self.assertRaises(ValueError):
            grid_from_cells([cell(0, 0, "first", 2), cell(0, 1, "second")], 1, 2)

    def test_numeric_separator_ambiguity_is_not_silently_resolved(self):
        self.assertEqual(numeric("1,200", "mm")["value"], 1200)
        self.assertIsNone(numeric("1,200", "m³")["value"])
        self.assertEqual(numeric("1,25", "m")["value"], 1.25)
        self.assertEqual(numeric("1 250", "kg")["value"], 1250)

    def test_section_qualifiers_do_not_collide_or_become_values(self):
        cells = [cell(0, 0, "Series"), cell(0, 1, ""), cell(0, 2, "10"), cell(0, 3, "20"), cell(0, 4, "30"),
                 cell(1, 0, "Height", 5), cell(2, 0, "33 mm"), cell(2, 1, "mm"), cell(2, 2, "500", 3),
                 cell(3, 0, "Weight", 5), cell(4, 0, "33 mm"), cell(4, 1, "kg"), cell(4, 2, "100", 3)]
        facts = self.normalize(cells, 5, 5)["facts"]
        self.assertEqual({f["parameter"] for f in facts}, {"Height / 33 mm", "Weight / 33 mm"})
        self.assertEqual(len(facts), 6)

    def test_malformed_power_speed_punctuation_is_blocked(self):
        fact = self.normalize([cell(0, 0, "Servo"), cell(0, 1, "2.2.3000")], 1, 2)["facts"][0]
        self.assertEqual(fact["status"], "ambiguous")
        self.assertIn("malformed_numeric_punctuation", [i["code"] for i in fact["issues"]])

    def test_speed_grouping_retains_source_review_warning(self):
        parsed = numeric("1,500", "rpm")
        self.assertEqual(parsed["value"], 1500)
        self.assertEqual(parsed["issues"][0]["severity"], "warning")

    def test_combined_depth_and_following_frequency_keep_parent_scope(self):
        cells = [cell(0, 0, "Series"), cell(0, 1, ""), cell(0, 2, "10"), cell(0, 3, "20"), cell(0, 4, "30"),
                 cell(1, 0, "Height", 5), cell(2, 0, "45 and 60 mm"), cell(2, 1, "mm"), cell(2, 2, "500", 3),
                 cell(3, 0, "75 mm"), cell(3, 1, "mm"), cell(3, 2, "600", 3),
                 cell(4, 0, "Main pump 50 Hz"), cell(4, 1, "kW"), cell(4, 2, "10", 3),
                 cell(5, 0, "60 Hz"), cell(5, 1, "hp"), cell(5, 2, "15", 3)]
        labels = {f["parameter"] for f in self.normalize(cells, 6, 5)["facts"]}
        self.assertEqual(labels, {"Height / 45 and 60 mm", "Height / 75 mm", "Main pump 50 Hz", "Main pump / 60 Hz"})


if __name__ == "__main__":
    unittest.main()
