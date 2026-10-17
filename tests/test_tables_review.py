"""Synthetic semantic regressions found in the independent table review."""

import unittest

from metso_assistant.tables import make_fact, normalize_table, split_quantities


def cell(row, column, value, span=1):
    return {"row_index": row, "col_index": column, "row_span": 1, "col_span": span,
            "cell_id": f"synthetic-cell-{row}-{column}", "text": value, "bbox": None}


class QuantityScopeTests(unittest.TestCase):
    def facts(self, raw):
        table = {"table_id": "synthetic-table", "product_family": "SyntheticFilter",
                 "source": {"file": "synthetic.pdf", "sha": "a" * 64, "page": 1}}
        fact = make_fact(table, "F1", "Motor", cell(1, 1, raw), "IEC", {"row": 1, "column": 1})
        return split_quantities(fact)

    def assert_not_unqualified_scalar(self, raw):
        facts = self.facts(raw)
        self.assertFalse(any(f["status"] == "reported" and f["value_kind"] == "scalar"
                             and not f.get("quantity_role") and not f.get("relation")
                             and not f.get("operator") and not f.get("qualifiers") for f in facts),
                         "A compound/qualified source must not become a bare scalar by matching a substring")

    def test_slash_alternatives_are_not_reduced_to_the_last_frequency(self):
        self.assert_not_unqualified_scalar("47/63 Hz")

    def test_motor_count_times_power_is_not_reduced_to_a_single_total(self):
        self.assert_not_unqualified_scalar("4 x 7 kW")

    def test_optional_power_is_not_a_second_unqualified_rating(self):
        self.assert_not_unqualified_scalar("7 kW (optional 11 kW)")

    def test_upper_bound_is_not_an_exact_rating(self):
        self.assert_not_unqualified_scalar("at most 7 kW")
        for fact in self.facts("at most 7 kW"):
            if fact["status"] == "reported" and fact["value_kind"] == "scalar":
                self.assertEqual(fact.get("operator"), "<=")


class ConfigurationIdentityTests(unittest.TestCase):
    def test_arbitrary_data_values_do_not_become_model_configuration_labels(self):
        rows = [["Width", "mm", "11", "22", "33"],
                ["Height", "mm", "101", "202", "303"],
                ["Depth", "mm", "1", "2", "3"]]
        cells = [cell(r, c, value) for r, row in enumerate(rows) for c, value in enumerate(row)]
        table = normalize_table("synthetic-table", cells, 3, 5,
                                {"file": "synthetic.pdf", "sha": "a" * 64, "page": 1},
                                {"parser": "synthetic", "version": "1", "models": {}, "settings": {}},
                                "b" * 64)
        self.assertFalse(any(f["status"] == "reported" and f["configuration"] in {"11", "22", "33"}
                             for f in table["facts"]),
                         "Width values are not source model identities")

    def test_an_omitted_model_row_cell_is_not_a_verified_source_blank(self):
        rows = [["Model", "Width", "Height", "Mass", "Power"],
                ["", "mm", "mm", "kg", "kW"],
                ["F1", "11", None, "22", "33"],
                ["F2", "12", "13", "23", "34"]]
        cells = [cell(r, c, value) for r, row in enumerate(rows) for c, value in enumerate(row)
                 if value is not None]
        table = normalize_table("synthetic-table", cells, 4, 5,
                                {"file": "synthetic.pdf", "sha": "a" * 64, "page": 1},
                                {"parser": "synthetic", "version": "1", "models": {}, "settings": {}},
                                "b" * 64)
        affected = [f for f in table["facts"] if f["configuration"] == "F1" and f["parameter"] == "Height"]
        self.assertTrue(not affected or all(f["status"] in {"ambiguous", "invalid"} for f in affected),
                        "An extraction gap is not evidence that a printed source cell is blank")


if __name__ == "__main__":
    unittest.main()
