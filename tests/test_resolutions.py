from copy import deepcopy
import unittest

from metso_assistant.resolutions import apply_resolutions, fingerprint_conflict, validate_resolution_config
from metso_assistant.review import apply_reviews, fingerprint_candidate


def fixture(alternate_value=41):
    source = {"file": "synthetic.pdf", "sha": "a" * 64, "page": 2}
    primary = {"fact_id": "synthetic-fact", "product_family": "SyntheticFilter",
               "configuration": "F1", "parameter": "Width", "value": 37,
               "unit": "mm", "raw_unit": "mm", "raw_cell_text": "37",
               "status": "reported", "value_kind": "scalar", "issues": [],
               "source_refs": [{**source, "cell_id": "primary-cell"}], "review_status": "pending"}
    alternate = deepcopy(primary)
    alternate.update(fact_id="alternate-fact", value=alternate_value, raw_cell_text=str(alternate_value))
    alternate["source_refs"][0]["cell_id"] = "alternate-cell"
    fingerprint = fingerprint_conflict(source, primary, alternate)
    conflict = {"fact_id": primary["fact_id"], "candidate": deepcopy(primary), "alternate": alternate,
                "conflict_fingerprint": fingerprint,
                "identity": ["F1", "width", "mm", None, "scalar", None],
                "alternate_identity": ["F1", "width", "mm", None, "scalar", None]}
    primary["issues"].append({"code": "source_geometry_disagreement", "severity": "blocker",
                              "conflict_fingerprint": fingerprint})
    table = {"table_id": "synthetic-table", "source": source, "facts": [primary], "issues": [],
             "raw_table_sha256": "b" * 64, "normalizer_version": "synthetic-1",
             "parser_fingerprint": {"parser": "synthetic", "models": {}, "settings": {}},
             "normalized_table": {"rows": [["Width", "37"]],
                                  "geometry_crosscheck": {"conflicts": [conflict]}}}
    return table, conflict


def rule(table, conflict, choice="candidate"):
    return {"source_sha256": table["source"]["sha"], "page": table["source"]["page"],
            "table_id": table["table_id"], "fact_id": conflict["fact_id"],
            "conflict_fingerprint": conflict["conflict_fingerprint"], "decision": choice,
            "reviewer": "synthetic-reviewer", "reviewed_at_utc": "2026-01-01T12:00:00Z",
            "reason": "Synthetic original-page inspection"}


def config(*rules):
    return {"schema_version": 1, "rules": list(rules)}


class ResolutionSelectionTests(unittest.TestCase):
    def test_no_rule_preserves_disagreement_and_does_not_approve(self):
        table, _ = fixture()
        table["facts"][0]["review_status"] = "verified"
        before = deepcopy(table)
        result = apply_resolutions(table)
        self.assertEqual(result["facts"][0]["value"], 37)
        self.assertEqual(result["facts"][0]["issues"], before["facts"][0]["issues"])
        self.assertEqual(result["facts"][0]["review_status"], "pending")
        self.assertEqual(result["resolution_diagnostics"]["applied"], 0)
        self.assertEqual(table, before)

    def test_candidate_choice_removes_only_its_corresponding_blocker(self):
        table, conflict = fixture()
        unrelated = {"code": "parser_disagreement", "severity": "blocker", "conflict_fingerprint": "c" * 64}
        table["facts"][0]["issues"].append(unrelated)
        result = apply_resolutions(table, config(rule(table, conflict)))
        self.assertEqual(result["facts"][0]["value"], 37)
        self.assertIn(unrelated, result["facts"][0]["issues"])
        self.assertFalse(any(i.get("severity") == "blocker" and i.get("conflict_fingerprint") == conflict["conflict_fingerprint"]
                             for i in result["facts"][0]["issues"]))
        self.assertEqual(result["normalized_table"]["geometry_crosscheck"]["conflicts"], [conflict])
        self.assertEqual(result["facts"][0]["review_status"], "pending")

    def test_alternate_choice_reads_existing_snapshot_and_keeps_original_identity(self):
        table, conflict = fixture()
        before = deepcopy(table)
        result = apply_resolutions(table, config(rule(table, conflict, "alternate")))
        fact = result["facts"][0]
        self.assertEqual((fact["fact_id"], fact["configuration"], fact["parameter"]), ("synthetic-fact", "F1", "Width"))
        self.assertEqual((fact["value"], fact["raw_cell_text"]), (41, "41"))
        self.assertEqual(fact["source_refs"], conflict["alternate"]["source_refs"])
        self.assertEqual(fact["review_status"], "pending")
        self.assertEqual(table, before)
        self.assertNotEqual(fingerprint_candidate(result), fingerprint_candidate(table))

    def test_unrelated_intrinsic_blocker_survives_choice_and_prevents_review_approval(self):
        table, conflict = fixture()
        intrinsic = {"code": "structural_gap", "severity": "blocker"}
        table["facts"][0]["issues"].append(intrinsic)
        result = apply_resolutions(table, config(rule(table, conflict, "alternate")))
        self.assertIn(intrinsic, result["facts"][0]["issues"])
        approval = {"schema_version": 1, "decisions": [{"table_fingerprint": fingerprint_candidate(result),
                    "scope": "table", "decision": "approve", "reviewer": "synthetic-reviewer",
                    "reviewed_at_utc": "2026-01-01T12:00:00Z", "reason": "Synthetic review"}]}
        self.assertNotEqual(apply_reviews(result, approval)["facts"][0]["review_status"], "verified")

    def test_intrinsic_ambiguous_status_cannot_be_replaced_by_clean_alternate(self):
        table, conflict = fixture()
        table["facts"][0]["status"] = "ambiguous"
        conflict["candidate"]["status"] = "ambiguous"
        fingerprint = fingerprint_conflict(table["source"], conflict["candidate"], conflict["alternate"])
        conflict["conflict_fingerprint"] = fingerprint
        table["facts"][0]["issues"][0]["conflict_fingerprint"] = fingerprint
        result = apply_resolutions(table, config(rule(table, conflict, "alternate")))
        self.assertEqual(result["facts"][0]["status"], "ambiguous")
        self.assertEqual(result["facts"][0]["value"], 37)
        self.assertEqual(result["resolution_diagnostics"]["applied"], 0)


class ResolutionGuardsTests(unittest.TestCase):
    def test_every_rule_guard_is_required_to_match(self):
        table, conflict = fixture()
        for key, value in (("source_sha256", "c" * 64), ("page", 3), ("table_id", "other"),
                           ("fact_id", "absent"), ("conflict_fingerprint", "c" * 64)):
            with self.subTest(guard=key):
                changed = dict(rule(table, conflict, "alternate"), **{key: value})
                result = apply_resolutions(table, config(changed))
                self.assertEqual(result["facts"][0]["value"], 37)
                self.assertEqual(result["resolution_diagnostics"]["applied"], 0)
                self.assertTrue(result["resolution_diagnostics"]["unmatched"])
                self.assertTrue(any(i["severity"] == "blocker" for i in result["facts"][0]["issues"]))

    def test_snapshot_edit_cannot_reuse_its_old_hash(self):
        table, conflict = fixture()
        selected_rule = rule(table, conflict, "alternate")
        conflict["alternate"]["value"] = 999
        result = apply_resolutions(table, config(selected_rule))
        self.assertEqual(result["facts"][0]["value"], 37)
        self.assertEqual(result["resolution_diagnostics"]["blocked"][0]["reason"], "conflict_snapshot_hash_mismatch")

    def test_current_candidate_edit_cannot_be_reverted_by_an_old_conflict(self):
        table, conflict = fixture()
        table["facts"][0]["value"] = 999
        result = apply_resolutions(table, config(rule(table, conflict, "alternate")))
        self.assertEqual(result["facts"][0]["value"], 999)
        self.assertEqual(result["resolution_diagnostics"]["blocked"][0]["reason"], "current_candidate_payload_changed")

    def test_different_configuration_is_not_a_value_resolution(self):
        table, conflict = fixture()
        conflict["alternate"]["configuration"] = "F2"
        fingerprint = fingerprint_conflict(table["source"], conflict["candidate"], conflict["alternate"])
        conflict["conflict_fingerprint"] = fingerprint
        table["facts"][0]["issues"][0]["conflict_fingerprint"] = fingerprint
        result = apply_resolutions(table, config(rule(table, conflict, "alternate")))
        self.assertEqual(result["resolution_diagnostics"]["blocked"][0]["reason"], "semantic_identity_mismatch")

    def test_inconsistent_choices_for_one_fact_stay_blocked(self):
        table, first = fixture()
        second = deepcopy(first)
        second["alternate"]["value"] = 43
        second["alternate"]["raw_cell_text"] = "43"
        second["conflict_fingerprint"] = fingerprint_conflict(table["source"], second["candidate"], second["alternate"])
        table["normalized_table"]["alternate_parser_comparison"] = {"conflicts": [second]}
        table["facts"][0]["issues"].append({"code": "parser_disagreement", "severity": "blocker",
                                           "conflict_fingerprint": second["conflict_fingerprint"]})
        result = apply_resolutions(table, config(rule(table, first, "alternate"), rule(table, second, "alternate")))
        self.assertEqual(result["facts"][0]["value"], 37)
        self.assertEqual(result["resolution_diagnostics"]["applied"], 0)
        self.assertEqual(len([i for i in result["facts"][0]["issues"] if i["severity"] == "blocker"]), 2)

    def test_matching_rule_cannot_remove_a_different_issue_with_the_same_hash(self):
        table, conflict = fixture()
        table["facts"][0]["issues"] = [{"code": "numeric_separator_ambiguity", "severity": "blocker",
                                         "conflict_fingerprint": conflict["conflict_fingerprint"]}]
        result = apply_resolutions(table, config(rule(table, conflict)))
        self.assertEqual(result["resolution_diagnostics"]["applied"], 0)
        self.assertEqual(result["facts"][0]["issues"][0]["code"], "numeric_separator_ambiguity")


class ResolutionConfigTests(unittest.TestCase):
    def test_duplicate_targets_and_replacement_payloads_are_rejected(self):
        table, conflict = fixture()
        record = rule(table, conflict)
        with self.assertRaisesRegex(ValueError, "Duplicate resolution"):
            validate_resolution_config(config(record, record))
        with self.assertRaisesRegex(ValueError, "replacement values"):
            validate_resolution_config(config(dict(record, replacement_value=999)))

    def test_unattributed_and_non_utc_choices_are_rejected(self):
        table, conflict = fixture()
        for key, value in (("reviewer", ""), ("reviewed_at_utc", "2026-01-01T12:00:00"),
                           ("reason", ""), ("page", True)):
            with self.subTest(field=key):
                with self.assertRaises(ValueError):
                    validate_resolution_config(config(dict(rule(table, conflict), **{key: value})))


if __name__ == "__main__":
    unittest.main()
