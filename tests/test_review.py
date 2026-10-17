from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from metso_assistant.review import (
    apply_reviews, content_hash, fingerprint_candidate, main, review_summary, source_code_hash,
    validate_manifest,
)


def candidate():
    return {
        "source": {"file": "synthetic.pdf", "sha": "a" * 64, "page": 2},
        "table_id": "synthetic-table",
        "parser_fingerprint": {"parser": "synthetic-parser", "version": "1",
                               "models": {"layout": "b" * 64}, "settings": {"mode": "native"}},
        "normalizer_version": "synthetic-normalizer-1",
        "normalized_table": {"raw_table_sha256": "c" * 64,
                             "rows": [["Model", "Width"], ["F1", "10"]],
                             "geometry_validation": {"status": "valid"}},
        "facts": [{"fact_id": "width-f1", "product_family": "SyntheticFilter",
                   "configuration": "F1", "parameter": "width", "value": 10,
                   "unit": "mm", "status": "reported", "issues": [],
                   "source_refs": [{"page": 2, "table_id": "synthetic-table", "row": 1, "column": 1}]}],
        "issues": [],
    }


def decision(table, action="approve", fact_id=None):
    record = {"table_fingerprint": fingerprint_candidate(table),
              "scope": "fact" if fact_id else "table", "decision": action,
              "reviewer": "synthetic-reviewer", "reviewed_at_utc": "2026-01-01T12:00:00Z",
              "reason": "Synthetic source inspection"}
    if fact_id:
        record["fact_id"] = fact_id
    return record


def ledger(*records):
    return {"schema_version": 1, "decisions": list(records)}


class FingerprintTests(unittest.TestCase):
    def test_object_key_order_does_not_change_content_identity(self):
        self.assertEqual(content_hash({"b": [2, 3], "a": 1}), content_hash({"a": 1, "b": [2, 3]}))

    def test_checkout_line_endings_preserve_implementation_bound_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "normalizer.py"
            path.write_bytes(b"# implementation\nUNIT = 'mm'\n")
            table = candidate()
            table["normalizer_version"] = source_code_hash(path)
            approval = ledger(decision(table))
            path.write_bytes(b"# implementation\r\nUNIT = 'mm'\r\n")
            relocated = deepcopy(table)
            relocated["normalizer_version"] = source_code_hash(path)
            self.assertEqual(apply_reviews(relocated, approval)["facts"][0]["review_status"], "verified")
            path.write_bytes(b"# implementation\r\nUNIT = 'cm'\r\n")
            changed = deepcopy(table)
            changed["normalizer_version"] = source_code_hash(path)
            self.assertEqual(apply_reviews(changed, approval)["facts"][0]["review_status"], "pending")

    def test_source_code_identity_requires_utf8_and_preserves_non_newline_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "normalizer.py"
            path.write_bytes(b"UNIT = 'mm'\n")
            original = source_code_hash(path)
            path.write_bytes(b"UNIT  = 'mm'\n")
            self.assertNotEqual(source_code_hash(path), original)
            path.write_bytes(b"# invalid utf8: \xff\n")
            with self.assertRaises(UnicodeDecodeError):
                source_code_hash(path)

    def test_all_evidence_dimensions_invalidate_prior_approval(self):
        original = candidate()
        approval = ledger(decision(original))
        variants = []
        edits = [
            lambda t: t["source"].update(sha="d" * 64),
            lambda t: t["source"].update(page=3),
            lambda t: t.update(table_id="other-table"),
            lambda t: t["normalized_table"].update(raw_table_sha256="d" * 64),
            lambda t: t["normalized_table"]["rows"][1].__setitem__(1, "11"),
            lambda t: t["facts"][0].update(value=11),
            lambda t: t["facts"][0].update(configuration="F2"),
            lambda t: t["facts"][0].update(unit="cm"),
            lambda t: t["facts"][0]["source_refs"][0].update(row=2),
            lambda t: t["parser_fingerprint"]["settings"].update(mode="other"),
            lambda t: t["parser_fingerprint"]["models"].update(layout="d" * 64),
            lambda t: t.update(normalizer_version="synthetic-normalizer-2"),
        ]
        for edit in edits:
            changed = deepcopy(original)
            edit(changed)
            variants.append(changed)
        for index, changed in enumerate(variants):
            with self.subTest(evidence_dimension=index):
                self.assertNotEqual(fingerprint_candidate(changed), fingerprint_candidate(original))
                result = apply_reviews(changed, approval)
                self.assertEqual(result["facts"][0]["review_status"], "pending")
                self.assertEqual(result["review_diagnostics"]["matched_decisions"], 0)

    def test_cached_fingerprint_cannot_hide_changed_facts(self):
        table = candidate()
        table["table_fingerprint"] = fingerprint_candidate(table)
        table["facts"][0]["value"] = 999
        with self.assertRaisesRegex(ValueError, "does not match incoming evidence"):
            apply_reviews(table, ledger())

    def test_raw_digest_is_required_and_non_finite_content_is_rejected(self):
        table = candidate()
        del table["normalized_table"]["raw_table_sha256"]
        with self.assertRaisesRegex(ValueError, "raw_table_sha256"):
            fingerprint_candidate(table)
        for invalid in (float("nan"), float("inf"), {1: "non-string key"}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    content_hash(invalid)


class ReviewDecisionTests(unittest.TestCase):
    def test_approved_missing_cell_stays_null_and_zero_stays_zero(self):
        table = candidate()
        table["facts"][0]["value"] = 0
        missing = deepcopy(table["facts"][0])
        missing.update(fact_id="missing-f1", value=None, unit=None, status="missing")
        table["facts"].append(missing)
        before = deepcopy(table)
        result = apply_reviews(table, ledger(decision(table)))
        self.assertEqual(result["review_status"], "verified")
        self.assertEqual(result["facts"][0]["value"], 0)
        self.assertIsNone(result["facts"][1]["value"])
        self.assertEqual(result["facts"][1]["status"], "missing")
        self.assertEqual(table, before)

    def test_approval_cannot_clear_ambiguous_or_invalid_status(self):
        for status, expected in (("ambiguous", "ambiguous"), ("invalid", "pending")):
            with self.subTest(status=status):
                table = candidate()
                table["facts"][0]["status"] = status
                result = apply_reviews(table, ledger(decision(table)))
                self.assertEqual(result["facts"][0]["status"], status)
                self.assertEqual(result["facts"][0]["review_status"], expected)
                self.assertTrue(result["facts"][0]["review_blockers"])

    def test_scoped_blocker_does_not_block_unaffected_fact_or_clear_warning(self):
        table = candidate()
        second = deepcopy(table["facts"][0])
        second.update(fact_id="width-f2", configuration="F2",
                      issues=[{"code": "printed_unit_warning", "severity": "warning"}])
        table["facts"].append(second)
        table["issues"] = [{"code": "ambiguous_association", "severity": "blocker", "fact_ids": ["width-f1"]}]
        result = apply_reviews(table, ledger(decision(table)))
        self.assertEqual(result["facts"][0]["review_status"], "ambiguous")
        self.assertEqual(result["facts"][1]["review_status"], "verified")
        self.assertEqual(result["facts"][1]["issues"], second["issues"])

    def test_fact_approval_cannot_override_global_rejection_or_ambiguity(self):
        table = candidate()
        for action, expected in (("reject", "rejected"), ("ambiguous", "ambiguous")):
            with self.subTest(action=action):
                result = apply_reviews(table, ledger(decision(table, action), decision(table, fact_id="width-f1")))
                self.assertEqual(result["facts"][0]["review_status"], expected)

    def test_fact_rejection_can_restrict_table_approval(self):
        table = candidate()
        result = apply_reviews(table, ledger(decision(table), decision(table, "reject", "width-f1")))
        self.assertEqual(result["facts"][0]["review_status"], "rejected")

    def test_old_annotation_is_not_an_approval_without_a_matching_ledger(self):
        table = candidate()
        reviewed = apply_reviews(table, ledger(decision(table)))
        again = apply_reviews(reviewed)
        self.assertEqual(fingerprint_candidate(reviewed), fingerprint_candidate(table))
        self.assertEqual(again["facts"][0]["review_status"], "pending")
        self.assertIsNone(again["facts"][0]["review"])

    def test_unknown_fact_and_duplicate_candidate_ids_are_rejected(self):
        table = candidate()
        with self.assertRaisesRegex(ValueError, "unknown fact_id"):
            apply_reviews(table, ledger(decision(table, fact_id="absent")))
        table["facts"].append(deepcopy(table["facts"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate fact_id"):
            apply_reviews(table)

    def test_issue_scope_must_identify_existing_facts(self):
        table = candidate()
        table["issues"] = [{"code": "ambiguous_association", "severity": "blocker", "fact_ids": ["absent"]}]
        with self.assertRaisesRegex(ValueError, "existing facts"):
            apply_reviews(table)


class LedgerTests(unittest.TestCase):
    def test_conflicting_decisions_and_duplicate_record_ids_are_rejected(self):
        table = candidate()
        with self.assertRaisesRegex(ValueError, "Duplicate review decision"):
            validate_manifest(ledger(decision(table), decision(table, "reject")))
        first = dict(decision(table), decision_id="same-id")
        second = dict(decision(table, fact_id="width-f1"), decision_id="same-id")
        with self.assertRaisesRegex(ValueError, "Duplicate decision_id"):
            validate_manifest(ledger(first, second))

    def test_review_is_not_a_replacement_answer_channel(self):
        record = dict(decision(candidate()), replacement_value=42)
        with self.assertRaisesRegex(ValueError, "replacement values"):
            validate_manifest(ledger(record))

    def test_malformed_hash_and_unattributed_or_non_utc_decisions_are_rejected(self):
        for key, value in (("table_fingerprint", "bad-hash"), ("reviewer", ""),
                           ("reviewed_at_utc", "2026-01-01T12:00:00"),
                           ("reviewed_at_utc", "2026-01-01T12:00:00+02:00")):
            with self.subTest(field=key, value=value):
                record = dict(decision(candidate()), **{key: value})
                with self.assertRaises(ValueError):
                    validate_manifest(ledger(record))

    def test_public_summary_does_not_publish_values_identities_or_notes(self):
        table = candidate()
        reviewed = apply_reviews(table, ledger(decision(table)))
        summary = review_summary([reviewed])
        self.assertEqual(summary["facts"]["verified"], 1)
        encoded = json.dumps(summary)
        for private in ("SyntheticFilter", "synthetic-reviewer", "Synthetic source inspection", "width-f1"):
            self.assertNotIn(private, encoded)

    def test_cli_rejects_duplicate_json_keys_without_modifying_the_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            content = '{"schema_version":1,"decisions":[],"decisions":[]}'
            path.write_text(content, encoding="utf-8")
            error = io.StringIO()
            with redirect_stderr(error), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate", str(path)]), 2)
            self.assertIn("Duplicate JSON key", error.getvalue())
            self.assertEqual(path.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
