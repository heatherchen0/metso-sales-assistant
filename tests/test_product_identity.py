"""Invented brochure identities; no source answers or model calls."""
from copy import deepcopy
import unittest

from metso_assistant.product_identity import name_occurs, products_from_tool, source_product_identity


def document():
    return {"id": "helix-upgrade", "title": "helix upgrade", "source_file": "helix.pdf",
            "source_sha256": "a" * 64, "page_count": 3}


def chunk(text, *, page=1, heading=False, sha="a" * 64):
    return {"id": f"p{page}", "doc_id": "helix-upgrade", "page": page, "text": text,
            "blocks": [{"text": text, "label": "section_header" if heading else "text"}],
            "source_sha256": sha, "kind": "narrative", "evidence_status": "source_text"}


class ProductIdentityTests(unittest.TestCase):
    def test_source_heading_names_product_even_when_retrieved_capability_has_no_name(self):
        result = source_product_identity(document(), [chunk("Helix\u00ae upgrade", heading=True),
                                                       chunk("The mechanism improves mixing in an existing vessel.", page=2)])
        self.assertEqual(result["name"], "Helix\u00ae upgrade")
        self.assertEqual(result["aliases"], ["Helix\u00ae upgrade", "Helix"])
        self.assertEqual(result["identity_evidence"][0]["page"], 1)
        self.assertEqual(result["identity_evidence"][0]["quote"], result["name"])
        self.assertEqual(result["identity_evidence"][0]["source_sha256"], "a" * 64)
        self.assertEqual(result["authority"], "source_identity_only")
        self.assertTrue(name_occurs(result, "How does Helix work?"))
        self.assertFalse(name_occurs(result, "What other upgrades could help?"))

    def test_filename_and_metadata_without_actual_name_do_not_authorize_identity(self):
        result = source_product_identity(document(), [chunk("A mechanism improves mixing.", page=2)])
        self.assertIsNone(result["name"])
        self.assertEqual(result["identity_evidence"], [])
        self.assertEqual(result["status"], "unresolved")

    def test_wrong_hash_or_non_narrative_cannot_authorize_identity(self):
        wrong = chunk("Helix upgrade", heading=True, sha="b" * 64)
        table = chunk("Helix upgrade", heading=True)
        table.update(kind="table_context", evidence_status="raw_table")
        result = source_product_identity(document(), [wrong, table])
        self.assertIsNone(result["name"])

    def test_source_subject_fallback_for_injected_tools_is_not_a_filename_name_map(self):
        tool = {"data": {"passages": [{"doc_id": "helix-upgrade", "source_id": "S1", "text": "The Helix upgrade improves mixing."}]},
                "sources": [{"id": "S1", "page": 2, "file": "helix.pdf", "excerpt": "The Helix upgrade improves mixing."}]}
        record = products_from_tool(tool)[0]
        self.assertEqual(record["name"], "Helix upgrade")
        self.assertEqual(record["identity_evidence"][0]["quote"], "Helix upgrade")

    def test_short_code_and_marks_are_searchable_without_inventing_descriptive_aliases(self):
        doc = document()
        doc.update(id="nova-px-ds", title="nova px ds")
        row = chunk("Nova\u00ae PX-DS", heading=True)
        row["doc_id"] = doc["id"]
        record = source_product_identity(doc, [row])
        self.assertTrue(name_occurs(record, "What is the PX-DS layout?"))
        self.assertFalse(name_occurs(record, "What is the layout?"))
        self.assertNotIn("Nova pressure filter upgrade", record["aliases"])

    def test_same_canonical_name_does_not_collapse_different_source_editions(self):
        first = source_product_identity(document(), [chunk("Helix upgrade", heading=True)])
        other_doc = document()
        other_doc.update(id="helix-upgrade-b", source_file="helix-b.pdf", source_sha256="b" * 64)
        other_chunk = chunk("Helix upgrade", heading=True, sha="b" * 64)
        other_chunk["doc_id"] = other_doc["id"]
        second = source_product_identity(other_doc, [other_chunk])
        self.assertEqual(first["name"], second["name"])
        self.assertNotEqual(first["product_id"], second["product_id"])
        self.assertNotEqual(first["identity_evidence"], second["identity_evidence"])

    def test_authoritative_retrieval_catalogue_is_used_without_mutating_it(self):
        record = source_product_identity(document(), [chunk("Helix upgrade", heading=True)])
        original = deepcopy(record)
        actual = products_from_tool({"data": {"products": [record], "passages": []}})
        self.assertEqual(actual, [original])
        self.assertEqual(record, original)


if __name__ == "__main__":
    unittest.main()
