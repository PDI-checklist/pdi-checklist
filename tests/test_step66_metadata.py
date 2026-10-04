import unittest
from metadata_engine import enrich_observation, normalize_checklist_date


class Step66MetadataTests(unittest.TestCase):
    def test_date_normalization(self):
        self.assertEqual(normalize_checklist_date("01-10-26"), "01-10-2026")
        self.assertEqual(normalize_checklist_date("1/10/2026"), "01-10-2026")

    def test_metadata_is_enriched_without_changing_observation(self):
        record = {"Observation": "Door bush missing"}
        mapping = {"door bush missing": ("Integration", "PDI Stage", "Missing Part")}
        out = enrich_observation(record, mapping=mapping, inspector="S. Harish", closure_date="01-10-2026")
        self.assertEqual(out["Observation"], "Door bush missing")
        self.assertEqual(out["Department"], "Integration")
        self.assertEqual(out["Station"], "PDI Stage")
        self.assertEqual(out["Defect Category"], "Missing Part")
        self.assertEqual(out["Status"], "Closed")
        self.assertEqual(out["Cleared by"], "S. Harish")
        self.assertEqual(out["Closure date"], "01-10-2026")
        self.assertEqual(out["Closure Remarks"], "Verified OK")

    def test_missing_mapping_does_not_drop_observation(self):
        out = enrich_observation({"Observation": "New wording"}, mapping={}, inspector="Harish", closure_date="01-10-2026")
        self.assertEqual(out["Observation"], "New wording")
        self.assertEqual(out["Department"], "")
        self.assertEqual(out["Station"], "PDI Stage")
        self.assertEqual(out["Status"], "Closed")


if __name__ == "__main__":
    unittest.main()
