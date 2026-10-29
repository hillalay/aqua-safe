import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.video_classification.dataset import validate_split_leakage
from src.video_review.inventory import stable_id
from src.video_review.diagnose_eligibility import diagnostic_rows
from src.video_review.triage_reviews import note_evidence, triage
from src.video_review.source_analysis import components
from src.video_review.reviews import atomic_write_json, export_training_metadata, merge_reviews, training_reason, validate_review


class VideoReviewTests(unittest.TestCase):
    def master_row(self):
        return {
            "video_id": "V001", "filename": "a.mp4", "path": "data\\raw\\kelebek\\a.mp4",
            "candidate_label": "butterfly", "style": "unknown", "behavior": "unknown",
            "swimmer_count": "unknown", "mixed_styles": "unknown", "review_status": "unreviewed",
            "source_id": "unknown", "session_id": "unknown", "notes": "", "split": "unassigned",
        }

    def valid_review(self):
        return {"video_id": "V001", "style": "butterfly", "behavior": "normal", "swimmer_count": "single",
                "mixed_styles": "no", "review_status": "verified", "source_id": "SRC1", "session_id": "S1", "notes": "ok"}

    def test_label_change_and_drowning_is_not_auto_verified(self):
        review = validate_review(self.valid_review(), {"V001"})
        self.assertEqual(merge_reviews([self.master_row()], {"V001": review})[0]["style"], "butterfly")
        candidate = validate_review({**self.valid_review(), "behavior": "drowning_candidate"}, {"V001"})
        self.assertEqual(candidate["review_status"], "ambiguous")

    def test_video_and_person_style_verification_are_separate(self):
        review = validate_review({**self.valid_review(), "swimmer_count": "unknown",
                                  "video_style_verification": "verified",
                                  "person_style_verification": "unknown"}, {"V001"})
        self.assertEqual(review["video_style_verification"], "verified")
        self.assertEqual(review["person_style_verification"], "unknown")
        self.assertIsNone(training_reason(review))

    def test_atomic_export_and_training_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_write_json(root / "state.json", {"V001": self.valid_review()})
            self.assertEqual(json.loads((root / "state.json").read_text())["V001"]["style"], "butterfly")
            report = export_training_metadata([self.master_row()], {"V001": self.valid_review()}, root)
            self.assertEqual(report["training_eligible"], 1)
            with (root / "training_metadata.csv").open(encoding="utf-8-sig") as source:
                self.assertEqual(next(csv.DictReader(source))["video_id"], "V001")

    def test_file_matching_preserves_previous_then_legacy_id(self):
        path = "data\\raw\\kelebek\\a.mp4"
        previous = {path: {"video_id": "KEEP"}}
        legacy = {"data\\single\\kelebek\\a.mp4": {"video_id": "V001"}}
        self.assertEqual(stable_id(path, legacy, previous), "KEEP")
        self.assertEqual(stable_id(path, legacy, {}), "V001")

    def test_unknown_group_is_not_treated_as_independent_and_known_group_cannot_leak(self):
        master = self.master_row()
        review = {**self.valid_review(), "source_id": "unknown"}
        report = export_training_metadata([master], {"V001": review}, Path(tempfile.mkdtemp()))
        self.assertEqual(report["training_eligible"], 0)
        rows = [
            {"video_id": "V1", "swimmer_id": "P1", "session_id": "S1", "split": "train"},
            {"video_id": "V2", "swimmer_id": "P1", "session_id": "S1", "split": "test"},
        ]
        with self.assertRaises(ValueError):
            validate_split_leakage(rows)

    def test_diagnostic_separates_label_and_split_eligibility(self):
        review = {**self.valid_review(), "source_id": "unknown", "session_id": "unknown"}
        row = diagnostic_rows([self.master_row()], {"V001": review})[0]
        self.assertTrue(row["style_label_eligible"])
        self.assertFalse(row["split_eligible"])
        self.assertEqual(row["exclusion_reasons"], "source_unknown|session_unknown")

    def test_triage_only_proposes_clean_ambiguous_records(self):
        clean = {**self.valid_review(), "review_status": "ambiguous", "source_id": "unknown", "session_id": "unknown"}
        self.assertEqual(triage(clean), "bulk_confirmation_candidate")
        self.assertEqual(triage({**clean, "mixed_styles": "yes"}), "deferred_complex")
        self.assertEqual(triage({**clean, "swimmer_count": "multiple"}), "same_style_multiple_swimmers")
        self.assertEqual(note_evidence({**clean, "notes": "Karma stil içeren video"}),
                         "note_conflicts_with_mixed_styles_no")

    def test_similarity_components_do_not_invent_unconnected_groups(self):
        self.assertEqual(components([("A", "B"), ("B", "C"), ("D", "E")]), [["A", "B", "C"], ["D", "E"]])


if __name__ == "__main__":
    unittest.main()
