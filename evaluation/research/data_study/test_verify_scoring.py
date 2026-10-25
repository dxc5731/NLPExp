"""Focused synthetic checks for the small independent author crosscheck."""

import json
from pathlib import Path
import tempfile
import unittest

import verify_scoring as verification


class VerificationTests(unittest.TestCase):
    def test_eight_cases_match_manual_counts_and_author(self):
        cases = [verification.check_case(case) for case in verification.fixtures()]
        self.assertEqual(len(cases), 8)
        self.assertTrue(all(case["status"] == "PASS" for case in cases))
        for case in cases:
            self.assertEqual(case["manual_counts"], case["author_counts"])
            for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                self.assertAlmostEqual(case["ours_micro"][key], case["author_micro"][key], delta=1e-12)

    def test_merge_and_split_have_distinct_counts(self):
        source = {case["name"]: case for case in verification.fixtures()}
        merge = verification.check_case(source["merge_adjacent_entities"])
        split = verification.check_case(source["split_one_entity"])
        self.assertEqual(merge["ours_micro"]["fp"], 1)
        self.assertEqual(merge["ours_micro"]["fn"], 2)
        self.assertEqual(split["ours_micro"]["fp"], 2)
        self.assertEqual(split["ours_micro"]["fn"], 1)
        self.assertAlmostEqual(merge["ours_micro"]["f1"], 10 / 13, delta=1e-12)

    def test_identity_isolation_cannot_increase_true_positives(self):
        for case in verification.fixtures():
            if "identity_isolation" in case["name"]:
                actual = verification.check_case(case)["ours_micro"]
                self.assertEqual((actual["tp"], actual["fp"], actual["fn"]), (5, 1, 1))

    def test_incorrect_manual_expectation_is_not_silently_accepted(self):
        case = dict(verification.fixtures()[0])
        case["counts"] = (6, 0, 0)
        with self.assertRaisesRegex(AssertionError, "manually counted"):
            verification.check_case(case)

    def test_zero_entity_difference_is_explicit_not_numeric_failure(self):
        check = verification.protocol_checks()[0]
        self.assertEqual(check["status"], "EXPECTED_PROTOCOL_DIFFERENCE")
        self.assertEqual(check["ours"]["f1"], 0)
        self.assertEqual(check["author_original_helpers_zero_prf"], [0, 0, 0.0])
        self.assertEqual(check["author_full_entity_prefix"]["type"], "KeyError")

    def test_invalid_bio_is_rejected_and_author_behavior_disclosed(self):
        check = verification.protocol_checks()[1]
        self.assertEqual(check["status"], "EXPECTED_PROTOCOL_DIFFERENCE")
        self.assertEqual([row["ours"] for row in check["cases"]], ["REJECTED", "REJECTED"])
        self.assertEqual(check["cases"][0]["author_helper_entities"], [])
        self.assertEqual(check["cases"][1]["author_helper_entities"], [["Symptom", 0, 0]])

    def test_report_is_traceable_and_repeat_writes_never_overwrite(self):
        report = verification.run_checks()
        self.assertEqual(report["purpose"], "scorer_validation")
        self.assertEqual(report["sources"]["author_sha256"], verification.AUTHOR_SHA256)
        self.assertEqual(report["legal_case_count"], 8)
        self.assertEqual(report["protocol_check_count"], 2)
        with tempfile.TemporaryDirectory(prefix="scoring_verification_") as folder:
            first = verification.save_report(report, folder)
            before = first.read_bytes()
            second = verification.save_report(report, folder)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
