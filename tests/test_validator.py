"""
LiveDoc Unit Tests for DocumentValidator & Integrity Engine
Verifies:
- Confidence score auditing against minimum thresholds
- Arithmetic consistency checks on table data (Qty * Unit Price == Total)
- Grand total column summation checking
- Comprehensive validation report generation
"""

import unittest
from src.validator import DocumentValidator


class TestDocumentValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DocumentValidator(min_confidence_threshold=0.72)

    def test_validate_confidence_high_score(self):
        result = self.validator.validate_confidence("Total Amount", 0.95)
        self.assertFalse(result["uncertain"])
        self.assertEqual(result["status"], "Valid")

    def test_validate_confidence_low_score(self):
        result = self.validator.validate_confidence("S0m3BlurryTxt", 0.55)
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["status"], "Verification Required")

    def test_table_arithmetic_consistency_valid(self):
        table_element = {
            "type": "Table",
            "headers": ["Item", "Qty", "Price", "Total"],
            "rows": [
                ["Notebook", "2", "15.00", "30.00"],
                ["Pen", "5", "2.00", "10.00"]
            ]
        }
        flags = self.validator.validate_table_consistency(table_element)
        self.assertEqual(len(flags), 0)

    def test_table_arithmetic_consistency_discrepancy(self):
        table_element = {
            "type": "Table",
            "headers": ["Item", "Qty", "Price", "Total"],
            "rows": [
                ["Notebook", "2", "15.00", "50.00"]  # 2 * 15 = 30 != 50
            ]
        }
        flags = self.validator.validate_table_consistency(table_element)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["type"], "arithmetic_inconsistency")
        self.assertEqual(flags[0]["calculated_value"], 30.0)
        self.assertEqual(flags[0]["printed_value"], 50.0)

    def test_generate_report_overall_status(self):
        elements = [
            {"type": "Heading", "content": "Tax Invoice", "confidence": 0.98},
            {"type": "Paragraph", "content": "Faded footnote", "confidence": 0.65}
        ]
        report = self.validator.generate_report(elements)
        self.assertEqual(report["status"], "NEEDS_VERIFICATION")
        self.assertEqual(report["uncertain_items_count"], 1)


if __name__ == "__main__":
    unittest.main()
