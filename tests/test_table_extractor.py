"""
LiveDoc Unit Tests for TableExtractor & Matrix Reconstructor
Verifies:
- Token row clustering by vertical coordinates
- Column slot alignment and header discovery
- Clean cell sanitization and pipe character escaping
- Markdown and HTML table synthesis
"""

import unittest
import numpy as np
from src.table_extractor import TableExtractor


class TestTableExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TableExtractor()
        self.dummy_crop = np.zeros((200, 400, 3), dtype=np.uint8)

    def test_empty_tokens_returns_empty_table(self):
        result = self.extractor.extract_table(self.dummy_crop, [])
        self.assertEqual(result["type"], "Table")
        self.assertEqual(result["headers"], [])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["markdown"], "")
        self.assertEqual(result["html"], "")

    def test_clean_cell_escapes_pipe_characters(self):
        raw = "Item | Description"
        cleaned = self.extractor._clean_cell(raw)
        self.assertEqual(cleaned, "Item / Description")

    def test_2x2_table_matrix_reconstruction(self):
        # Synthetic tokens forming a 2x2 table:
        # Row 0 (Headers): Item (x=50, y=20), Price (x=200, y=20)
        # Row 1 (Data):    Book (x=50, y=60), 25.00 (x=200, y=60)
        tokens = [
            {"text": "Item", "cx": 50, "cy": 20, "min_x": 30, "min_y": 10, "h": 20, "conf": 0.98},
            {"text": "Price", "cx": 200, "cy": 20, "min_x": 180, "min_y": 10, "h": 20, "conf": 0.99},
            {"text": "Book", "cx": 50, "cy": 60, "min_x": 30, "min_y": 50, "h": 20, "conf": 0.95},
            {"text": "25.00", "cx": 200, "cy": 60, "min_x": 180, "min_y": 50, "h": 20, "conf": 0.97},
        ]
        result = self.extractor.extract_table(self.dummy_crop, tokens)

        self.assertEqual(result["type"], "Table")
        self.assertEqual(result["headers"], ["Item", "Price"])
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0], ["Book", "25.00"])

        # Check Markdown table syntax
        self.assertIn("| Item | Price |", result["markdown"])
        self.assertIn("| :---: | :---: |", result["markdown"])
        self.assertIn("| Book | 25.00 |", result["markdown"])

        # Check HTML table generation
        self.assertIn("<table", result["html"])
        self.assertIn(">Item</th>", result["html"])
        self.assertIn(">Book</td>", result["html"])


if __name__ == "__main__":
    unittest.main()
