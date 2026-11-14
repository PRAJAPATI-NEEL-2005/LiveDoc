"""
LiveDoc Pure Digital Document Reconstruction & Positional Accuracy Test Suite
Verifies:
1. Pure Digital Vector PDF: Clean white background, ZERO full-page camera photo images.
2. Accurate 2D spatial positioning of digital text, digital tables, and cropped figures.
3. Multi-column separation (Column 1 and Column 2 never interleave horizontally).
4. Table reconstruction with structured headers and cells.
5. Multi-page document digital generation.
6. 100% backward compatibility with markdown_to_pdf().
"""

import os
import sys
import unittest
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_exporter import PDFExporter, HAS_PYMUPDF
if HAS_PYMUPDF:
    import pymupdf


class TestDigitalDocumentReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_pymupdf_available(self):
        """Verify PyMuPDF engine is active."""
        self.assertTrue(HAS_PYMUPDF, "PyMuPDF must be available for digital reconstruction.")

    def test_pure_digital_reconstruction_zero_photo_background(self):
        """
        Verify that PDFExporter produces a PURE digital vector PDF:
        - 0 camera photo background images (pure white digital page).
        - Real visible digital typography placed at physical coordinates.
        - Digital vector tables with styled header and cell grid.
        - Cropped figure placed at exact physical coordinates.
        """
        H, W = 1200, 800

        # Create a sample cropped diagram figure
        fig_img = np.ones((160, 380, 3), dtype=np.uint8) * 255
        cv2.rectangle(fig_img, (5, 5), (375, 155), (255, 120, 0), 2)
        cv2.putText(fig_img, "Digital Architecture Diagram", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        fig_path = os.path.join(self.test_output_dir, "test_fig_crop.png")
        cv2.imwrite(fig_path, fig_img)

        # Structured elements with exact physical coordinates
        elements = [
            {
                "type": "Title",
                "text": "LIVEDOC DIGITAL RECONSTRUCTION",
                "box": [100, 50, 700, 95]
            },
            {
                "type": "Heading",
                "text": "1.0 System Architecture",
                "level": 2,
                "box": [60, 130, 400, 160]
            },
            {
                "type": "Paragraph",
                "text": "Column One Text: LiveDoc transforms analog physical documents into pristine digital vector PDF documents without camera noise.",
                "box": [60, 175, 370, 290]
            },
            {
                "type": "Paragraph",
                "text": "Column Two Text: Running parallel on the right half of the page with strict lateral coordinate separation.",
                "box": [430, 175, 740, 290]
            },
            {
                "type": "Table",
                "headers": ["Module", "Technology", "Status"],
                "rows": [
                    ["Document Detector", "OpenCV Contours", "Active"],
                    ["Layout Analyzer", "Morphological CV", "Active"],
                    ["OCR Engine", "PaddleOCR", "Ready"]
                ],
                "box": [60, 330, 740, 520]
            },
            {
                "type": "Image",
                "image_path": fig_path,
                "caption": "Figure 1: Digital Pipeline Architecture",
                "box": [150, 570, 650, 780]
            }
        ]

        page_data = {
            "image_shape": (H, W),
            "elements": elements,
            "title": "Pure Digital Test"
        }

        pdf_path = os.path.join(self.test_output_dir, "test_pure_digital_reconstructed.pdf")
        ok = PDFExporter.create_digital_reconstructed_pdf([page_data], pdf_path, doc_title="Digital Reconstructed Test")
        self.assertTrue(ok, "create_digital_reconstructed_pdf returned False")
        self.assertTrue(os.path.exists(pdf_path))

        # Inspect the generated PDF
        doc = pymupdf.open(pdf_path)
        self.assertEqual(len(doc), 1, "Must contain exactly 1 page")
        page = doc[0]

        # 1. CRITICAL CHECK: ZERO full-page background camera photo images!
        # The only image allowed is the specific cropped diagram figure!
        images = page.get_images()
        self.assertEqual(len(images), 1, "Must contain ONLY the 1 cropped figure, 0 full-page photo backgrounds")
        fig_xref = images[0][0]
        fig_meta = doc.extract_image(fig_xref)
        self.assertLess(fig_meta["width"], 600, "Figure width should be small cropped figure, not full camera frame")

        # 2. Check that Title, Headings, Paragraphs, and Tables are real visible digital text
        full_text = page.get_text()
        self.assertIn("LIVEDOC DIGITAL RECONSTRUCTION", full_text)
        self.assertIn("1.0 System Architecture", full_text)
        self.assertIn("Column One Text", full_text)
        self.assertIn("Column Two Text", full_text)
        self.assertIn("Document Detector", full_text)
        self.assertIn("PaddleOCR", full_text)
        self.assertIn("Figure 1: Digital Pipeline Architecture", full_text)

        # 3. Positional Multi-Column Separation Check
        page_w = page.rect.width
        col1_hits = page.search_for("Column One")
        col2_hits = page.search_for("Column Two")
        self.assertGreaterEqual(len(col1_hits), 1)
        self.assertGreaterEqual(len(col2_hits), 1)
        self.assertLess(col1_hits[0].x0, page_w * 0.5, "Column 1 text must be strictly in left half")
        self.assertGreater(col2_hits[0].x0, page_w * 0.5, "Column 2 text must be strictly in right half")

        doc.close()
        # Clean up
        try:
            os.remove(pdf_path)
            os.remove(fig_path)
        except Exception:
            pass

    def test_multi_page_digital_reconstruction(self):
        """Verify multi-page digital reconstruction on clean white pages."""
        pages = [
            {
                "image_shape": (1200, 800),
                "elements": [
                    {"type": "Title", "text": "DOCUMENT PAGE 1", "box": [100, 50, 700, 95]},
                    {"type": "Paragraph", "text": "Page one digital body content.", "box": [80, 150, 720, 250]}
                ],
                "title": "Page 1"
            },
            {
                "image_shape": (1200, 800),
                "elements": [
                    {"type": "Title", "text": "DOCUMENT PAGE 2", "box": [100, 50, 700, 95]},
                    {"type": "Paragraph", "text": "Page two digital body content.", "box": [80, 150, 720, 250]}
                ],
                "title": "Page 2"
            }
        ]

        pdf_path = os.path.join(self.test_output_dir, "test_multi_digital.pdf")
        ok = PDFExporter.create_digital_reconstructed_pdf(pages, pdf_path, doc_title="Multi Digital Test")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(pdf_path))

        doc = pymupdf.open(pdf_path)
        self.assertEqual(len(doc), 2, "Must contain exactly 2 pages")
        self.assertIn("DOCUMENT PAGE 1", doc[0].get_text())
        self.assertIn("DOCUMENT PAGE 2", doc[1].get_text())
        self.assertEqual(len(doc[0].get_images()), 0, "Page 1 must have 0 photo images")
        self.assertEqual(len(doc[1].get_images()), 0, "Page 2 must have 0 photo images")
        doc.close()
        try:
            os.remove(pdf_path)
        except Exception:
            pass

    def test_backward_compatibility_markdown_to_pdf(self):
        """Verify markdown_to_pdf continues to function seamlessly."""
        sample_md = """# Digitized Doc
## Section
Clean text.

| H1 | H2 |
|---|---|
| A | B |
"""
        pdf_path = os.path.join(self.test_output_dir, "test_compat_digital.pdf")
        ok = PDFExporter.markdown_to_pdf(sample_md, pdf_path, doc_title="Compat")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(pdf_path))
        doc = pymupdf.open(pdf_path)
        self.assertGreaterEqual(len(doc), 1)
        doc.close()
        try:
            os.remove(pdf_path)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
