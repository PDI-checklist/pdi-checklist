import unittest

import cv2
import numpy as np

from row_engine import detect_checklist_rows


def make_table(scale=1, skew=False):
    width, height = 800 * scale, 560 * scale
    image = np.full((height, width, 3), 255, np.uint8)
    x = [40, 110, 470, 550, 630, 760]
    y = [40 + 35 * i for i in range(13)]
    x = [v * scale for v in x]
    y = [v * scale for v in y]
    for xx in x:
        cv2.line(image, (xx, y[0]), (xx, y[-1]), (0, 0, 0), max(1, scale))
    for yy in y:
        cv2.line(image, (x[0], yy), (x[-1], yy), (0, 0, 0), max(1, scale))
    if skew:
        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        dst = np.float32([[35, 15], [width - 30, 40], [width - 5, height - 10], [10, height - 30]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, matrix, (width, height), borderValue=(245, 245, 245))
    return image, x, y


class RowEngineTests(unittest.TestCase):
    def test_table_detection(self):
        image, x, y = make_table()
        result = detect_checklist_rows(image, perspective=False)
        self.assertLessEqual(abs(result.table_bbox[0] - x[0]), 4)
        self.assertLessEqual(abs(result.table_bbox[2] - x[-1]), 4)

    def test_row_segmentation(self):
        image, _, _ = make_table()
        result = detect_checklist_rows(image, perspective=False)
        self.assertEqual(len(result.rows), 12)
        self.assertEqual([row.row_index for row in result.rows], list(range(12)))
        self.assertTrue(all(row.boundaries[1] > row.boundaries[0] for row in result.rows))

    def test_column_detection(self):
        image, x, _ = make_table()
        result = detect_checklist_rows(image, perspective=False)
        self.assertEqual(len(result.column_boundaries), len(x))
        self.assertTrue(all(b > a for a, b in zip(result.column_boundaries, result.column_boundaries[1:])))
        self.assertTrue(all(abs(found - expected) <= 2 for found, expected in zip(result.column_boundaries, x)))

    def test_not_ok_cell_geometry(self):
        image, x, y = make_table()
        result = detect_checklist_rows(image, perspective=False)
        actual = result.rows[0].not_ok_bbox
        expected = (x[3], y[0], x[4], y[1])
        self.assertTrue(all(abs(found - target) <= 2 for found, target in zip(actual, expected)))

    def test_different_image_resolutions(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                image, _, _ = make_table(scale=scale)
                result = detect_checklist_rows(image, perspective=False)
                self.assertEqual(len(result.rows), 12)
                self.assertGreater(result.table_bbox[2] - result.table_bbox[0], image.shape[1] * 0.8)

    def test_perspective_angle_variation(self):
        image, _, _ = make_table(skew=True)
        result = detect_checklist_rows(image, perspective=True)
        self.assertEqual(len(result.rows), 12)
        self.assertEqual(result.rows[0].not_ok_bbox[2] > result.rows[0].not_ok_bbox[0], True)


if __name__ == "__main__":
    unittest.main()
