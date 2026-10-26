"""Small synthetic diagnostics tests; no SAM weights or GPU inference required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).with_name("predict_orthomosaic.py")


class TinyProbabilityModel:
    def predict_proba(self, features):
        green = np.where(features[:, 0] >= 150, 0.9, 0.1)
        return np.column_stack((1 - green, green))


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("orthomosaic_diagnostics", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        with patch("pickle.load", side_effect=AssertionError("Import must not load model weights")):
            spec.loader.exec_module(cls.module)

    def test_statistics_use_independent_sam_and_final_denominators(self):
        sam = np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8)
        final = np.array([[1, 2, 0], [2, 0, 0]], dtype=np.uint8)
        stats = self.module.summarize_masks(sam, final, "rice.png")
        expected = dict(image_name="rice.png", image_width=3, image_height=2,
                        total_pixels=6, sam_vegetation_pixels=4, green_pixels=1,
                        yellow_pixels=2, other_pixels=3)
        for key, value in expected.items():
            self.assertEqual(stats[key], value, key)
        self.assertAlmostEqual(stats["vegetation_coverage_percent"], 100 * 4 / 6)
        self.assertAlmostEqual(stats["yellow_ratio_percent"], 100 * 2 / 3)
        self.assertIn("metric_definitions", stats)

    def test_no_vegetation_produces_finite_zero_ratios(self):
        zero = np.zeros((2, 3), dtype=np.uint8)
        stats = self.module.summarize_masks(zero, zero, "empty.png")
        self.assertEqual(stats["vegetation_coverage_percent"], 0)
        self.assertEqual(stats["yellow_ratio_percent"], 0)
        self.assertEqual(stats["other_pixels"], 6)
        json.dumps(stats, allow_nan=False)

    def test_overlay_blends_only_yellow_pixels_in_rgb(self):
        rgb = np.array([[[10, 50, 200], [80, 100, 120]]], dtype=np.uint8)
        original = rgb.copy()
        result = self.module.create_abnormal_overlay(rgb, np.array([[0, 255]], dtype=np.uint8))
        np.testing.assert_array_equal(result, [[[10, 50, 200], [168, 50, 60]]])
        np.testing.assert_array_equal(rgb, original)

    def test_output_files_have_original_dimensions_and_binary_masks(self):
        rgb = np.array([[[10, 50, 200], [80, 100, 120], [25, 180, 40]]], dtype=np.uint8)
        sam = np.array([[0, 1, 1]], dtype=np.uint8)
        final = np.array([[0, 2, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new_results"
            self.module.save_diagnostic_outputs(rgb, sam, final, "rice.png", out)
            self.assertEqual({p.name for p in out.iterdir()}, {
                "plant_mask.png", "yellow_mask.png", "abnormal_overlay.png",
                "Disease Detection.png", "summary.json"})
            for path in out.glob("*.png"):
                with Image.open(path) as im:
                    self.assertEqual(im.size, (3, 1))
            np.testing.assert_array_equal(np.array(Image.open(out / "plant_mask.png")), [[0, 255, 255]])
            np.testing.assert_array_equal(np.array(Image.open(out / "yellow_mask.png")), [[0, 255, 0]])
            overlay = np.array(Image.open(out / "abnormal_overlay.png"))
            np.testing.assert_array_equal(overlay[0, [0, 2]], rgb[0, [0, 2]])
            legacy = np.array(Image.open(out / "Disease Detection.png"))
            np.testing.assert_array_equal(legacy[0, 0], [4, 20, 80])
            with (out / "summary.json").open() as stream:
                self.assertEqual(json.load(stream)["sam_vegetation_pixels"], 2)

    def test_overlap_preserves_maximum_labels_and_handles_edges(self):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        tiles, shape = self.module.tile_image(image, tile_size=3, overlap=1)
        self.assertEqual(tiles[-1][0], (2, 3, 4, 5))
        masks = [(coords, np.ones(tile.shape[:2], dtype=np.uint8)) for coords, tile in tiles]
        masks[0][1][0, 2] = 2
        result = self.module.stitch_tiles(masks, shape)
        np.testing.assert_array_equal(result, [[1, 1, 2, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        sam = self.module.stitch_tiles([(c, m > 0) for c, m in masks], shape)
        np.testing.assert_array_equal(sam, np.ones((3, 5), dtype=np.uint8))

    def test_shape_empty_and_tile_geometry_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "plant_mask"):
            self.module.summarize_masks(np.ones((2, 2)), np.zeros((1, 2)), "rice.png")
        with self.assertRaises(ValueError):
            self.module.create_abnormal_overlay(np.empty((0, 2, 3), dtype=np.uint8), np.empty((0, 2)))
        with self.assertRaisesRegex(ValueError, "yellow_mask"):
            self.module.create_abnormal_overlay(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "tile"):
            self.module.stitch_tiles([((0, 2, 0, 2), np.zeros((1, 2)))], (2, 2))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.module.tile_image(np.zeros((3, 3, 3), dtype=np.uint8), 3, 3)

    def test_missing_and_unreadable_input_and_checkpoint_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = str(root / "missing.png")
            with self.assertRaisesRegex(FileNotFoundError, "missing.png"):
                self.module.segveg_on_orthomosaic(missing, "sam.pth", None, root / "out")
            bad = root / "broken.png"
            bad.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "broken.png"):
                self.module.segveg_on_orthomosaic(str(bad), "sam.pth", None, root / "out")
            good = root / "rice.png"
            Image.new("RGB", (2, 2)).save(good)
            with self.assertRaisesRegex(FileNotFoundError, "missing-sam.pth"):
                self.module.segveg_on_orthomosaic(str(good), str(root / "missing-sam.pth"), None, root / "out")

    def test_empty_sam_does_not_require_classifier_features(self):
        mask, probability = self.module.prediction_XG_SVM(
            np.zeros((2, 2, 3), dtype=np.uint8), None, 0.8, 1,
            np.zeros((2, 2), dtype=np.uint8))
        np.testing.assert_array_equal(mask, np.zeros((2, 2)))
        np.testing.assert_array_equal(probability, np.zeros((2, 2)))

    def test_current_hsv_filter_depends_only_on_red_blue_invariant_sv(self):
        rgb = np.array([[[200, 180, 30], [30, 180, 200], [60, 40, 20], [190, 190, 190]]], dtype=np.uint8)
        mask = np.array([[2, 2, 2, 2]], dtype=np.uint8)
        original_call = self.module.hsv_filter_senescent(rgb, mask)
        swapped_call = self.module.hsv_filter_senescent(rgb[..., ::-1].copy(), mask)
        np.testing.assert_array_equal(original_call, [[2, 2, 0, 0]])
        np.testing.assert_array_equal(original_call, swapped_call)

    def test_pipeline_preserves_sam_before_classification_and_hsv_filter(self):
        # Low saturation removes one senescent pixel after classification.
        rgb = np.array([[[80, 220, 20], [180, 220, 20], [80, 80, 80]],
                        [[80, 220, 20], [180, 220, 20], [80, 220, 20]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "rice.png"
            Image.fromarray(rgb).save(image)
            checkpoint = root / "sam.pth"
            checkpoint.touch()
            def synthetic_sam(tile, _checkpoint):
                return np.ones(tile.shape[:2], dtype=np.uint8)
            with patch.object(self.module, "generate_plant_mask", side_effect=synthetic_sam):
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    self.module.segveg_on_orthomosaic(str(image), str(checkpoint), TinyProbabilityModel(),
                                                     root / "out", tile_size=2, overlap=1)
            with (root / "out/summary.json").open() as stream:
                stats = json.load(stream)
            self.assertEqual(stats["sam_vegetation_pixels"], 6)
            self.assertEqual(stats["green_pixels"], 2)
            self.assertEqual(stats["yellow_pixels"], 3)
            self.assertEqual(stats["other_pixels"], 1)
            self.assertAlmostEqual(stats["yellow_ratio_percent"], 60)
            np.testing.assert_array_equal(np.array(Image.open(root / "out/plant_mask.png")),
                                          [[255, 255, 255], [255, 255, 255]])
            np.testing.assert_array_equal(np.array(Image.open(root / "out/yellow_mask.png")),
                                          [[255, 0, 0], [255, 0, 255]])
            self.assertIn("SAM", log.getvalue())


if __name__ == "__main__":
    unittest.main()
