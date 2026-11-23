"""Capacity of the native tensors before the pipeline materializes them."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from native_preparation import dense_tensor_capacity


class CapacityTests(unittest.TestCase):
    def test_ocrapt_uses_behavior_columns_when_embedding_dimension_is_unset(self):
        options = SimpleNamespace(use_lifespan=False, use_cumulative_active_time=False)
        features = SimpleNamespace(used_method='ocrapt_features', emb_dim=None, ocrapt_features=options)
        for lifespan, cumulative in ((False, False), (True, False), (True, True)):
            with self.subTest(lifespan=lifespan, cumulative=cumulative):
                options.use_lifespan = lifespan
                options.use_cumulative_active_time = cumulative
                # Deliberately different dimensions for the histogram and the edge one-hot.
                result = dense_tensor_capacity(features, 7, node_types=3, edge_types=10, ocr_histogram_bins=8)
                expected_width = 19 + int(lifespan) + int(cumulative)
                self.assertEqual(result['feature_dimension'], expected_width)
                message_width = 3 + expected_width + 10 + 3 + expected_width
                self.assertEqual(result['native_dense_feature_tensor_bytes'], 7 * (4 * message_width + 32))

    def test_other_layouts(self):
        for method, dimension, expected_width in [('word2vec', 128, 272), ('hierarchical_hashing', 16, 48),
                                                  ('only_type', None, 16), ('unsupported', 128, None)]:
            with self.subTest(method=method):
                result = dense_tensor_capacity(SimpleNamespace(used_method=method, emb_dim=dimension), 2, 3, 10)
                self.assertEqual(result.get('native_message_width'), expected_width)
                expected_bytes = None if expected_width is None else 2 * (4 * expected_width + 32)
                self.assertEqual(result['native_dense_feature_tensor_bytes'], expected_bytes)


if __name__ == '__main__':
    unittest.main()
