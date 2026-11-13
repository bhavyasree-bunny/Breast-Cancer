import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from legacy_multimodal_gat import cosine_edges, fuse, compare, write, train


class GraphTests(unittest.TestCase):
    def test_validation_training_does_not_load_test_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "embeddings").mkdir()
            rows = []
            for i in range(12):
                split = "train" if i < 6 else "validation" if i < 10 else "test"
                rows.append({"case_uuid": str(i), "patient_id": str(i), "label": str(i % 2),
                    "split": split, "age_at_diagnosis": "15000", "race": "a", "ethnicity": "b"})
                path = work / "embeddings" / f"{i}.npz"
                if split == "test":
                    path.write_text("invalid: test features must never be loaded")
                else:
                    np.savez(path, feature=np.array([i % 2, i / 12.]))
            write(work / "patients.csv", rows)
            with redirect_stdout(StringIO()):
                train(SimpleNamespace(work=work, run_dir=work / "run", validation_only=True, neighbours=2, epochs=2))
            report = json.loads((work / "run/metrics.json").read_text())
            self.assertEqual(set(report["evaluation"]), {"validation"})
            self.assertEqual(report["split_counts"]["test"], 0)
            self.assertFalse((work / "gat.pt").exists())

    def test_comparison_preserves_results_and_rejects_mismatched_gat_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "embeddings").mkdir()
            rows = []
            for i in range(20):
                rows.append({"case_uuid": str(i), "patient_id": str(i), "label": str(i % 2),
                    "split": "train" if i < 12 else "validation" if i < 16 else "test",
                    "age_at_diagnosis": "15000", "race": "a", "ethnicity": "b"})
                np.savez(work / "embeddings" / f"{i}.npz", feature=np.array([i % 2, i / 20.]))
            write(work / "patients.csv", rows)
            write(work / "predictions.csv", [{"case_uuid": "different", "split": "test", "label": "0", "lobular_probability": ".2"}])
            original = (work / "predictions.csv").read_bytes()
            with redirect_stdout(StringIO()):
                compare(SimpleNamespace(work=work))
            report = json.loads((work / "comparison/metrics.json").read_text())
            self.assertEqual(set(report["models"]), {"image_only", "clinical_only", "fused"})
            self.assertIn("gat_note", report)
            self.assertEqual(original, (work / "predictions.csv").read_bytes())

    def test_cosine_neighbours_exclude_self(self):
        vectors = np.array([[1., 0], [.9, .1], [0, 1.]])
        edges, scores = cosine_edges(vectors, k=1)
        self.assertEqual(edges.shape, (2, 3))
        self.assertTrue(np.all(edges[0] != edges[1]))
        self.assertEqual(edges[:, 0].tolist(), [1, 0])
        self.assertTrue(np.isfinite(scores).all())

    def test_queries_only_receive_training_edges(self):
        edges, scores = cosine_edges(np.eye(3), np.array([[1., 0, 0], [0, 1, 0]]), k=2)
        self.assertTrue(np.all(edges[0] < 3))
        self.assertTrue(np.all(edges[1] >= 3))
        self.assertEqual(set(edges[1]), {3, 4})

    def test_zero_vectors_and_large_k(self):
        edges, scores = cosine_edges(np.zeros((2, 3)), k=10)
        self.assertEqual(edges.shape, (2, 2))
        self.assertTrue(np.isfinite(scores).all())

    def test_preprocessing_fits_only_training(self):
        images = np.array([[1., 2], [3., 4], [1000., 2000.]])
        clinical = np.array([[30., "a", "x"], [50., "b", "x"], [90., "unseen", "y"]], dtype=object)
        features, transforms = fuse(images[:2], images, clinical[:2], clinical)
        np.testing.assert_allclose(transforms["image_scaler"][0].mean_, [2., 3.])
        self.assertTrue(np.isfinite(features).all())
        categories = transforms["clinical_encoder"].named_transformers_["categorical"].categories_
        self.assertNotIn("unseen", categories[0])

    def test_two_layer_gat_queries_do_not_change_training_outputs(self):
        try:
            import torch
            from torch_geometric.nn import GATConv
        except ImportError:
            self.skipTest("Run in .venv-model for the GAT isolation test")
        torch.manual_seed(42)
        reference = np.eye(3, dtype=np.float32)
        query = np.array([[2., 3, 4]], dtype=np.float32)
        edge, values = cosine_edges(reference, k=2)
        extra, extra_values = cosine_edges(reference, query, k=2)
        layers = [GATConv(3, 3, edge_dim=1, fill_value=1.), GATConv(3, 2, edge_dim=1, fill_value=1.)]
        for layer in layers:
            layer.eval()
        def run(x, e, v):
            with torch.inference_mode():
                for layer in layers:
                    x = layer(x, e, v)
            return x
        original = run(torch.tensor(reference), torch.tensor(edge), torch.tensor(values))
        extended = run(torch.tensor(np.concatenate([reference, query])),
                       torch.tensor(np.concatenate([edge, extra], axis=1)),
                       torch.tensor(np.concatenate([values, extra_values])))
        torch.testing.assert_close(original, extended[:3])


if __name__ == "__main__":
    unittest.main()
