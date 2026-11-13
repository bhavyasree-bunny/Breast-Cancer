import unittest
import numpy as np
import torch
from patient_graph_model import PatientGraphGAT, filtered_edges, make_graph, evaluate


class PatientGraphModelTests(unittest.TestCase):
    def test_similarity_filter_does_not_force_weak_neighbours(self):
        vectors = np.eye(3, dtype=np.float32)
        edge, scores = filtered_edges(vectors, k=2, minimum=.25)
        self.assertEqual(edge.shape, (2, 0))
        model = PatientGraphGAT(3).eval()
        with torch.inference_mode():
            logits = model(*make_graph(vectors, None, 2, .25))
        self.assertTrue(torch.isfinite(logits).all())

    def test_held_out_nodes_cannot_change_training_or_other_queries(self):
        torch.manual_seed(42)
        reference = np.eye(3, dtype=np.float32)
        query = np.array([[1., 2, 3]], dtype=np.float32)
        model = PatientGraphGAT(3).eval()
        with torch.inference_mode():
            original = model(*make_graph(reference, None, 2, 0.))
            extended = model(*make_graph(reference, query, 2, 0.))
            batched = model(*make_graph(reference, np.concatenate([query, query*20]), 2, 0.))
        torch.testing.assert_close(original, extended[:3])
        torch.testing.assert_close(extended[3], batched[3])

    def test_accuracy_is_distinct_from_balanced_accuracy(self):
        report = evaluate(np.array([0, 0, 0, 1]), np.zeros(4))
        self.assertEqual(report['accuracy'], .75)
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertEqual(report['majority_class_accuracy'], .75)
        low, high = report['accuracy_95_wilson_interval']
        self.assertLess(low, .75)
        self.assertGreater(high, .75)


if __name__ == '__main__':
    unittest.main()
