import unittest
from unittest.mock import patch, call

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from MatNexus.VecVisualizer import Word2VecVisualizer


class MockModel:
    def __init__(self, vec):
        self.vec = vec
        self.model = self

    def __getitem__(self, key):
        return self.vec

    def similar_by_vector(self, vec, topn=20):
        return [("word1", 0.9), ("word2", 0.8)] * topn

    def similar_by_word(self, word, topn=20):
        return [("word1", 0.9), ("word2", 0.8)] * topn

    @property
    def wv(self):
        return self


class TestWord2VecVisualizer(unittest.TestCase):
    def setUp(self):
        self.property_list = ["property1", "property2"]
        self.mock_model = MockModel([0.5, 0.5])
        self.visualizer = Word2VecVisualizer(self.mock_model)
        self.coordinates = np.array([[0.5, 0.5], [1.0, 1.0]])
        # Mocking the data attribute with sample data
        mock_data = {
            "label": ["property1", "property2"],
            "level": [0, 1],
            "word": ["word1", "word2"],
        }
        self.visualizer.data = pd.DataFrame(mock_data)

    def test_collect_similar_words(self):
        expected_words = [
            ("word1", 1),
            ("word1", 0),
            ("word2", 0),
            ("word1", 0),
            ("word2", 0),
        ]
        words = self.visualizer.collect_similar_words("word1", level=1,
                                                      top_n_similar=2)
        self.assertEqual(words, expected_words)

    def test_get_property_vectors(self):
        expected_vectors = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
        self.visualizer.get_property_vectors(self.property_list)
        np.testing.assert_array_equal(
            self.visualizer.property_vectors, expected_vectors
        )

    def test_get_words_data(self):
        self.visualizer.get_property_vectors(self.property_list)
        self.visualizer.get_words_data(self.property_list, level=1, top_n_similar=5)

        self.assertIsNotNone(self.visualizer.word_vectors)
        self.assertIsNotNone(self.visualizer.data)

    def test_plot_2d(self):
        fig = self.visualizer.plot_2d(self.coordinates, self.property_list)

        # Check that the function returns the expected type of output
        self.assertIsInstance(fig, go.Figure)

        # Check that the figure layout is adjusted based on input arguments
        self.assertEqual(fig.layout.width, 720)
        self.assertEqual(fig.layout.height, 576)

    def test_plot_data(self):
        with self.assertRaises(ValueError):
            self.visualizer.plot_data(self.property_list, plot_method="invalid_method")

    @patch("plotly.express.scatter")
    @patch.object(
        TSNE, "fit_transform", return_value=np.array([[0.1, 0.2], [0.3, 0.4]])
    )
    @patch(
        "MatNexus.VecGenerator.VectorOperations.generate_material_vector",
        return_value=np.array([0.5, 0.5]),
    )
    def test_plot_material_vectors(
        self, mock_gen_mat_vec, mock_tsne_fit_transform, mock_px_scatter
    ):
        material_list = ["material1", "material2"]

        self.visualizer.plot_material_vectors(material_list)

        calls = [call(material, self.mock_model) for material in material_list]
        mock_gen_mat_vec.assert_has_calls(calls)

        mock_tsne_fit_transform.assert_called_once()

        mock_px_scatter.assert_called_once()

    def test_plot_similarity_scatter(self):
        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
            "Similarity": [0.8, 0.6, 0.9, 0.7, 0.5],
            "Element1": [0.2, 0.5, 0.3, 0.8, 0.9],
            "Element2": [0.7, 0.3, 0.6, 0.4, 0.1],
            "Experimental_Indicator": [2.5, 1.8, 2.1, 1.9, 3.0],
        }
        df = pd.DataFrame(data)

        elements = ["Element1", "Element2", "Similarity", "Experimental_Indicator"]

        fig = self.visualizer.plot_similarity_scatter(
            df, elements, nrows=2, ncols=2, figsize=(10, 10)
        )
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
