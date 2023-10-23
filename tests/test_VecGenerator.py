import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from MatNexus.VecGenerator import (
    Corpus,
    Word2VecModel,
    VectorOperations,
    MaterialSimilarityCalculator,
    MaterialListGenerator,
)


class TestCorpus(unittest.TestCase):
    def setUp(self):
        data = {"abstract": ["Hydrogen is very reactive."]}
        self.mock_df = pd.DataFrame(data)
        self.corpus_instance = Corpus(self.mock_df)

    def test_tokenize_abstracts(self):
        tokenized_abstracts = self.corpus_instance.tokenize_abstracts()
        expected_output = [["Hydrogen", "is", "very", "reactive", "."]]
        self.assertEqual(tokenized_abstracts, expected_output)


class TestWord2VecModel(unittest.TestCase):
    def setUp(self):
        self.sentences = [["Hydrogen", "is", "very", "reactive", "."]]
        self.w2v_model = Word2VecModel(self.sentences)
        self.w2v_model.fit()

    def test_fit(self):
        self.assertIsNotNone(self.w2v_model.model)

    def test_getitem(self):
        self.assertIsNotNone(self.w2v_model["Hydrogen"])

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.w2v_model.save(tmp.name)

        loaded_model = Word2VecModel.load(tmp.name)
        self.assertIsNotNone(loaded_model.model)

        os.unlink(tmp.name)

    def test_most_similar(self):
        similar_words = self.w2v_model.most_similar("Hydrogen", topn=2)
        self.assertIsInstance(similar_words, list)
        self.assertEqual(len(similar_words), 2)
        for word, similarity in similar_words:
            self.assertIsInstance(word, str)
            self.assertIsInstance(similarity, float)


class MockModel(Word2VecModel):
    def __init__(self, vec):
        self.vec = vec

    def __getitem__(self, key):
        return self.vec


class TestVectorOperations(unittest.TestCase):
    def setUp(self):
        self.mock_model = MockModel(np.array([0.1, 0.2, 0.3]))

    def test_split_chemical_formula(self):
        result = VectorOperations.split_chemical_formula("H2O")
        self.assertEqual(result, [("H", 2), ("O", 1)])

    def test_generate_material_vector(self):
        result = VectorOperations.generate_material_vector("H2O", self.mock_model)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3,))

    def test_get_vector(self):
        sample_input = "sample_material"
        vector = VectorOperations.get_vector(sample_input, self.mock_model)
        self.assertIsNotNone(vector)
        self.assertEqual(vector.shape, (3,))

    def test_generate_property_vectors(self):
        property_list = ["active", "robust", "electrocatalyst"]
        result = VectorOperations.generate_property_vectors(
            property_list, self.mock_model
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(property_list))


class TestMaterialListGenerator(unittest.TestCase):
    def test_generate_material_list(self):
        elements_ranges = [("A", (50, 52)), ("B", (48, 50)), ("C", (1, 3))]
        generator = MaterialListGenerator(elements_ranges)
        material_df = generator.generate_material_list()
        material_list = material_df["Material"].tolist()

        expected_material_list = [
            "A50B48C2",
            "A50B49C1",
            "A51B48C1",
        ]

        self.assertEqual(material_list, expected_material_list)


class TestMaterialSimilarityCalculator(unittest.TestCase):
    def setUp(self):
        self.mock_model = MockModel(np.array([0.1, 0.2, 0.3]))
        self.property_list = ["property1", "property2", "property3"]
        self.calculator = MaterialSimilarityCalculator(
            self.mock_model, self.property_list
        )

    def test_calculate_similarity_vectors(self):
        material_list = ["H2O", "H2"]
        similarity_vectors = self.calculator._calculate_similarity_vectors(
            material_list
        )

        self.assertIsInstance(similarity_vectors, dict)
        self.assertEqual(set(similarity_vectors.keys()), set(material_list))

    def test_find_top_similar_materials(self):
        material_list = ["H2O", "H2"]
        target_material = "H2"
        top_materials = self.calculator.find_top_similar_materials(
            target_material, material_list, top_n=1
        )
        self.assertEqual(len(top_materials), 1)
        self.assertEqual(top_materials[0][0], "H2")

    def test_calculate_similarity_from_dataframe(self):
        data = {
            "A": [10, 20],
            "B": [20, 30],
            "C": [70, 50],
            "Resistance": [1.5, 2.0],  # Added 'Resistance' column for example
        }
        df = pd.DataFrame(data)
        target_material = "H2O"
        element_columns = ["A", "B", "C"]
        result = self.calculator.calculate_similarity_from_dataframe(
            df,
            element_columns,
            target_material,
            top_n=None,
            percentages_as_decimals=False,
        )
        expected_material_names = ["A10B20C70", "A20B30C50"]
        expected_similarity = [1.0, 1.0]
        expected_exp_indicator = [1 / 1.5, 1 / 2.0]
        self.assertEqual(result["Material_Name"].tolist(), expected_material_names)
        for actual, expected in zip(result["Similarity"].tolist(), expected_similarity):
            self.assertTrue(math.isclose(actual, expected))
        self.assertEqual(
            result["Experimental_Indicator"].tolist(), expected_exp_indicator
        )


if __name__ == "__main__":
    unittest.main()
