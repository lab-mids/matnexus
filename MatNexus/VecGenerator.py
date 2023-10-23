import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import word2vec
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


class Corpus:
    """
    Preprocess text from abstracts.
    """

    def __init__(self, df):
        """
        Initialize the TextPreprocessor with a DataFrame containing abstracts.

        Parameters:
            df: The DataFrame with abstracts to preprocess.
        """
        self.abstracts = df["abstract"].tolist()
        self.sentences = self.preprocess()

    def preprocess(self):
        """
        Apply preprocessing on the abstracts.

        Returns:
            List of tokenized sentences.
        """
        return self.tokenize_abstracts()

    def tokenize_abstracts(self):
        """
        Tokenize the abstracts.

        Returns:
            List of tokenized sentences.
        """
        abstracts = [str(abstract) for abstract in self.abstracts]
        sentences = [
            nltk.sent_tokenize(abstract, language="english") for abstract in abstracts
        ]
        all_sentences = []
        for sentence in sentences:
            all_sentences += [word_tokenize(s) for s in sentence]
        return all_sentences


class Model:
    """
    Abstract base class for models.
    """

    def fit(self):
        """
        Fit the model.

        Note:
            This method should be overridden by a subclass.
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Save the model to a file.

        Parameters:
            filename: The path of the file to save the model.

        Note:
            This method should be overridden by a subclass.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        """
        Load a model from a file.

        Parameters:
            filename: The path of the file to load the model from.

        Returns:
            The loaded model.

        Note:
            This method should be overridden by a subclass.
        """
        raise NotImplementedError

    def __getitem__(self, key):
        """
        Get an item from the model.

        Parameters:
            key: The key of the item to get.

        Note:
            This method should be overridden by a subclass.
        """
        raise NotImplementedError


class Word2VecModel(Model):
    """
    Class representing a Word2Vec model.
    """

    def __init__(self, sentences):
        """
        Initialize the Word2VecModel with sentences.

        Parameters:
            sentences: The sentences to train the model on.
        """
        self.sentences = sentences
        self.model = None

    def fit(self):
        """
        Fit the Word2Vec model to the sentences.
        """
        self.model = word2vec.Word2Vec(
            self.sentences,
            sg=1,
            vector_size=200,
            hs=1,
            window=5,
            min_count=1,
            workers=4,
        )

    def save(self, filename):
        """
        Save the Word2Vec model to a file.

        Parameters:
            filename: The path of the file to save the model.
        """
        self.model.save(filename)

    @classmethod
    def load(cls, filename):
        """
        Load a Word2Vec model from a file.

        Parameters:
            filename: The path of the file to load the model from.

        Returns:
            The loaded Word2Vec model.
        """
        instance = cls.__new__(cls)
        instance.model = gensim.models.Word2Vec.load(filename)
        return instance

    def __getitem__(self, key):
        """
        Get a word vector from the Word2Vec model.

        Parameters:
            key: The word to get the vector of.

        Returns:
            The word vector.
        """
        return self.model.wv.__getitem__(key)

    def most_similar(self, word, topn=10):
        """
        Calculate the similarity between a word and the topn most similar
        words.

        Parameters:
            word (str): The target word.
            topn (int): The number of similar words to retrieve (default is 10).

        Returns:
            list: A list of tuples containing the most similar words and their
                similarity scores.
        """

        return self.model.wv.most_similar(word, topn=topn)


class VectorOperations:
    """
    A utility class containing static methods for various operations related
    to vectors, specifically focused on processing chemical formulas and
    generating corresponding vectors.

    The class provides methods to split chemical formulas, generate vectors
    for materials based on their formulas, and obtain vectors for words or phrases.
    """

    @staticmethod
    def split_chemical_formula(word: str) -> list:
        """
        Splits a chemical formula into its constituent elements and their counts.

        For example, the formula 'H2O' would be split into [('H', 2), ('O', 1)].

        Parameters:
            word (str): The chemical formula to split.

        Returns:
            list: A list of tuples where each tuple contains an element
                  from the formula and its count.
        """
        elements_counts = []
        i = 0
        while i < len(word):
            char = word[i]
            if char.isupper():
                j = i + 1
                while j < len(word) and word[j].islower():
                    j += 1
                element = word[i:j]
                i = j
                count = ""
                while i < len(word) and (
                    word[i].isdigit() or word[i] == "." or word[i] == "-"
                ):
                    count += word[i]
                    i += 1
                count = float(count) if count else 1
                elements_counts.append((element, count))
        return elements_counts

    @staticmethod
    def generate_material_vector(formula: str, model) -> np.ndarray:
        """
        Generate a vector representation for a material based on its chemical formula.

        The resultant vector for a formula like 'H2O' would be composition-weighted,
        i.e., weighted twice for 'H' and once for 'O'.

        The mathematical representation is:
            V = Σ(C_i * v_i) / N
        where:
            - C_i: the count of the i-th element in the formula.
            - v_i: the vector representation of the i-th element
                   from the word2vec model.
            - N: the total number of elements in the formula.

        Parameters:
            formula (str): The chemical formula of the material.
            model (gensim.models.Word2Vec): The word2vec model to use for vector
            generation.

        Returns:
            numpy.ndarray: The generated vector for the given formula.
        """
        elements_counts = VectorOperations.split_chemical_formula(formula)
        multiplied_vectors = []
        for element, count in elements_counts:
            element_vec = model.__getitem__(element.lower())
            percentage = count / 100
            multiplied_vector = element_vec * percentage
            multiplied_vectors.append(multiplied_vector)
        material_vec = np.mean(multiplied_vectors, axis=0)
        return material_vec

    @staticmethod
    def get_vector(word_or_phrase: str, model) -> np.ndarray:
        """
        Obtain the vector representation for a word or a phrase.

        If a phrase is provided, the function computes the mean vector representation
        of all the words in the phrase.

        Parameters:
            word_or_phrase (str): The word or phrase for which the vector is needed.
            model: The word2vec model to retrieve the vector from.

        Returns:
            numpy.ndarray: The vector representation of the given word or phrase.
        """
        return np.mean([model.__getitem__(w) for w in word_or_phrase.split()], axis=0)

    @staticmethod
    def generate_property_vectors(property_list: list, model) -> list:
        """
        Generate vectors for a list of properties.

        The function computes vectors for each property in the list using the
        provided word2vec model.

        Parameters:
            property_list (list): A list of properties for which vectors are required.
            model: The word2vec model to use for vector generation.

        Returns:
            list: A list of vectors corresponding to the provided properties.
        """
        return [VectorOperations.get_vector(p, model) for p in property_list]


class MaterialListGenerator:
    """
    Generates combinations of chemical materials based on given element ranges.

    This class creates a list or a DataFrame of chemical combinations based
    on provided elements and their respective atomic percentages.
    """

    def __init__(self, elements_ranges):
        """
        Initializes a MaterialListGenerator instance.

        Parameters:
            elements_ranges (list): A list of tuples where each tuple contains
                                    an element name and its atomic percentage range.
        """
        self.elements_ranges = elements_ranges

    def generate_material_list(self, step=1):
        """
        Generates a list of materials and returns them in a dataframe.

        The method produces combinations of materials based on the given
        element percentage ranges. For a single element, it creates a list
        of that element with different atomic percentages.

        Parameters:
            step (int): The increment between each percentage in the range.
                        Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame of generated materials with element
                          names as columns and respective atomic percentages
                          as values.
        """
        material_dict = {}
        if len(self.elements_ranges) == 1:
            element, percentage_range = self.elements_ranges[0]
            for percentage in range(percentage_range[0], percentage_range[1], step):
                material = f"{element}{percentage}"
                material_dict[material] = dict(
                    VectorOperations.split_chemical_formula(material)
                )
        else:
            self._generate_material_combinations(
                self.elements_ranges, [], "", material_dict, step
            )

        material_df = pd.DataFrame.from_dict(
            material_dict, orient="index"
        ).reset_index()
        material_df.columns = ["Material"] + list(material_df.columns[1:])
        return material_df

    def _generate_material_combinations(
        self,
        elements_ranges,
        current_combination,
        current_material,
        material_dict,
        step,
    ):
        """
        Recursive method to generate all combinations of materials.

        This method creates combinations of materials by recursively traversing
        through the provided element ranges and appending them to the current
        material until all combinations are generated.

        Parameters:
            elements_ranges (list): A list of tuples containing element name
                                    and its atomic percentage range.
            current_combination (list): Current combination of elements.
            current_material (str): Current material representation.
            material_dict (dict): Dictionary to store generated materials
                                  with element names and counts.
            step (int): The increment between each percentage in the range.
        """
        if elements_ranges:
            element, percentage_range = elements_ranges[0]
            for percentage in range(percentage_range[0], percentage_range[1] + 1):
                # Round percentage to nearest multiple of step
                rounded_percentage = round(percentage / step) * step
                new_combination = current_combination + [(element, rounded_percentage)]
                new_material = current_material + f"{element}{rounded_percentage}"
                self._generate_material_combinations(
                    elements_ranges[1:],
                    new_combination,
                    new_material,
                    material_dict,
                    step,
                )
        else:
            if sum(percentage for element, percentage in current_combination) == 100:
                material_dict[current_material] = dict(
                    VectorOperations.split_chemical_formula(current_material)
                )


class MaterialSimilarityCalculator:
    """
    Computes similarity scores between materials based on their vector representations.

    This class utilizes a Word2Vec model to represent materials as vectors and
    then calculates their similarity to target materials or properties.
    """

    def __init__(self, model, property_list=None):
        """
        Initialize the MaterialSimilarityCalculator with a model and property vectors.

        Parameters:
            model (gensim.models.Word2Vec): The word2vec model for generating vector
                                            representations.

            property_list (list, optional): A list of properties for which vectors
                                            are pre-generated. Defaults to None.
        """
        self.model = model
        if property_list is not None:
            self.property_vectors = VectorOperations.generate_property_vectors(
                property_list, model
            )
        else:
            self.property_vectors = None

    def _calculate_similarity_vectors(self, material_list):
        """
        Calculate similarity vectors for a list of materials.

        Parameters:
            material_list (list): A list of materials for which similarity vectors
                                  are to be calculated.

        Returns:
            dict: A dictionary mapping each material in the list to its similarity
                  vector.
        """
        material_vectors = [
            VectorOperations.generate_material_vector(material, self.model)
            for material in material_list
        ]
        similarity_vectors = [
            cosine_similarity([material_vector], self.property_vectors)[0]
            for material_vector in material_vectors
        ]
        return dict(zip(material_list, similarity_vectors))

    def find_top_similar_materials(self, target_material, material_list, top_n=10):
        """Find the top-n materials most similar to the target material.

        This method calculates the cosine similarity between the
        target material and materials from the provided list and then
        returns the top-n most similar ones.

        Parameters:
            target_material (str): The name of the target material.
            material_list (list): List of materials to compare against the target.
            top_n (int, optional): Number of top materials to return. Default is 10.

        Returns:
            list: List of tuples containing the top-n similar materials and their
                  respective similarity scores.

        """
        material_similarity_vectors = self._calculate_similarity_vectors(material_list)
        material_list = list(material_similarity_vectors.keys())
        target_vec = self.model[target_material]
        target_sim_to_properties = cosine_similarity(
            [target_vec], self.property_vectors
        )[0]

        new_target_vec = target_sim_to_properties
        new_material_vectors = np.array(list(material_similarity_vectors.values()))

        similarities_to_new_target = cosine_similarity(
            new_material_vectors, [new_target_vec]
        ).flatten()
        top_material_indices = similarities_to_new_target.argsort()[-top_n:][::-1]

        top_materials_with_similarity = [
            (material_list[i], similarities_to_new_target[i])
            for i in top_material_indices
        ]

        print("Top", top_n, "similar materials:")
        for material, similarity in top_materials_with_similarity:
            print(f"{material}: {similarity:.4f}")

        return top_materials_with_similarity

    def calculate_similarity_from_dataframe(
        self,
        df,
        element_columns,
        target_material,
        top_n=None,
        percentages_as_decimals=False,
        experimental_indicator_column="Resistance",
        experimental_indicator_func=lambda x: 1 / x,
    ):

        """Calculate similarity scores for materials in a DataFrame
        compared to a target material.

        This method computes the cosine similarity between each
        material in the DataFrame and the target material. The
        resulting similarity scores are added to the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing materials and their properties.

            element_columns (list): List of column names in the DataFrame representing
                                    the elements.

            target_material (str): The name of the target material.

            top_n (int, optional): Number of top similar materials to return.
                                   If None, returns the entire DataFrame.

            percentages_as_decimals (bool, optional): Whether the percentages in the
                                                      DataFrame are given as decimals.
                                                      Default is False.

            experimental_indicator_column (str, optional): Name of the column used to
                                                           compute the
                                                           'Experimental_Indicator'.
                                                           Default is 'Resistance'.

            experimental_indicator_func (function, optional): Function to compute the
                                                              'Experimental_Indicator'
                                                              value. Default is inverse
                                                              function.

        Returns:
            pd.DataFrame or list: If top_n is None, returns DataFrame with similarity
                                  scores. Otherwise, returns list of top-n similar
                                  materials with their scores.

        """

        target_vec = self.model[target_material]
        target_sim_to_properties = cosine_similarity(
            [target_vec], self.property_vectors
        )[0]

        # Dataframe-specific operation to be applied to rows
        def calculate_similarity(row):
            element_percentages = row[element_columns]
            if not percentages_as_decimals:
                element_percentages = [p / 100 for p in element_percentages]
            multiplied_vectors = [
                self.model[element.lower()] * percentage
                for element, percentage in zip(element_columns, element_percentages)
            ]

            material_vec = np.mean(multiplied_vectors, axis=0)
            material_vec_to_properties = cosine_similarity(
                [material_vec], self.property_vectors
            )[0]

            similarity = cosine_similarity(
                [material_vec_to_properties], [target_sim_to_properties]
            )[0]
            return similarity[0]

        df["Similarity"] = df.apply(calculate_similarity, axis=1)
        experimental_indicator = df[experimental_indicator_column].values
        experimental_indicator = experimental_indicator_func(experimental_indicator)
        df["Experimental_Indicator"] = experimental_indicator
        df["Material_Name"] = df[element_columns].apply(
            lambda row: "".join(
                [
                    f"{element}{percentage}"
                    for element, percentage in zip(element_columns, row)
                ]
            ),
            axis=1,
        )

        if top_n is not None:
            top_materials = df.nlargest(top_n, "Similarity")[
                ["Material_Name", "Similarity"]
            ]
            return list(top_materials.itertuples(index=False, name=None))

        return df

    def calculate_similarity_to_list(
        self, material_list, target_words=None, target_materials=None
    ):
        """Compute similarity scores between a list of materials and
        target words/materials.

        This method calculates the cosine similarity between each
        material in the list and a given set of target words or
        materials. The method then returns the resulting similarity
        scores.

        Parameters:
            material_list (list): List of materials to compute similarity scores for.

            target_words (list, optional): List of target words or phrases to compare
                                           against.

            target_materials (list, optional): List of target materials to compare
                                               against.

        Returns:
            list: List of similarity scores corresponding to each material in the
                  `material_list`.

        """
        target_vectors = []
        if target_words:
            word_vectors = [
                VectorOperations.get_vector(word, self.model) for word in target_words
            ]
            target_vectors.extend(word_vectors)
        if target_materials:
            material_vectors = [
                VectorOperations.generate_material_vector(material, self.model)
                for material in target_materials
            ]
            target_vectors.extend(material_vectors)

        average_target_vector = np.mean(target_vectors, axis=0)

        if self.property_vectors is not None:
            average_target_vector = cosine_similarity(
                [average_target_vector], self.property_vectors
            )[0]

        material_vectors = [
            VectorOperations.generate_material_vector(material, self.model)
            for material in material_list
        ]

        if self.property_vectors is not None:
            material_vectors = [
                cosine_similarity([material_vector], self.property_vectors)[0]
                for material_vector in material_vectors
            ]

        similarity_scores = [
            cosine_similarity([material_vector], [average_target_vector])[0][0]
            for material_vector in material_vectors
        ]

        return similarity_scores
