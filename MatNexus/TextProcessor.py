import re
from string import punctuation

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords


class TextProcessor:
    """
    A class to preprocess textual data contained in a DataFrame.
    This class provides functionalities for filtering sentences, lemmatizing,
    and other preprocessing steps.

    Attributes:
        df (pd.DataFrame): Input DataFrame containing text data.
        nlp (spacy.lang): Spacy language model for lemmatization.
        processed_df (pd.DataFrame): DataFrame containing processed text data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize a TextPreprocessor object.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing text data.
        """
        self.df = df
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.processed_df = self.process_dataframe()

    def filter_sentences(self, s: str) -> str:
        """
        Filter sentences in the input string based on certain criteria.

        This method:
            1. Finds sentences ending with © or & Co. and adds a blank space after them.
            2. Removes sentences containing specific phrases.

        Parameters:
            s (str): The input string to filter.

        Returns:
            str: The filtered string.
        """
        s = re.sub(r"([©|& Co\.])(\.)", r"\1\2 ", s)

        sentences = nltk.sent_tokenize(s)
        indices_to_remove = []

        for i, sentence in enumerate(sentences):
            if "©" in sentence or "rights reserved" in sentence or "& Co." in sentence:
                indices_to_remove.append(i)

        sentences_filtered = [
            sentences[i] for i in range(len(sentences)) if (i not in indices_to_remove)
        ]

        return " ".join(sentences_filtered)

    def lemmatize_corpus(self, corpus: str) -> str:
        """
        Lemmatize the input text corpus using spaCy.

        Lemmatization is the process of converting a word to its base or root form.

        Parameters:
            corpus (str): The input text corpus to lemmatize.

        Returns:
            str: The lemmatized corpus.
        """
        doc = self.nlp(corpus)
        lemmatized_corpus = " ".join([token.lemma_ for token in doc])
        return lemmatized_corpus

    def filter_words(self, s: str) -> str:
        """
        Filter words in the input string based on certain criteria.

        This method:
            1. Tokenizes the input string into words.
            2. Removes punctuation, stop words, pure numbers, and other specific patterns.

        Parameters:
            s (str): The input string to filter.

        Returns:
            str: The filtered string.
        """

        # Tokenize the sentences into words
        words = nltk.word_tokenize(s)

        # Remove punctuation from the string except for hyphens
        pattern = r"(?<!\d)[^\w\s.-]+(?!\d)"
        line = re.sub(pattern, "", " ".join(words))

        # Split the string on hyphens and whitespace
        tokens = re.split(r"[-\s]+", line)

        # Regex pattern for chemical formulas
        chemical_formula_pattern = r"([A-Z][a-z]*)(\d*\.\d+|\d*)"

        # Remove stop words and words that are pure numbers, empty,
        # or contain digits not part of a formula
        stop = set(stopwords.words("english"))
        filter_text = []
        for word in tokens:
            if word in punctuation:
                continue
            if not word:
                continue
            if word.lower() in stop or word.isdigit():
                continue

            if re.match(chemical_formula_pattern, word) and any(char.isdigit() for char in word):
                elements = re.findall(chemical_formula_pattern, word)
                filter_text.append(word.lower())
                for element, _ in elements:
                    filter_text.append(element.lower())
            else:
                filter_text.append(word.lower())

        # Join the filtered words into a single string
        return " ".join(filter_text)

    def process_dataframe(self) -> pd.DataFrame:
        """
        Process the input DataFrame with multiple preprocessing steps.

        This method applies a series of preprocessing steps to the input
        DataFrame, including sentence filtering, word filtering, and lemmatization.
        The steps are applied in the order mentioned.

        Returns:
            pd.DataFrame: The processed DataFrame containing cleaned text data.
        """
        processed_df = self.df.copy()
        processed_df["abstract"] = processed_df["abstract"].fillna("")

        for step, step_name in zip(
            [self.filter_sentences, self.filter_words, self.lemmatize_corpus],
            ["Filtering sentences", "Filtering words", "Lemmatizing"],
        ):
            if step == self.lemmatize_corpus:
                abstracts = processed_df["abstract"].tolist()
                lemmatized_abstracts = list(self.nlp.pipe(abstracts, batch_size=500))
                lemmatized_abstracts = [
                    " ".join([token.lemma_ for token in doc])
                    for doc in lemmatized_abstracts
                ]
                for i, lemmatized_abstract in enumerate(lemmatized_abstracts):
                    processed_df.loc[i, "abstract"] = lemmatized_abstract
            else:
                for i, abstract in enumerate(processed_df["abstract"]):
                    processed_df.loc[i, "abstract"] = step(abstract)

        return processed_df
