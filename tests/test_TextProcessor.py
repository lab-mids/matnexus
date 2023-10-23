import unittest

import pandas as pd

from MatNexus.TextProcessor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        import nltk

        if not nltk.data.find("tokenizers/punkt"):
            raise ValueError(
                "The NLTK 'punkt' dataset is missing. "
                "Please download it using: nltk.download('punkt')"
            )

        if not nltk.corpus.stopwords.words("english"):
            raise ValueError(
                "The NLTK 'stopwords' dataset is missing. "
                "Please download it using: nltk.download('stopwords')"
            )

        self.df = pd.DataFrame(
            {
                "abstract": [
                    "This is a test sentence ©. This is another test sentence"
                    " & Co. This sentence has rights reserved.",
                    "This is an abstract to test lemmatization. "
                    "It contains words like running, easily, and geese.",
                    "Here is a string with numbers 123 and a chemical "
                    "formula H2O. Some stop words too.",
                ]
            }
        )
        self.references = [
            "",
            "this be an abstract to test lemmatization . "
            "it contain word like run , easily , "
            "and geese .",
            "string numbers chemical formula h2o h o " "stop words",
        ]

        self.references_process = [
            "",
            "abstract test lemmatization contain word like run easily geese",
            "string number chemical formula h2o h o stop word",
        ]

        self.processor = TextProcessor(self.df)

    def test_filter_sentences(self):
        result = self.processor.filter_sentences(self.df.loc[0, "abstract"])
        self.assertEqual(result, self.references[0])

    def test_lemmatize_corpus(self):
        result = self.processor.lemmatize_corpus(self.df.loc[1, "abstract"])
        self.assertEqual(result, self.references[1])

    def test_filter_words(self):
        result = self.processor.filter_words(self.df.loc[2, "abstract"])
        self.assertEqual(result, self.references[2])

    def test_all_filters(self):
        results = self.processor.processed_df
        for result, reference in zip(results["abstract"], self.references_process):
            self.assertEqual(result, reference)


if __name__ == "__main__":
    unittest.main()
