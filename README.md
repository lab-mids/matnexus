# MatNexus

MatNexus is a Python project designed to collect, process, and analyze scientific 
papers, especially in material science. It is divided into four main modules:

1. `PaperCollector`: Collects papers from the Scopus database based on a given query.
2. `TextProcessor`: Processes the text content of the papers for further analysis.
3. `VecGenerator`: Generates vector representations of the processed text for machine learning applications.
4. `VecVisualizer`: A module for visualizing word embeddings using a Word2Vec model.

## Installation

This project uses a number of external dependencies, which are listed in the `environment.yml` file.

To create a new conda environment named 'matnexus_env' with all the necessary dependencies, run the following command:

```
conda env create -f environment.yml
```

To install this package, clone the repository and run one of the following commands:

For a system-wide installation:

```
python setup.py install
```

For a developer installation (i.e., changes to the source code immediately take effect):

```
python setup.py develop
```

Alternatively, you can install the package with pip:

```
pip install .
```
For a developer installation with pip:

```
pip install -e .
```

## PaperCollector

`PaperCollector` is a module for collecting and processing papers from Scopus.

**Key Classes**

- `ScopusPaperCollector`: The `ScopusPaperCollector` class is used for collecting 
  and processing papers from Scopus.

Example usage:

```python
from MatNexus import PaperCollector
query = PaperCollector.ScopusPaperCollector.build_query(keywords='electrocatalyst', 
                                                        startyear=2020, endyear=2023, openaccess=True)
collector = PaperCollector.ScopusPaperCollector(query, limit=100, config_path="path/to/config.ini")
collector.collect_papers()
sorted_papers = collector.sort_by("citedby_count", ascending=False)
collector.plot_publications_per_year(width=0.3)
```

## TextProcessor

`TextProcessor` is a module for preprocessing text data obtained from papers. It 
includes the `TextProcessor` class which provides methods for filtering sentences, 
lemmatizing corpus, filtering words, and processing a DataFrame of text data.

**Key Classes**
- `TextProcessor`: Represents the main class responsible for text preprocessing. 
  It takes an input DataFrame containing text data and performs various 
  preprocessing steps on it.

**Example usage**:

```python
from MatNexus import TextProcessor
import pandas as pd

df = pd.DataFrame({"abstract": ["This is a test abstract."]})
processor = TextProcessor(df)
processed_df = processor.processed_df
```

## VecGenerator

`VecGenerator` is a module focused on processing abstracts and preparing them for vector generation.

**Key Classes**
- `Corpus`: Represents a class responsible for preprocessing text from abstracts. 
  It tokenizes abstracts and prepares them for subsequent vector representation.

**Example usage**:
```python
from MatNexus import VecGenerator
import pandas as pd

df = pd.DataFrame({"abstract": ["This is a test abstract.", "Another abstract for testing."]})
corpus = VecGenerator.Corpus(df)
sentences = corpus.sentences
```

- `Word2VecModel`: Represents a class responsible for generating a Word2Vec model. 
  It takes a list of sentences as input and generates a Word2Vec model from them.

**Example usage**:

```python
from MatNexus import VecGenerator

sentences = [["This", "is", "a", "sentence."], ["Another", "sentence", "for", "testing."]]
model = VecGenerator.Word2VecModel(sentences)
model.fit()
model.save("model")
loaded_model = VecGenerator.Word2VecModel.load("model")
similar_words = loaded_model.most_similar("word")
```

- `MaterialListGenerator`: A class for generating a list of materials.

**Example usage**:

```python
from MatNexus import VecGenerator

elements_ranges = [("H", [0, 100]), ("O", [0, 100])]
generator = VecGenerator.MaterialListGenerator(elements_ranges)
material_list = generator.generate_material_list(step=10)
```

- `MaterialSimilarityCalculator`:     Computes similarity scores between materials 
  based on their vector representations. This class utilizes a Word2Vec model to 
  represent materials as vectors and then calculates their similarity to target 
  materials or properties.

**Example usage:**

```python
from MatNexus import VecGenerator

word2vec_model = VecGenerator.Word2VecModel.load("model")
property_list = ["liquid", "transparent"]
calculator = VecGenerator.MaterialSimilarityCalculator(word2vec_model, property_list)
material_list = ["H2O", "H2", "O2"]
target_material = "H2O"
top_materials = calculator.find_top_similar_materials(target_material, material_list, top_n=2)
similarity_scores = calculator.calculate_similarity_to_list(material_list, target_words=["water"])
```

## VecVisualizer
`VecVisualizer` is a module designed for visualizing word embeddings generated using 
the Word2Vec model.

**Key Classes**:
- `Word2VecVisualizer`: Represents a class responsible for visualizing word 
  embeddings. It provides functionalities such as collecting similar words and 
  visualizing word embeddings in a 2D space.

**Example usage:**
```python
from MatNexus import VecGenerator, VecVisualizer

# Load your Word2Vec model
visualizer = VecGenerator.Word2VecModel.load('your_model')

# Create Word2VecVisualizer instance
visualizer = VecVisualizer.Word2VecVisualizer('your_model')

# Prepare property list
property_list = ['your_property_list']

# Plot data using the specified plot method
fig_plot = visualizer.plot_data(property_list, plot_method='t_sne', level=1, 
                                top_n_similar=10)

```

## Usage example
A jupyter notebook how the MatNexus suite can be used is provided in 
`Example/MatNexus_example.ipynb`

## License

This project is licensed under the GNU Lesser General Public License v2.1 (LGPLv2.1). The terms and conditions for copying, distribution, and modification are outlined in the `LICENSE`file in the root directory of this project.

By using or distributing this project, you agree to abide by the terms and conditions of this license. For more details regarding the GNU LGPLv2.1, please see [GNU Lesser General Public License v2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html).

