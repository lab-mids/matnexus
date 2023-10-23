import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.manifold import TSNE

from MatNexus.VecGenerator import VectorOperations


class Word2VecVisualizer:
    """
    Class for visualizing word embeddings using Word2Vec model.
    """

    def __init__(self, model):
        """
        Initialize the Word2VecVisualizer.

        Parameters:
            model: Pre-trained Word2Vec model used for generating word embeddings.
        """
        self.model = model
        self.property_vectors = None
        self.word_vectors = None
        self.data = None

    def collect_similar_words(self, word, level, top_n_similar):
        """Recursively collect words that are similar to the target
        word up to a specified level.

        At each level, the method finds the top-n similar words to the
        target word and then recursively collects similar words for
        each of them until the specified depth (level) is reached.

        Parameters:
            word (str): The target word to start the collection from.
            level (int, optional): Depth of similarity levels to collect.
                                   Level 1 means only directly similar words to
                                   the target word will be collected.  Default is 1.

            top_n_similar (int, optional): Number of top similar words to consider
                                           at each level. Default is 30.

        Returns:
            list: A list of tuples where each tuple contains a word and its
                  corresponding depth level relative to the target word.

        """
        if level == 0:
            return [(word, level)]
        else:
            similar_words = self.model.model.wv.similar_by_word(
                word, topn=top_n_similar
            )
            collected_words = [(word, level)]
            for sim_word, _ in similar_words:
                collected_words.extend(
                    self.collect_similar_words(sim_word, level - 1, top_n_similar)
                )
            return collected_words

    def get_property_vectors(self, property_list):
        """
        Compute vectors for a list of properties using the Word2Vec model.

        Parameters:
            property_list (list): List of properties for which to compute vectors.

        Returns:
            list: Vectors corresponding to the given properties.
        """
        self.property_vectors = VectorOperations.generate_property_vectors(
            property_list, self.model
        )
        return self.property_vectors

    def get_words_data(self, property_list, level, top_n_similar):
        """
        Retrieve vectors and data for words related to given properties. This data will
        be used for visualization purposes.

        Parameters:
            property_list (list[str]): List of properties to retrieve related words for.
            level (int, optional): Depth of similarity levels to collect words from.
                Level 0 means only directly similar words to the property will be
                collected.
                Default is 0.
            top_n_similar (int, optional): Number of top similar words to consider at
                each level. Default is 30.

        Returns:
            tuple: A tuple containing the word vectors (numpy.ndarray) and a DataFrame
                with labels, levels, and words.
        """
        self.property_vectors = self.get_property_vectors(property_list)
        word_vectors = []
        word_labels = []
        word_levels = []
        exact_words = []
        distinct_words = set()

        for property_vec, property_name in zip(self.property_vectors, property_list):
            first_level_similar_words = self.model.model.wv.similar_by_vector(
                property_vec, topn=top_n_similar
            )
            for word, similarity in first_level_similar_words:
                words = self.collect_similar_words(word, level, top_n_similar)
                for word_2, word_level in words:
                    if word_2 not in distinct_words:
                        word_vectors.append(self.model.model.wv[word_2])
                        word_labels.append(property_name)
                        word_levels.append(word_level)
                        exact_words.append(word_2)
                        distinct_words.add(word_2)

        self.word_vectors = np.array(word_vectors)

        self.data = pd.DataFrame(
            {
                "label": word_labels,
                "level": word_levels,
                "word": exact_words,
            }
        )
        return self.word_vectors, self.data

    def plot_2d(
        self,
        coordinates,
        property_list,
        width=720,
        height=576,
        marker_size=5,
        textfont_size=10,
        legendfont_size=10,
        axisfont_size=10,
        tickfont_size=10,
        scale_factor=1,
        margin=dict(r=150),
    ):
        """Generate a 2D scatter plot of word embeddings.

        Parameters:

            coordinates (numpy.ndarray): 2D array of x, y coordinates for
                                         each word.

            property_list (list[str]): List of properties to label the words by.

            width (int, optional): Width of the plot. Default is 720.

            height (int, optional): Height of the plot. Default is 576.

            marker_size (int, optional): Size of the markers. Default is 5.

            textfont_size (int, optional): Size of the text font. Default is 10.

            legendfont_size (int, optional): Size of the legend font. Default is 10.

            axisfont_size (int, optional): Size of the axis font. Default is 10.

            tickfont_size (int, optional): Size of the tick font. Default is 10.

            scale_factor (float, optional): Factor to scale the plot sizes
                                            Default is 1.0.

            margin (dict, optional): Margins for the plot. Default is a right margin
                                     of 150.

        Returns:
            plotly.graph_objs._figure.Figure: A 2D scatter plot of the word embeddings.

        """
        width *= scale_factor
        height *= scale_factor
        marker_size *= scale_factor
        textfont_size *= scale_factor
        legendfont_size *= scale_factor
        axisfont_size *= scale_factor
        tickfont_size *= scale_factor

        fig = go.Figure()
        custom_color_list = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.D3
            + px.colors.qualitative.G10
        )
        shape_list = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "triangle-left",
            "triangle-right",
            "pentagon",
            "hexagon",
            "star",
            "hexagram",
            "star-triangle-up",
            "star-triangle-down",
            "star-square",
            "star-diamond",
            "diamond-tall",
            "diamond-wide",
            "hourglass",
            "bowtie",
        ]
        num_levels = self.data["level"].max() + 1

        for label, color in zip(property_list, custom_color_list):
            subset = self.data[self.data["label"] == label]
            for level, shape in zip(range(num_levels), shape_list[:num_levels]):
                level_subset = subset[subset["level"] == level]
                hovertext = [
                    f"{label} - Word: {row.word} - Level: {row.level}"
                    for _, row in level_subset.iterrows()
                ]

                fig.add_trace(
                    go.Scatter(
                        x=coordinates[level_subset.index, 0],
                        y=coordinates[level_subset.index, 1],
                        mode="markers",
                        name=f"{label} - Level {level}",
                        marker=dict(color=color, size=marker_size, symbol=shape),
                        text=hovertext,
                        hoverinfo="text",
                        textfont=dict(size=textfont_size),
                        hoverlabel=dict(font_size=textfont_size),
                    )
                )

        fig.update_layout(
            margin=margin,
            title=None,
            width=width,
            height=height,
            legend_font_size=legendfont_size,
            xaxis=dict(
                title_text="t-SNE Dimension 1",
                title_font_size=axisfont_size,
                tickfont_size=tickfont_size,
            ),
            yaxis=dict(
                title_text="t-SNE Dimension 2",
                title_font_size=axisfont_size,
                tickfont_size=tickfont_size,
            ),
        )

        return fig

    def plot_data(
        self,
        property_list,
        plot_method="t_sne",
        level=0,
        top_n_similar=30,
        width=720,
        height=576,
        marker_size=5,
        scale_factor=1,
        textfont_size=10,
        legendfont_size=10,
        axisfont_size=10,
        tickfont_size=10,
        margin=dict(r=150),
        **kwargs,
    ):
        """Visualize word embeddings in 2D using various
        dimensionality reduction techniques.

        Parameters:
            property_list (list[str]): List of properties to visualize
                                       related words for.

            plot_method (str, optional): The dimensionality reduction technique to use.
                                         Options are: 'isomap', 'md_scaling',
                                         'spectral', or 't_sne'. Default is 't_sne'.

            level (int, optional): Depth of similarity levels to collect words from.
                                   Default is 1.

            top_n_similar (int, optional): Number of top similar words to consider.
                                           Default is 30.

            width (int, optional): Width of the plot. Default is 720.

            height (int, optional): Height of the plot. Default is 576.

            marker_size (int, optional): Size of the markers. Default is 5.

            scale_factor (float, optional): Factor to scale the plot sizes.
                                            Default is 1.0.

            textfont_size (int, optional): Size of the text font. Default is 10.

            legendfont_size (int, optional): Size of the legend font. Default is 10.

            axisfont_size (int, optional): Size of the axis font. Default is 10.

            tickfont_size (int, optional): Size of the tick font. Default is 10.

            margin (dict, optional): Margins for the plot. Default is a right margin
                                     of 150.

            **kwargs: Additional keyword arguments for the dimensionality reduction
                      technique.

        Returns:
            plotly.graph_objs._figure.Figure: A 2D scatter plot of the word embeddings.

        """
        self.get_words_data(property_list, level=level, top_n_similar=top_n_similar)
        if plot_method == "isomap":
            isomap = manifold.Isomap(**kwargs)
            coordinates = isomap.fit_transform(self.word_vectors)
        elif plot_method == "md_scaling":
            md_scaling = manifold.MDS(**kwargs)
            coordinates = md_scaling.fit_transform(self.word_vectors)
        elif plot_method == "spectral":
            spectral = manifold.SpectralEmbedding(**kwargs)
            coordinates = spectral.fit_transform(self.word_vectors)
        elif plot_method == "t_sne":
            t_sne = manifold.TSNE(**kwargs)
            coordinates = t_sne.fit_transform(self.word_vectors)
        else:
            raise ValueError(
                "Invalid plot_method. Choose from 'isomap', "
                "'md_scaling', 'spectral', or 't_sne'."
            )

        return self.plot_2d(
            coordinates,
            property_list,
            width=width,
            height=height,
            marker_size=marker_size,
            textfont_size=textfont_size,
            legendfont_size=legendfont_size,
            axisfont_size=axisfont_size,
            tickfont_size=tickfont_size,
            scale_factor=scale_factor,
            margin=margin,
        )

    def plot_material_vectors(
        self,
        material_list,
        width=720,
        height=576,
        scale_factor=1,
        marker_size=15,
        textfont_size=5,
        legendfont_size=10,
        axisfont_size=10,
        tickfont_size=10,
        **kwargs,
    ):
        """
        Plot a 2D scatter plot of material vectors using t-SNE.

        Parameters:
            material_list (list[str]): List of materials for which the vectors are
                                       to be plotted.

            width (int, optional): Width of the plot. Default is 720.

            height (int, optional): Height of the plot. Default is 576.

            scale_factor (float, optional): Factor to scale the plot sizes.
                                            Default is 1.0.

            marker_size (int, optional): Size of the markers. Default is 15.

            textfont_size (int, optional): Size of the text font. Default is 5.

            legendfont_size (int, optional): Size of the legend font. Default is 10.

            axisfont_size (int, optional): Size of the axis font. Default is 10.

            tickfont_size (int, optional): Size of the tick font. Default is 10.

            **kwargs: Additional keyword arguments for the t-SNE method.

        Returns:
            plotly.graph_objs._figure.Figure: A 2D scatter plot of the material vectors.
        """

        # Scale sizes
        width *= scale_factor
        height *= scale_factor
        marker_size *= scale_factor
        textfont_size *= scale_factor
        legendfont_size *= scale_factor
        axisfont_size *= scale_factor
        tickfont_size *= scale_factor

        material_vectors = [
            VectorOperations.generate_material_vector(material, self.model)
            for material in material_list
        ]
        material_vectors = np.vstack(material_vectors)

        labels = [material for material in material_list]
        tsne = TSNE(n_components=2, random_state=0, perplexity=2)

        vectors_2d = tsne.fit_transform(material_vectors)

        df = pd.DataFrame(
            {"x": vectors_2d[:, 0], "y": vectors_2d[:, 1], "label": labels}
        )

        custom_color_list = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.D3
            + px.colors.qualitative.G10
        )

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            text="label",
            width=width,
            height=height,
            color_discrete_sequence=custom_color_list,
        )
        fig.update_traces(
            textposition="middle center",
            marker_size=marker_size,
            textfont_size=textfont_size,
            hoverlabel=dict(font_size=textfont_size),
        )

        fig.update_layout(
            autosize=False,
            legend_title_text=None,
            legend_font_size=legendfont_size,
            xaxis=dict(
                title_text="t-SNE Dimension 1",
                title_font_size=axisfont_size,
                tickfont=dict(size=tickfont_size),
            ),
            yaxis=dict(
                title_text="t-SNE Dimension 2",
                title_font_size=axisfont_size,
                tickfont=dict(size=tickfont_size),
            ),
        )

        return fig

    def plot_similarity_scatter(
        self, data, elements, nrows=2, ncols=3, figsize=(20, 10)
    ):
        """Plot a scatter plot showing the similarity of materials
        based on specified elements.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the materials and
                                 their similarity scores.

            elements (list[str]): The list of elements to include in the plot.

            nrows (int, optional): The number of rows in the plot grid.
                                   Default is 2.

            ncols (int, optional): The number of columns in the plot grid.
                                   Default is 3.

            figsize (tuple, optional): The size (width, height) of the figure in inches.
                                       Default is (20, 10).

        Returns:
            matplotlib.figure.Figure: A scatter plot showing the similarity of
                                      materials.

        """
        total_subplots = nrows * ncols
        num_elements = len(elements)

        if num_elements > total_subplots:
            raise ValueError("Too many elements for the provided grid size")

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()  # Flatten the axes array

        for i in range(total_subplots):
            if i < num_elements:
                element = elements[i]
                if element == "Similarity" or element == "Experimental_Indicator":
                    cmap = "plasma"
                else:
                    cmap = "viridis"
                sc = axes[i].scatter(
                    data["x"], data["y"], c=data[element], cmap=cmap, marker="o", s=100
                )
                axes[i].set_title(element)
                fig.colorbar(sc, ax=axes[i])
            else:
                fig.delaxes(axes[i])  # Remove the unused subplot

        plt.tight_layout()

        return fig
