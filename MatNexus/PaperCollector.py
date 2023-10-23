import os

import plotly.graph_objects as go
import pandas as pd


class Paper:
    """
    Represents an individual paper with properties inherited from pybliometrics.
    This class extracts details like DOI, title, year, etc., for a given paper.

    Attributes:
        eid (str): The EID (Electronic Identifier) of the paper in Scopus.
        doi (str): Digital Object Identifier of the paper.
        title (str): Title of the paper.
        year (str): Year of publication.
        citedby_count (int): Number of times the paper was cited.
        cache_file_mdate (str): Cache file modification date.
        abstract (str): Abstract of the paper.

    Example:
        paper = Paper(eid)
        print(paper.title)
        print(paper.year)
    """

    def __init__(self, eid: str):
        """
        Initialize a Paper object.

        Parameters:
            eid (str): The EID (Electronic Identifier) of the paper in Scopus.
        """
        abstract = AbstractRetrieval(eid, view="FULL")
        self.eid = eid
        self.doi = abstract.doi
        self.title = abstract.title
        self.year = abstract.coverDate[:4]  # Extract year from coverDate
        self.citedby_count = abstract.citedby_count
        self.cache_file_mdate = abstract.get_cache_file_mdate()
        self.abstract = abstract.abstract


class ScopusPaperCollector:
    """
    Collects and processes papers from Scopus using specified queries.
    This class allows users to retrieve, sort, and visualize paper data from Scopus.

    Attributes:
        query (str): The search query for Scopus.
        limit (int, optional): The maximum number of results to retrieve.
        results (pd.DataFrame): DataFrame containing the results of the search.

    Example:
        collector = ScopusPaperCollector(query, limit=100,
                                         config_path="path/to/config.ini")
        collector.collect_papers()
        sorted_papers = collector.sort_by("citedby_count", ascending=False)
        collector.plot_publications_per_year(width=0.3)
    """

    def __init__(self, query: str = None, limit: int = None, config_path: str = None):
        """
        Initialize a ScopusPaperCollector object.

        Parameters:
            query (str): The search query for Scopus.
            limit (int, optional): The maximum number of results to retrieve.
            config_path (str, optional): Path to Pybliometrics config file.
        """
        self.query = query
        self.limit = limit
        self.results = None
        self._set_config_path(config_path)
        self._import_pybliometrics()

    @staticmethod
    def build_query(
        keywords=None, startyear=None, endyear=None, fixyear=None, openaccess=None
    ):
        """
        Build a Scopus search query based on the provided parameters.

        Parameters:
            keywords (str, optional): The keywords to search for.
            startyear (int, optional): The start year for the search.
            endyear (int, optional): The end year for the search.
            fixyear (int, optional): A specific year to search within.
            openaccess (bool, optional): Whether to search for open access.

        Returns:
            query (str): A query string that fits the syntax of Scopus.
        """
        if not any([keywords, startyear, endyear, fixyear, openaccess]):
            raise ValueError("At least one search parameter must be provided.")

        query = ""

        if keywords:
            query += "ALL('{}')".format(keywords)
        if startyear or endyear:
            query += " AND PUBYEAR < {}".format(endyear) if endyear else ""
            query += " AND PUBYEAR > {}".format(startyear) if startyear else ""
        elif fixyear:
            query += " AND PUBYEAR = {}".format(fixyear)
        if openaccess:
            query += " AND OA(all)"

        return query

    @staticmethod
    def _set_config_path(config_path: str):
        """
        Set the path for the Pybliometrics configuration file.

        Parameters:
            config_path (str): The path to the Pybliometrics config file.
        """
        if config_path is not None:
            os.environ["PYB_CONFIG_FILE"] = config_path
        else:
            # Windows version
            os.environ["PYB_CONFIG_FILE"] = "\.pybliometrics\pybliometrics.cfg"  # noqa,

    @staticmethod
    def _import_pybliometrics():
        """
        Import the required pybliometrics classes: ScopusSearch and
        AbstractRetrieval.
        """
        global ScopusSearch, AbstractRetrieval
        from pybliometrics.scopus import ScopusSearch, AbstractRetrieval

    def collect_papers(self, sort="relevancy"):
        """
        Collect papers from Scopus based on the specified query.

        Parameters:
            sort (str, optional): The sorting option for the results.
        """
        valid_sort_options = ["relevancy", "date", "citedby-count"]
        if sort not in valid_sort_options:
            raise ValueError(
                f"Invalid sort option '{sort}'. "
                f"Valid options are {valid_sort_options}."
            )

        search = ScopusSearch(self.query)
        eids = search.get_eids()

        if self.limit:
            eids = eids[: self.limit]

        papers = [Paper(eid) for eid in eids]
        self.results = pd.DataFrame(
            [vars(p) for p in papers], index=[p.eid for p in papers]
        )

    def sort_by(self, column: str, ascending: bool = True) -> pd.DataFrame:
        """
        Sort the results DataFrame by a specified column.

        Parameters:
            column (str): The column to sort by.
            ascending (bool, optional): Whether to sort in ascending order.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        if self.results is None:
            raise ValueError("No results found. Run collect_papers() first.")

        return self.results.sort_values(by=column, ascending=ascending)

    def plot_publications_per_year(self, **kwargs):
        """
        Plot the number of publications per year using Plotly.

        Parameters:
            **kwargs: Additional keyword arguments to pass to go.Bar function.

        Example:
            collector.plot_publications_per_year(width=0.3)
        """
        if self.results is None:
            raise ValueError("No results found. Run collect_papers() first.")

        yearly_counts = self.results.groupby("year").size()

        fig = go.Figure(go.Bar(x=yearly_counts.index, y=yearly_counts.values, **kwargs))
        fig.update_layout(
            title="Number of Publications per Year",
            xaxis_title="Year",
            yaxis_title="Number of Publications",
        )

        return fig
