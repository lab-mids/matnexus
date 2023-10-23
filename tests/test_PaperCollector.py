import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
from MatNexus.PaperCollector import ScopusPaperCollector


class TestScopusPaperCollector(unittest.TestCase):
    def setUp(self):
        self.keywords = "electrocatalyst"
        self.fixyear = 2015
        self.openaccess = True

        self.query = ScopusPaperCollector.build_query(
            keywords=self.keywords, fixyear=self.fixyear, openaccess=self.openaccess
        )

        self.limit = 10
        self.collector = ScopusPaperCollector(self.query, limit=self.limit)

        # Enhanced mock results for the collector
        mock_results = MagicMock()
        mock_results.get_eids.return_value = ["eid1", "eid2", "eid3"]
        mock_results.groupby().size().index.return_value = [2015, 2016, 2017]
        mock_results.groupby().size().values.return_value = [10, 20, 30]
        mock_data = [
            {"citedby_count": 30, "eid": "eid1", "year": 2015},
            {"citedby_count": 20, "eid": "eid2"},
            {"citedby_count": 10, "eid": "eid3"},
        ]
        mock_df = pd.DataFrame(mock_data)
        type(mock_results).citedby_count = PropertyMock(
            return_value=mock_df["citedby_count"]
        )
        self.collector.results = mock_df

    def test_build_query(self):
        query = ScopusPaperCollector.build_query(
            keywords=self.keywords, fixyear=self.fixyear, openaccess=self.openaccess
        )
        self.assertIn("electrocatalyst", query)
        self.assertIn("2015", query)

    @patch("MatNexus.PaperCollector.ScopusSearch")
    @patch("MatNexus.PaperCollector.AbstractRetrieval", MagicMock())
    def test_collect_papers(self, MockScopusSearch):
        mock_search = MockScopusSearch()
        mock_search.get_eids.return_value = ["eid1", "eid2", "eid3"]
        self.collector.collect_papers()
        self.assertIsNotNone(self.collector.results)
        self.assertTrue(len(self.collector.results) > 0)

    def test_sort_by(self):
        sorted_papers = self.collector.sort_by("citedby_count", ascending=False)
        self.assertEqual(
            sorted_papers.iloc[0]["citedby_count"], max(sorted_papers["citedby_count"])
        )

    def test_plot_publications_per_year(self):
        try:
            # Just testing if a figure is created
            fig = self.collector.plot_publications_per_year()  # noqa
            success = True
        except Exception as e:
            print(f"Error in plot_publications_per_year: {e}")
            success = False
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
