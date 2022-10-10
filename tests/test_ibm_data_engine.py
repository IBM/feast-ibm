import pytest
from feast.errors import DataSourceNoNameException

from ibm_data_engine import DataEngineDataSource, __version__


def test_version():
    assert __version__ == "0.25.0"


class TestDataEngineDataSource:
    def test_must_have_name_or_table(self):
        with pytest.raises(DataSourceNoNameException):
            DataEngineDataSource(query="<query>")
        DataEngineDataSource(table="table", query="<query>")
        DataEngineDataSource(name="name", query="<query>")

    def test_must_have_query_or_table(self):
        with pytest.raises(ValueError):
            DataEngineDataSource()
        DataEngineDataSource(table="table")
        DataEngineDataSource(query="<query>", name="name")

    def test_proto(self):
        source = DataEngineDataSource(table="table", query="query")
        assert source == DataEngineDataSource.from_proto(source.to_proto())
