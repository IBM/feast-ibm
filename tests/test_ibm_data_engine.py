from unittest.mock import MagicMock

import pytest
from feast.errors import DataSourceNoNameException, DataSourceNotFoundException

from ibm_data_engine import (
    DataEngineDataSource,
    DataEngineOfflineStoreConfig,
    __version__,
    data_engine_offline_store,
)


def test_version():
    assert __version__ == "0.25.0"


class SQLQueryMock(data_engine_offline_store.SQLQuery):
    def __init__(self, api_key, instance_crn, target_cos_url):
        self.api_key = api_key
        self.instance_crn = instance_crn
        self.target_cos_url = target_cos_url

    def get_schema_table(self, table):
        if table == "FAIL":
            return None
        return "ok"

    def get_cos_summary(self, cos):
        if "FAIL" in cos:
            raise RuntimeError()

    def execute_sql(self, sql):
        if "FAIL" in sql:
            raise RuntimeError()


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

    def test_validate_cos(self, monkeypatch):
        monkeypatch.setattr(data_engine_offline_store, "SQLQuery", SQLQueryMock)
        config = MagicMock(
            offline_store=DataEngineOfflineStoreConfig(
                api_key="API_KEY", instance_crn="CRN", target_cos_url="TARGET"
            )
        )
        DataEngineDataSource(table="table").validate(config)
        with pytest.raises(DataSourceNotFoundException):
            DataEngineDataSource(table="FAIL").validate(config)
        with pytest.raises(DataSourceNotFoundException):
            DataEngineDataSource(name="name", query="FAIL").validate(config)
        with pytest.raises(DataSourceNotFoundException):
            DataEngineDataSource(table="cos://FAIL").validate(config)
