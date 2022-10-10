from unittest.mock import MagicMock

import pandas as pd
import pytest
from feast import ValueType
from feast.errors import DataSourceNoNameException, DataSourceNotFoundException

from ibm_data_engine import (
    DataEngineDataSource,
    DataEngineOfflineStoreConfig,
    DataEngineSchemaError,
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
        return pd.DataFrame(
            {"col_name": ["A", "B", "C"], "data_type": ["binary", "integer", "string"]}
        )

    def get_schema_data(self, cos, type):
        if "FAIL" in cos:
            raise RuntimeError()
        return pd.DataFrame(
            {"col_name": ["A", "B", "C"], "data_type": ["binary", "integer", "string"]}
        )

    def get_cos_summary(self, cos):
        if "FAIL" in cos:
            raise RuntimeError()

    def execute_sql(self, sql):
        if "FAIL" in sql:
            raise RuntimeError()


class TestDataEngineDataSource:
    def test_eq(self):
        assert DataEngineDataSource(table="table") == DataEngineDataSource(table="table")
        with pytest.raises(TypeError):
            assert DataEngineDataSource(table="table") == 0

    def test_hash(self):
        assert (
            DataEngineDataSource(table="table").__hash__()
            == DataEngineDataSource(table="table").__hash__()
        )

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
        DataEngineDataSource(table="cos://url").validate(config)
        DataEngineDataSource(query="select * from table", name="name").validate(config)
        with pytest.raises(DataSourceNotFoundException):
            DataEngineDataSource(table="FAIL").validate(config)
        with pytest.raises(DataSourceNotFoundException):
            DataEngineDataSource(name="name", query="FAIL").validate(config)
        with pytest.raises(DataSourceNotFoundException):
            DataEngineDataSource(table="cos://FAIL").validate(config)

    def test_get_table_query_string(self):
        assert DataEngineDataSource(table="table").get_table_query_string() == "`table`"
        assert (
            DataEngineDataSource(query="<query>", name="name").get_table_query_string()
            == "(<query>)"
        )

    def test_get_table_column_names_and_types(self, monkeypatch):
        monkeypatch.setattr(data_engine_offline_store, "SQLQuery", SQLQueryMock)
        config = MagicMock(
            offline_store=DataEngineOfflineStoreConfig(
                api_key="API_KEY", instance_crn="CRN", target_cos_url="TARGET"
            )
        )
        assert DataEngineDataSource(table="table").get_table_column_names_and_types(config) == [
            ("A", "binary"),
            ("B", "integer"),
            ("C", "string"),
        ]
        assert DataEngineDataSource(table="cos://uri").get_table_column_names_and_types(config) == [
            ("A", "binary"),
            ("B", "integer"),
            ("C", "string"),
        ]
        with pytest.raises(DataEngineSchemaError):
            DataEngineDataSource(table="FAIL").get_table_column_names_and_types(config)
        with pytest.raises(DataEngineSchemaError):
            DataEngineDataSource(table="cos://FAIL").get_table_column_names_and_types(config)
        with pytest.raises(ValueError):
            DataEngineDataSource(name="name", query="query").get_table_column_names_and_types(
                config
            )

    def test_source_datatype_to_feast_value_type(self):
        """Regression test for the correct mapping."""
        types = DataEngineDataSource.source_datatype_to_feast_value_type()
        assert types("binary") == ValueType.BYTES
        assert types("array<binary>") == ValueType.BYTES_LIST
        assert types("binary") == ValueType.BYTES
        assert types("boolean") == ValueType.BOOL
        assert types("tinyint") == ValueType.INT32
        assert types("smallint") == ValueType.INT32
        assert types("int") == ValueType.INT32
        assert types("integer") == ValueType.INT32
        assert types("bigint") == ValueType.INT64
        assert types("long") == ValueType.INT64
        assert types("float") == ValueType.FLOAT
        assert types("double") == ValueType.DOUBLE
        assert types("decimal") == ValueType.DOUBLE
        assert types("string") == ValueType.STRING
        assert types("timestamp") == ValueType.UNIX_TIMESTAMP
        assert types("array<binary>") == ValueType.BYTES_LIST
        assert types("array<boolean>") == ValueType.BOOL_LIST
        assert types("array<tinyint>") == ValueType.INT32_LIST
        assert types("array<smallint>") == ValueType.INT32_LIST
        assert types("array<int>") == ValueType.INT32_LIST
        assert types("array<integer>") == ValueType.INT32_LIST
        assert types("array<bigint>") == ValueType.INT64_LIST
        assert types("array<long>") == ValueType.INT64_LIST
        assert types("array<float>") == ValueType.FLOAT_LIST
        assert types("array<double>") == ValueType.DOUBLE_LIST
        assert types("array<decimal>") == ValueType.DOUBLE_LIST
        assert types("array<string>") == ValueType.STRING_LIST
        assert types("array<timestamp>") == ValueType.UNIX_TIMESTAMP_LIST
