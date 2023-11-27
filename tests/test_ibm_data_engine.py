# Copyright 2022 Abhay Ratnaparkhi, Michal Siedlaczek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pandas as pd
import pyarrow
import pytest
import testfixtures

from feast import RepoConfig, ValueType, OnDemandFeatureView
from feast.errors import DataSourceNoNameException, DataSourceNotFoundException
from feast.infra.offline_stores import offline_utils
from feast.infra.offline_stores.offline_store import RetrievalMetadata
from feast.infra.registry.base_registry import BaseRegistry
from feast.infra.registry.registry import Registry
from feast.repo_config import RegistryConfig
from ibmcloudsql import SQLQuery
from pandas.testing import assert_frame_equal

from ibm_data_engine import (
    DataEngineDataSource,
    DataEngineOfflineStore,
    DataEngineOfflineStoreConfig,
    DataEngineRetrievalJob,
    DataEngineSchemaError,
    __version__,
    data_engine_offline_store,
)
from tests.test_integration import get_driver_feature_view, EXPECTED_QUERY


def test_version():
    assert __version__ == "0.25.0"


class SQLQueryMock(data_engine_offline_store.SQLQuery):
    def __init__(self, api_key, instance_crn, target_cos_url, thread_safe=False):
        self.api_key = api_key
        self.instance_crn = instance_crn
        self.target_cos_url = target_cos_url
        self._thread_safe = thread_safe

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

    def test_sql_from(self):
        sql = lambda: SQLQuery(api_key="API", instance_crn="CRN")
        assert DataEngineDataSource(table="table").sql_from(sql()).get_sql() == "\nFROM TABLE"
        assert (
            DataEngineDataSource(table="cos://blah").sql_from(sql()).get_sql()
            == "\nFROM cos://blah stored AS parquet"
        )
        assert (
            DataEngineDataSource(table="cos://blah", cos_type="csv").sql_from(sql()).get_sql()
            == "\nFROM cos://blah stored AS csv"
        )
        assert (
            DataEngineDataSource(name="name", query="query").sql_from(sql()).get_sql()
            == "\nFROM (query)"
        )

    def test_sql_from_with_alias(self):
        sql = lambda: SQLQuery(api_key="API", instance_crn="CRN")
        assert (
            DataEngineDataSource(table="table").sql_from(sql(), alias="t").get_sql()
            == "\nFROM TABLE t"
        )
        assert (
            DataEngineDataSource(table="cos://blah").sql_from(sql(), alias="t").get_sql()
            == "\nFROM cos://blah stored AS parquet t"
        )
        assert (
            DataEngineDataSource(table="cos://blah", cos_type="csv")
            .sql_from(sql(), alias="t")
            .get_sql()
            == "\nFROM cos://blah stored AS csv t"
        )
        assert (
            DataEngineDataSource(name="name", query="query").sql_from(sql(), alias="t").get_sql()
            == "\nFROM (query)"
        )


class TestRetrievalJob:
    def test_metadata(self):
        """Metadata is simply passed to the constructor."""
        metadata = RetrievalMetadata(features=["f1", "f2"], keys=["k1"])
        assert DataEngineRetrievalJob(lambda: pd.DataFrame(), metadata).metadata == metadata

    def test_to_df(self):
        metadata = RetrievalMetadata(features=["f1", "f2"], keys=["k1"])
        df = pd.DataFrame({"f1": [1, 2], "f2": ["a", "b"], "k1": [0, 1]})
        assert_frame_equal(DataEngineRetrievalJob(lambda: df, metadata).to_df(), df)

    def test_to_arrow(self):
        metadata = RetrievalMetadata(features=["f1", "f2"], keys=["k1"])
        df = pd.DataFrame({"f1": [1, 2], "f2": ["a", "b"], "k1": [0, 1]})
        assert (
            DataEngineRetrievalJob(lambda: df, metadata)
            .to_arrow()
            .equals(pyarrow.Table.from_pandas(df))
        )


class MockRegistry(Registry):
    def list_on_demand_feature_views(
        self, project: str, allow_cache: bool = False
    ) -> List[OnDemandFeatureView]:
        OnDemandFeatureView
        return []


def test_cast_timestamp():
    assert (
        data_engine_offline_store.cast_timestamp(datetime(year=2022, month=10, day=11))
        == "CAST('2022-10-11 00:00:00.000000' AS TIMESTAMP)"
    )


class TestDataEngineOfflineStore:
    def test_pull_all_from_table_or_query(self, monkeypatch):
        df = pd.DataFrame({"f1": [1, 2], "f2": ["a", "b"], "k1": [0, 1]})
        sql = SQLQuery(
            api_key="API",
            instance_crn="CRN",
            target_cos_url="cos://us-south/sql/sql",
        )
        sql.run = MagicMock(return_value=MagicMock(data=df))
        monkeypatch.setattr(data_engine_offline_store, "_sql_builder", lambda _: sql)
        src = DataEngineDataSource(table="document_features")
        config = DataEngineOfflineStoreConfig(
            api_key="API",
            instance_crn="CRN",
            target_cos_url="cos://us-south/sql/sql",
        )
        repo_config = RepoConfig(
            project="test",
            provider="local",
            offline_store=config,
            entity_key_serialization_version=2,
        )
        offline_store = DataEngineOfflineStore()
        job = offline_store.pull_all_from_table_or_query(
            config=repo_config,
            data_source=src,
            join_key_columns=["docid"],
            feature_name_columns=["source"],
            timestamp_field="timestamp",
            start_date=datetime.strptime(
                "2022-08-12 09:09:27.108650",
                data_engine_offline_store.TIMESTAMP_FORMAT,
            ),
            end_date=datetime.strptime(
                "2022-08-12 09:09:27.108652",
                data_engine_offline_store.TIMESTAMP_FORMAT,
            ),
        )
        assert_frame_equal(job.to_df(), df)
        sql.run.assert_called_once_with(get_result=True)
        assert (
            re.sub("\\s+", " ", sql.get_sql())
            == "SELECT docid, SOURCE, timestamp FROM document_features "
            "WHERE timestamp BETWEEN cast('2022-08-12 09:09:27.108650' AS TIMESTAMP) "
            "AND cast('2022-08-12 09:09:27.108652' AS TIMESTAMP)"
        )

    def test_pull_latest_from_table_or_query(self, monkeypatch):
        df = pd.DataFrame({"f1": [1, 2], "f2": ["a", "b"], "k1": [0, 1]})
        sql = SQLQuery(
            api_key="API",
            instance_crn="CRN",
            target_cos_url="cos://us-south/sql/sql",
        )
        sql.run_sql = MagicMock(return_value=df)
        monkeypatch.setattr(data_engine_offline_store, "_sql_builder", lambda _: sql)
        src = DataEngineDataSource(table="document_features")
        config = DataEngineOfflineStoreConfig(
            api_key="API",
            instance_crn="CRN",
            target_cos_url="cos://us-south/sql/sql",
        )
        repo_config = RepoConfig(
            project="test",
            provider="local",
            offline_store=config,
            entity_key_serialization_version=2,
        )
        offline_store = DataEngineOfflineStore()
        job = offline_store.pull_latest_from_table_or_query(
            config=repo_config,
            data_source=src,
            join_key_columns=["docid"],
            feature_name_columns=["source"],
            timestamp_field="timestamp",
            created_timestamp_column=None,
            start_date=datetime.strptime(
                "2022-08-12 09:09:27.108650",
                data_engine_offline_store.TIMESTAMP_FORMAT,
            ),
            end_date=datetime.strptime(
                "2022-08-12 09:09:27.108652",
                data_engine_offline_store.TIMESTAMP_FORMAT,
            ),
        )
        assert_frame_equal(job.to_df(), df)
        expected_arg = "SELECT de_a.docid, de_a.source, de_a.timestamp FROM `document_features` as de_a JOIN (SELECT docid,\n       max(timestamp) AS timestamp\nFROM document_features\nWHERE timestamp BETWEEN cast('2022-08-12 09:09:27.108650' AS TIMESTAMP) AND cast('2022-08-12 09:09:27.108652' AS TIMESTAMP)\nGROUP BY docid) as de_b WHERE de_a.docid = de_b.docid AND de_a.timestamp = de_b.timestamp"
        sql.run_sql.assert_called_once_with(expected_arg)

    def test_get_historical_features(self, monkeypatch):
        offline_store = DataEngineOfflineStore()
        driver, driver_stats_source, driver_stats_feature_view = get_driver_feature_view()

        expected_df = pd.DataFrame(
            {
                "driver_id": [1001, 1002, 1003],
                "event_timestamp": [
                    datetime(2021, 4, 12, 10, 59, 42),
                    datetime(2021, 4, 12, 8, 12, 10),
                    datetime(2021, 4, 12, 16, 40, 26),
                ],
                "conv_rate": [1.0, 2.0, 3.0],
                "acc_rate": [1.0, 1.0, 0.0],
                "avg_daily_trips": [200, 300, 400],
                "label_driver_reported_satisfaction": [1, 5, 3],
            }
        )

        sql = SQLQuery(
            api_key="API",
            instance_crn="CRN",
            target_cos_url="cos://us-south/sql/sql",
        )

        sql.run_sql = MagicMock(return_value=expected_df)
        mock_upload = MagicMock(return_value=MagicMock(data="cospath"))
        mock_delete = MagicMock()

        monkeypatch.setattr(data_engine_offline_store, "_sql_builder", lambda _: sql)
        monkeypatch.setattr(data_engine_offline_store, "_upload_entity_df", mock_upload)
        monkeypatch.setattr(data_engine_offline_store, "_delete_entity_df", mock_delete)
        monkeypatch.setattr(
            offline_utils, "get_temp_entity_table_name", lambda: "FEAST_TEMP_ENTITY_TABLE"
        )

        entity_df = pd.DataFrame.from_dict(
            {
                "driver_id": [1001, 1002, 1003],
                "event_timestamp": [
                    datetime(2021, 4, 12, 10, 59, 42),
                    datetime(2021, 4, 12, 8, 12, 10),
                    datetime(2021, 4, 12, 16, 40, 26),
                ],
                "label_driver_reported_satisfaction": [1, 5, 3],
            }
        )

        config = DataEngineOfflineStoreConfig(
            api_key="API",
            instance_crn="CRN",
            target_cos_url="cos://us-south/sql/sql",
        )
        repo_config = RepoConfig(
            project="test_plugin",
            provider="local",
            offline_store=config,
            entity_key_serialization_version=2,
        )

        mock_registry = MockRegistry(
            RegistryConfig(path="../integration-test/registry.db"), Path("./tests")
        )
        job = offline_store.get_historical_features(
            config=repo_config,
            feature_views=[driver_stats_feature_view],
            feature_refs=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
            entity_df=entity_df,
            registry=mock_registry,
            project="test_plugin",
            full_feature_names=False,
        )
        assert isinstance(job, DataEngineRetrievalJob)
        training_df = job.to_df()
        query = sql.run_sql.call_args_list[0].args[0]
        assert (
            testfixtures.compare(query, EXPECTED_QUERY, blanklines=False, trailing_whitespace=False)
            is None
        )
        assert_frame_equal(training_df, expected_df)

        with pytest.raises(NotImplementedError):
            offline_store.get_historical_features(
                config=repo_config,
                feature_views=[driver_stats_feature_view],
                feature_refs=[
                    "driver_hourly_stats:conv_rate",
                    "driver_hourly_stats:acc_rate",
                    "driver_hourly_stats:avg_daily_trips",
                ],
                entity_df="driver",
                registry=mock_registry,
                project="test_plugin",
                full_feature_names=False,
            )
