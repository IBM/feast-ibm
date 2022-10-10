"""Implementation of the offline store backed by the IBM Cloud Data Engine."""

import json
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import pyarrow
from feast.data_source import DataSource
from feast.errors import DataSourceNoNameException
from feast.feature_view import FeatureView
from feast.infra.offline_stores.offline_store import OfflineStore, RetrievalJob, RetrievalMetadata
from feast.infra.registry.registry import Registry

# pylint: disable=no-name-in-module
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.saved_dataset import SavedDatasetStorage
from feast.value_type import ValueType
from typing_extensions import Literal


class DataEngineOfflineStoreConfig(FeastConfigBaseModel):
    """Offline store config for IBM Cloud Data Engine."""

    type: Literal[
        "ibm_data_engine.DataEngineOfflineStore"
    ] = "ibm_data_engine.DataEngineOfflineStore"


class DataEngineDataSource(DataSource):
    """Custom data source class for local files"""

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        table: Optional[str] = None,
        created_timestamp_column: Optional[str] = "",
        field_mapping: Optional[Dict[str, str]] = None,
        query: Optional[str] = None,
        description: Optional[str] = "",
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = "",
    ):
        """Create a DataEngineDataSource from an existing table or query.
        Args:
            name (optional): Name for the source. Defaults to the table if not specified.
            timestamp_field (optional): Event timestamp field used for point in time
                joins of feature values.
            table (optional): The Data Engine table where features can be found.
            created_timestamp_column (optional): Timestamp column when row was created, used for
                deduplicating rows.
            field_mapping (optional): A dictionary mapping of column names in this data source to
                feature names in a feature table or view. Only used for feature columns, not
                entities or timestamp columns.
            query (optional): SQL query to execute to generate data for this data source.
            description (optional): A human-readable description.
            tags (optional): A dictionary of key-value pairs to store arbitrary metadata.
            owner (optional): The owner of the bigquery source, typically the email of the primary
                maintainer.
        """
        if table is None and query is None:
            raise ValueError('No "table" or "query" argument provided.')

        self.table = table
        self.query = query

        # If no name, use the table as the default name.
        if name is None and table is None:
            raise DataSourceNoNameException()
        name = name or table
        assert name

        super().__init__(
            name=name,
            timestamp_field=timestamp_field,
            created_timestamp_column=created_timestamp_column,
            field_mapping=field_mapping,
            description=description,
            tags=tags,
            owner=owner,
        )

    @staticmethod
    def from_proto(data_source: DataSourceProto):
        custom_source_options = str(data_source.custom_options.configuration, encoding="utf8")
        table = json.loads(custom_source_options)["table"]
        query = json.loads(custom_source_options)["query"]
        return DataEngineDataSource(
            name=data_source.name,
            field_mapping=dict(data_source.field_mapping),
            table=table,
            timestamp_field=data_source.timestamp_field,
            created_timestamp_column=data_source.created_timestamp_column,
            query=query,
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
        )

    def to_proto(self) -> DataSourceProto:
        config_json = json.dumps({"table": self.table, "query": self.query})
        return DataSourceProto(
            name=self.name,
            type=DataSourceProto.CUSTOM_SOURCE,
            field_mapping=self.field_mapping,
            description=self.description,
            tags=self.tags,
            owner=self.owner,
            timestamp_field=self.timestamp_field,
            created_timestamp_column=self.created_timestamp_column,
            custom_options=DataSourceProto.CustomSourceOptions(
                configuration=bytes(config_json, encoding="utf8")
            ),
        )

    def validate(self, config: RepoConfig):
        # This will validate if the config is correct, e.g., check for existance
        # of the given table.
        raise NotImplementedError

    def get_table_query_string(self) -> str:
        """Returns a string that can directly be used to reference this table in SQL"""
        if self.table:
            return f"`{self.table}`"
        return f"({self.query})"

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        """
        Returns the callable method that returns Feast type given the raw column type.
        """
        raise NotImplementedError

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, DataEngineDataSource):
            raise TypeError("Comparisons should only involve DataEngineDataSource class objects.")
        return super().__eq__(other) and self.table == other.table and self.query == other.query


class DataEngineRetrievalJob(RetrievalJob):
    """Retrieval job for DataEngineOfflineStore."""

    def __init__(self, evaluation_function: Callable):
        """Initialize a lazy historical retrieval job"""
        self.evaluation_function = evaluation_function

    def persist(self, storage: SavedDatasetStorage, allow_overwrite: bool = False):
        raise NotImplementedError

    @property
    def metadata(self) -> Optional[RetrievalMetadata]:
        """Returns metadata about the retrieval job."""
        raise NotImplementedError

    @property
    def full_feature_names(self):
        raise NotImplementedError

    @property
    def on_demand_feature_views(self):
        raise NotImplementedError

    def _to_df_internal(self) -> pd.DataFrame:
        df = self.evaluation_function()
        return df

    def _to_arrow_internal(self) -> pyarrow.Table:
        df = self.evaluation_function()
        return pyarrow.Table.from_pandas(df)


class DataEngineOfflineStore(OfflineStore):
    """Implementation of the offline store backed by the IBM Cloud Data Engine."""

    # pylint: disable=too-many-arguments
    @staticmethod
    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Union[pd.DataFrame, str],
        registry: Registry,
        project: str,
        full_feature_names: bool = False,
    ) -> RetrievalJob:
        raise NotImplementedError

    # pylint: disable=too-many-arguments
    @staticmethod
    def pull_latest_from_table_or_query(
        config: RepoConfig,
        data_source: DataSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        created_timestamp_column: Optional[str],
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        raise NotImplementedError
