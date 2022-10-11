"""Implementation of the offline store backed by the IBM Cloud Data Engine."""

import json
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import pyarrow
from feast.data_source import DataSource
from feast.errors import DataSourceNoNameException, DataSourceNotFoundException
from feast.feature_view import FeatureView
from feast.infra.offline_stores.offline_store import OfflineStore, RetrievalJob, RetrievalMetadata
from feast.infra.registry.registry import Registry

# pylint: disable=no-name-in-module
from feast.protos.feast.core.DataSource_pb2 import DataSource as DataSourceProto
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.saved_dataset import SavedDatasetStorage
from feast.value_type import ValueType
from ibmcloudsql import SQLQuery
from typing_extensions import Literal

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def _sql_builder(config: RepoConfig) -> SQLQuery:
    assert isinstance(config.offline_store, DataEngineOfflineStoreConfig)
    return SQLQuery(
        api_key=config.offline_store.api_key,
        instance_crn=config.offline_store.instance_crn,
        target_cos_url=config.offline_store.target_cos_url,
    )


def _join(fields: List[str], alias: Optional[str] = None) -> str:
    if not alias:
        return ", ".join(fields)
    return ", ".join(f"{alias}.{field}" for field in fields)


def _where(fields: List[str], aliases: Tuple[str, str]) -> str:
    return ", ".join(f"{aliases[0]}.{field} = {aliases[1]}.{field}" for field in fields)


class DataEngineOfflineStoreConfig(FeastConfigBaseModel):
    """Offline store config for IBM Cloud Data Engine."""

    type: Literal[
        "ibm_data_engine.DataEngineOfflineStore"
    ] = "ibm_data_engine.DataEngineOfflineStore"

    api_key: str
    instance_crn: str
    target_cos_url: Optional[str] = None


class DataEngineSchemaError(Exception):
    """Raised when retrieving schema fails."""

    def __init__(self, table):
        super().__init__(f"Error retrieving schema from table: {table}")


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
        cos_type: str = "parquet",
        description: Optional[str] = "",
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = "",
    ):
        """Create a DataEngineDataSource from an existing table or query.
        Args:
            name (optional): Name for the source. Defaults to the table if not specified.
            timestamp_field (optional): Event timestamp field used for point in time
                joins of feature values.
            table (optional): The Data Engine table where features can be found;
                can also be COS address.
            created_timestamp_column (optional): Timestamp column when row was created, used for
                deduplicating rows.
            field_mapping (optional): A dictionary mapping of column names in this data source to
                feature names in a feature table or view. Only used for feature columns, not
                entities or timestamp columns.
            query (optional): SQL query to execute to generate data for this data source.
            cos_type (optional): If the provided table value is a COS URL, then this defines the
                type of the underlying data. Valid values are: json, csv, or parquet (default).
            description (optional): A human-readable description.
            tags (optional): A dictionary of key-value pairs to store arbitrary metadata.
            owner (optional): The owner of the bigquery source, typically the email of the primary
                maintainer.
        """
        if table is None and query is None:
            raise ValueError('No "table" or "query" argument provided.')

        self.table = table
        self.query = query
        self.cos_type = cos_type

        assert cos_type in {"json", "csv", "parquet"}, "cos_type must be one of: json, csv, parquet"

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
        cos_type = json.loads(custom_source_options)["cos_type"]
        return DataEngineDataSource(
            name=data_source.name,
            field_mapping=dict(data_source.field_mapping),
            table=table,
            timestamp_field=data_source.timestamp_field,
            created_timestamp_column=data_source.created_timestamp_column,
            query=query,
            cos_type=cos_type,
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
        )

    def to_proto(self) -> DataSourceProto:
        config_json = json.dumps(
            {"table": self.table, "query": self.query, "cos_type": self.cos_type}
        )
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
        builder = _sql_builder(config)

        if self.table and self.table.startswith("cos://"):
            try:
                builder.get_cos_summary(self.table)
                return
            except Exception as err:
                raise DataSourceNotFoundException(self.table) from err

        if self.table:
            if builder.get_schema_table(self.table) is None:
                raise DataSourceNotFoundException(self.table)
            return

        try:
            builder.execute_sql(f"SELECT * FROM ({self.query}) LIMIT 1")
        except Exception as err:
            raise DataSourceNotFoundException(self.query) from err

    def get_table_query_string(self) -> str:
        """Returns a string that can directly be used to reference this table in SQL"""
        if self.table:
            return f"`{self.table}`"
        return f"({self.query})"

    def get_table_column_names_and_types(self, config: RepoConfig) -> Iterable[Tuple[str, str]]:
        """
        Returns the list of column names and raw column types.

        Args:
            config: Configuration object used to configure a feature store.
        """
        if self.table and self.table.startswith("cos://"):
            try:
                schema = _sql_builder(config).get_schema_data(self.table, type=self.cos_type)
            except Exception as err:
                raise DataEngineSchemaError(self.table) from err
        elif self.table:
            schema = _sql_builder(config).get_schema_table(self.table)
        else:
            # Technically speaking we could implement this by executing the query and
            # reading the columns and dtypes from the data frame. The issue here is that
            # the Pandas types are not the same as Data Engine types.
            # We can address this at a later point, but this is not a priority, as we'll
            # most likely use tables and not queries.
            raise ValueError("Cannot retrieve schema from query")
        if schema is None:
            raise DataEngineSchemaError(self.table)
        return list(zip(schema.col_name, schema.data_type))

    @staticmethod
    def source_datatype_to_feast_value_type() -> Callable[[str], ValueType]:
        """
        Returns the callable method that returns Feast type given the raw column type.
        """
        raw_to_feast = {
            "binary": ValueType.BYTES,
            "boolean": ValueType.BOOL,
            "tinyint": ValueType.INT32,
            "smallint": ValueType.INT32,
            "int": ValueType.INT32,
            "integer": ValueType.INT32,
            "bigint": ValueType.INT64,
            "long": ValueType.INT64,
            "float": ValueType.FLOAT,
            "double": ValueType.DOUBLE,
            "decimal": ValueType.DOUBLE,
            "string": ValueType.STRING,
            "timestamp": ValueType.UNIX_TIMESTAMP,
            "array<binary>": ValueType.BYTES_LIST,
            "array<boolean>": ValueType.BOOL_LIST,
            "array<tinyint>": ValueType.INT32_LIST,
            "array<smallint>": ValueType.INT32_LIST,
            "array<int>": ValueType.INT32_LIST,
            "array<integer>": ValueType.INT32_LIST,
            "array<bigint>": ValueType.INT64_LIST,
            "array<long>": ValueType.INT64_LIST,
            "array<float>": ValueType.FLOAT_LIST,
            "array<double>": ValueType.DOUBLE_LIST,
            "array<decimal>": ValueType.DOUBLE_LIST,
            "array<string>": ValueType.STRING_LIST,
            "array<timestamp>": ValueType.UNIX_TIMESTAMP_LIST,
        }
        return lambda typ: raw_to_feast.get(typ, ValueType.UNKNOWN)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, DataEngineDataSource):
            raise TypeError("Comparisons should only involve DataEngineDataSource class objects.")
        return (
            super().__eq__(other)
            and self.table == other.table
            and self.query == other.query
            and self.cos_type == self.cos_type
        )

    def sql_from(self, sql: SQLQuery, alias: Optional[str] = None) -> SQLQuery:
        """Adds FROM clause to the given SQL builder.

        Note that alias is ignored for query sources.
        """
        if self.table and self.table.startswith("cos://"):
            return sql.from_cos_(self.table, format_type=self.cos_type, alias=alias)
        if self.table:
            return sql.from_table_(self.table, alias=alias)
        return sql.from_view_(self.query)


class DataEngineRetrievalJob(RetrievalJob):
    """Retrieval job for DataEngineOfflineStore."""

    def __init__(self, evaluation_function: Callable, retrieval_metadata: RetrievalMetadata):
        """Initialize a lazy historical retrieval job"""
        self.evaluation_function = evaluation_function
        self._retrieval_metadata = retrieval_metadata

    def persist(
        self, storage: SavedDatasetStorage, allow_overwrite: bool = False
    ):  # pragma: no cover
        raise NotImplementedError

    @property
    def metadata(self) -> Optional[RetrievalMetadata]:
        """Returns metadata about the retrieval job."""
        return self._retrieval_metadata

    @property
    def full_feature_names(self):  # pragma: no cover
        return False

    @property
    def on_demand_feature_views(self):
        return None

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
        assert isinstance(config.offline_store, DataEngineOfflineStoreConfig)
        assert isinstance(data_source, DataEngineDataSource)

        def inner():
            """Constructs the SQL string for the subquery that calculates max timestamp per key."""
            keys = _join(join_key_columns)
            sql = _sql_builder(config).select_(f"{keys}, max({timestamp_field}) as timestamp")
            data_source.sql_from(sql)
            start = cast_timestamp(start_date)
            end = cast_timestamp(end_date)
            sql.where_(f"{timestamp_field} BETWEEN {start} AND {end}")
            return sql.group_by_(keys).get_sql()

        def retrieve_df():
            fields = _join(
                join_key_columns + feature_name_columns + [timestamp_field], alias="de_a"
            )
            where_clause = _where(join_key_columns, aliases=("de_a", "de_b"))
            sql_query = (
                f"SELECT {fields} FROM {data_source.get_table_query_string()} as de_a "
                f"JOIN ({inner()}) as de_b WHERE {where_clause} AND de_a.timestamp = de_b.timestamp"
            )
            return _sql_builder(config).run_sql(sql_query)

        return DataEngineRetrievalJob(
            retrieve_df,
            RetrievalMetadata(
                feature_name_columns,
                join_key_columns,
                min_event_timestamp=start_date,
                max_event_timestamp=end_date,
            ),
        )

    # pylint: disable=too-many-arguments
    @staticmethod
    def pull_all_from_table_or_query(
        config: RepoConfig,
        data_source: DataSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        assert isinstance(config.offline_store, DataEngineOfflineStoreConfig)
        assert isinstance(data_source, DataEngineDataSource)

        def retrieve_df():
            sql = _sql_builder(config)
            sql.select_(", ".join(join_key_columns + feature_name_columns + [timestamp_field]))
            sql = data_source.sql_from(sql)
            start = cast_timestamp(start_date)
            end = cast_timestamp(end_date)
            sql = sql.where_(f"{timestamp_field} BETWEEN {start} AND {end}")
            return sql.run(get_result=True).data

        return DataEngineRetrievalJob(
            retrieve_df,
            RetrievalMetadata(
                feature_name_columns,
                join_key_columns,
                min_event_timestamp=start_date,
                max_event_timestamp=end_date,
            ),
        )


def cast_timestamp(timestamp: datetime) -> str:
    """Formats timestamp for SQL use.

    >>> cast_timestamp(datetime(year=2022, month=10, day=11))
    "CAST('2022-10-11 00:00:00.000000' AS TIMESTAMP)"
    """
    tstamp = timestamp.strftime(TIMESTAMP_FORMAT)
    return f"CAST('{tstamp}' AS TIMESTAMP)"
