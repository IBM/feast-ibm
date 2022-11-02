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

from datetime import timedelta, datetime
from unittest.mock import MagicMock

import pandas as pd
import testfixtures
from feast import Entity, FeatureView, Field, FeatureStore
from feast.infra.offline_stores import offline_utils
from feast.types import Float32, Int64
from ibmcloudsql import SQLQuery
from pandas._testing import assert_frame_equal

from ibm_data_engine import DataEngineDataSource, DataEngineRetrievalJob, data_engine_offline_store


def get_driver_feature_view():
    driver = Entity(name="driver", join_keys=["driver_id"])
    driver_stats_source = DataEngineDataSource(
        name="driver_hourly_stats_source",
        table="driver_stats",
        timestamp_field="event_timestamp",
    )
    driver_stats_fv = FeatureView(
        name="driver_hourly_stats",
        entities=[driver],
        ttl=timedelta(days=1),
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
            Field(name="avg_daily_trips", dtype=Int64),
        ],
        online=True,
        source=driver_stats_source,
        tags={"team": "driver_performance"},
    )
    return driver, driver_stats_source, driver_stats_fv


EXPECTED_QUERY = """
/*
 Compute a deterministic hash for the `left_table_query_string` that will be used throughout
 all the logic as the field to GROUP BY the data
*/
WITH entity_dataframe AS (
    SELECT *,
        event_timestamp AS entity_timestamp
            ,CAST(event_timestamp AS STRING) AS driver_hourly_stats__entity_row_unique_id
    FROM FEAST_TEMP_ENTITY_TABLE
),

driver_hourly_stats__entity_dataframe AS (
    SELECT
        entity_timestamp,
        driver_hourly_stats__entity_row_unique_id
    FROM entity_dataframe
    GROUP BY
        entity_timestamp,
        driver_hourly_stats__entity_row_unique_id
),

/*
 This query template performs the point-in-time correctness join for a single feature set table
 to the provided entity table.

 1. We first join the current feature_view to the entity dataframe that has been passed.
 This JOIN has the following logic:
    - For each row of the entity dataframe, only keep the rows where the `timestamp_field`
    is less than the one provided in the entity dataframe
    - If there a TTL for the current feature_view, also keep the rows where the `timestamp_field`
    is higher the the one provided minus the TTL
    - For each row, Join on the entity key and retrieve the `entity_row_unique_id` that has been
    computed previously

 The output of this CTE will contain all the necessary information and already filtered out most
 of the data that is not relevant.
*/

driver_hourly_stats__subquery AS (
    SELECT
        event_timestamp as event_timestamp,
            conv_rate as conv_rate,
            acc_rate as acc_rate,
            avg_daily_trips as avg_daily_trips
    FROM `driver_stats`
   WHERE event_timestamp <= to_timestamp('2021-04-12T16:40:26')
    AND event_timestamp >= to_timestamp('2021-04-11T08:12:10')
),

driver_hourly_stats__base AS (
    SELECT
        subquery.*,
        entity_dataframe.entity_timestamp,
        entity_dataframe.driver_hourly_stats__entity_row_unique_id
    FROM driver_hourly_stats__subquery AS subquery
    INNER JOIN driver_hourly_stats__entity_dataframe AS entity_dataframe
    ON TRUE
        AND subquery.event_timestamp <= entity_dataframe.entity_timestamp
        AND subquery.event_timestamp >= entity_dataframe.entity_timestamp -  interval 86400 second
),

/*
 2. If the `created_timestamp_column` has been set, we need to
 deduplicate the data first. This is done by calculating the
 `MAX(created_at_timestamp)` for each event_timestamp.
 We then join the data on the next CTE
*/


/*
 3. The data has been filtered during the first CTE "*__base"
 Thus we only need to compute the latest timestamp of each feature.
*/
driver_hourly_stats__latest AS (
    SELECT
        event_timestamp,
        driver_hourly_stats__entity_row_unique_id
    FROM
    (
        SELECT base.*,
            ROW_NUMBER() OVER(
                PARTITION BY base.driver_hourly_stats__entity_row_unique_id
                ORDER BY base.event_timestamp DESC
            ) AS row_number
        FROM driver_hourly_stats__base as base
    )
    WHERE row_number = 1
),

/*
 4. Once we know the latest value of each feature for a given timestamp,
 we can join again the data back to the original "base" dataset
*/
driver_hourly_stats__cleaned AS (
    SELECT base.*
    FROM driver_hourly_stats__base as base
    INNER JOIN driver_hourly_stats__latest as latest
    ON TRUE
        AND base.driver_hourly_stats__entity_row_unique_id = latest.driver_hourly_stats__entity_row_unique_id
        AND base.event_timestamp = latest.event_timestamp

)
/*
 Joins the outputs of multiple time travel joins to a single table.
 The entity_dataframe dataset being our source of truth here.
 */

SELECT driver_id, event_timestamp, label_driver_reported_satisfaction, conv_rate, acc_rate, avg_daily_trips
FROM entity_dataframe as entity_df

LEFT JOIN (
    SELECT
        driver_hourly_stats__entity_row_unique_id
            ,conv_rate
            ,acc_rate
            ,avg_daily_trips
    FROM driver_hourly_stats__cleaned
) as cleaned
ON TRUE
AND entity_df.driver_hourly_stats__entity_row_unique_id = cleaned.driver_hourly_stats__entity_row_unique_id
"""


class TestIntegrationDataEngineOfflineStore:
    def test_get_historical_features_integration(self, monkeypatch):
        store = FeatureStore(repo_path="./tests")
        driver, driver_stats_source, driver_stats_feature_view = get_driver_feature_view()
        store.registry.apply_data_source(driver_stats_source, "test_plugin")
        store.registry.apply_entity(driver, "test_plugin")
        store.registry.apply_feature_view(driver_stats_feature_view, "test_plugin")
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

        job = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
        )

        assert isinstance(job, DataEngineRetrievalJob)
        training_df = job.to_df()
        query = sql.run_sql.call_args_list[0].args[0]
        assert (
            testfixtures.compare(query, EXPECTED_QUERY, blanklines=False, trailing_whitespace=False)
            is None
        )
        assert_frame_equal(training_df, expected_df)
