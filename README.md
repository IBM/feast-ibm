[Feast](https://feast.dev/) plugin for IBM Cloud.  
This plugin implements Feast's offline store backed using [IBM Cloud Data Engine](https://www.ibm.com/cloud/data-engine) and [IBM Cloud Object Storage](https://www.ibm.com/cloud/object-storage)

# Installation

Project dependencies can be installed in a dedicated virtual environment
by running the following command:

```bash
poetry install
```

# Testing and Linting

```bash
poetry run pytest tests/
poetry run pylint ibm_data_engine
```

# Test with Feast

You use it with [Feast](https://feast.dev/) by defining your offline store and data sources.
The instructions below illustrate how it can be used in
[feast-ibm-quickstart](https://github.com/IBM/feast-ibm-quickstart).

## Define dependency

This library is currently not published in [PyPI](https://pypi.org/); you will have to
point to the repository directly. The easiest way to do it is to clone
the repository, and define the dependency as a path in `feast-ibm-quickstart`.

```toml
ibm-data-engine = { path = "/path/to/ibm-data-engine" }
```

After running `poetry update`, you should be able to use the IBM Cloud
Data Engine offline store.

## Define data source

You can modify the `src/feature_repo/example_repo.py` file to use the new data
source. Below is the minimal example of the file:

```python
from ibm_data_engine import DataEngineDataSource
driver_stats_source = DataEngineDataSource(
    name="driver_hourly_stats_source",
    table="driver_stats_demo",
    timestamp_field="event_timestamp",
)
```

## Define offline store

Then, `feature_repo/feature_store.yaml` must configure the offline store.

```yaml
project: test_plugin
entity_key_serialization_version: 2
registry: data/registry.db
provider: local
online_store:
    type: redis
    connection_string: ${REDIS_HOST}:${REDIS_PORT},username=${REDIS_USERNAME},password=${REDIS_PASSWORD},ssl=true,ssl_ca_certs=${REDIS_CERT_PATH},db=0

offline_store:
    type: ibm_data_engine.DataEngineOfflineStore
    api_key: ${DATA_ENGINE_API_KEY}
    instance_crn: ${DATA_ENGINE_INSTANCE_CRN}
    target_cos_url: ${IBM_CLOUD_OBJECT_STORE_URL}
```

Notice that you must define the environment variables:
 * `IBM_CLOUD_OBJECT_STORE_URL`
 * `REDIS_HOST`
 * `REDIS_PORT`
 * `REDIS_PASSWORD`
 * `REDIS_CERT_PATH`
 * `DATA_ENGINE_API_KEY`
 * `DATA_ENGINE_INSTANCE_CRN`

## Apply

To apply the definitions to the registry, run:

```
poetry run feast -c ./feature_repo apply
```

## Training

Run training by retrieving historical feature information from feature store
```
poetry run python training.py
```
## Materialize

To materialize to Redis, run:

```
poetry run feast -c ./ materialize '<START_TIMESTAMP>'  '<END_TIMESTAMP>'
```
## Inference

```
poetry run python inference.py
```
