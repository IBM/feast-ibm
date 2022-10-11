Implementation of the offline store backed by the IBM Cloud Data Engine.

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

You use it with Feast by defining your offline store and data sources.
The instructions below illustrate how it can be used in
[feast-config](https://github.ibm.com/w3-Search/feast-config).

## Define dependency

This library is currently not published in Artifactory; you will have to
point to the repository directly. The easiest way to do it is to clone
the repository, and define the dependency as a path in `feast-config`.

```toml
ibm-data-engine = { path = "/path/to/ibm-data-engine" }
```

After running `poetry update`, you should be able to use the IBM Cloud
Data Engine offline store.

## Define data source

You can modify the `src/ltr/features.py` file to use the new data
source. Below is the minimal example of the file:

```python
from ibm_data_engine import DataEngineDataSource
from w3_search_feast_config import letor

document_features = DataEngineDataSource(
    name="document_features",
    table="document_features",
    timestamp_field="timestamp",
)
docid = letor.Entities.docid
document_feature_view = letor.document_feature_view(document_features)
```

## Define offline store

Then, `feature_store.yaml` must configure the offline store.

```yaml
project: w3search_ltr
entity_key_serialization_version: 2
# Define some temporary registry for testing purposes
registry:
    registry_type: local
    path: /tmp/registry.db
provider: local
online_store:
    type: redis
    connection_string: 5aebc01f-35ac-4a10-87d7-cb32c6f94a22.bn2a0fgd0tu045vmv2i0.databases.appdomain.cloud:32615,username=ibm_cloud_1c205b7f_faf7_4c93_9a34_e71cc60fa741,password=${REDIS_PASSWORD},ssl=true,ssl_ca_certs=${REDIS_CERT_PATH},db=2
offline_store:
    type: "ibm_data_engine.DataEngineOfflineStore"
    api_key: "${API_KEY}"
    instance_crn: "${CRN}"
    target_cos_url: cos://us-south/w3-search-reports-new/tmp/sql
```

Notice that you must define the environment variables:
 * `REDIS_PASSWORD`
 * `REDIS_CERT_PATH`
 * `API_KEY`
 * `CRN`

## Apply

To apply the definitions to the registry, run:

```
poetry run feast -c src/ltr apply
```

## Materialize

To materialize to Redis, run:

```
poetry run feast -c src/ltr materialize \
    '2022-08-12 09:09:27.108650' \ # start date
    '2022-08-12 09:09:27.108652' \ # end date
    -v document_features
```
