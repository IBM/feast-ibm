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
