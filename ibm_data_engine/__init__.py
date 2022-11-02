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

"""Implementation of the offline store backed by the IBM Cloud Data Engine."""

__version__ = "0.25.0"

from ibm_data_engine.data_engine_offline_store import (
    DataEngineDataSource,
    DataEngineOfflineStore,
    DataEngineOfflineStoreConfig,
    DataEngineRetrievalJob,
    DataEngineSchemaError,
)
