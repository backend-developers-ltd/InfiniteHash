from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

_MODULE_DIR = os.path.dirname(__file__)
_METADATA_PATH = os.path.join(_MODULE_DIR, "metadata_version_15.json")
_RUNTIME_VERSION_PATH = os.path.join(_MODULE_DIR, "runtime_version_318.json")

with open(_METADATA_PATH) as _metadata_file:
    METADATA_V15_HEX = json.load(_metadata_file)

with open(_RUNTIME_VERSION_PATH) as _runtime_version_file:
    RUNTIME_VERSION = json.load(_runtime_version_file)


@lru_cache(maxsize=1)
def runtime_configuration() -> tuple[RuntimeConfiguration, Any]:
    """Load RuntimeConfiguration with portable registry from bundled metadata."""
    runtime_config = RuntimeConfiguration()
    runtime_config.update_type_registry(load_type_registry_preset("core"))

    from . import _scalecodec_turbobt

    runtime_config.update_type_registry_types(_scalecodec_turbobt.load_type_registry_v15_types())
    runtime_config.type_registry["types"]["metadataall"].type_mapping.append(["V15", "MetadataV15"])

    data_bytes = bytes.fromhex(METADATA_V15_HEX.removeprefix("0x"))
    option_decoder = runtime_config.create_scale_object("Option<Vec<u8>>", ScaleBytes(data_bytes))
    option_decoder.decode()
    inner_data = option_decoder.value

    metadata = None
    if inner_data:
        if isinstance(inner_data, str):
            inner_bytes = bytes.fromhex(inner_data.removeprefix("0x"))
        else:
            inner_bytes = inner_data

        metadata_decoder = runtime_config.create_scale_object("MetadataVersioned", ScaleBytes(inner_bytes))
        metadata_decoder.decode()
        metadata = metadata_decoder
        runtime_config.add_portable_registry(metadata)

    return runtime_config, metadata
