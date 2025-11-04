"""SCALE codec type registry extensions from turbobt for MetadataV15 support."""


def load_type_registry_v15_types():
    return {
        "genericruntimeapimethodmetadata": {
            "type": "struct",
        },
        "genericruntimeapimetadata": {
            "type": "struct",
        },
        "RuntimeApiMethodParamMetadataV15": {
            "type": "struct",
            "type_mapping": [["name", "Text"], ["type", "SiLookupTypeId"]],
        },
        "RuntimeApiMethodMetadataV15": {
            "type": "struct",
            "base_class": "GenericRuntimeApiMethodMetadata",
            "type_mapping": [
                ["name", "Text"],
                ["inputs", "Vec<RuntimeApiMethodParamMetadataV15>"],
                ["output", "SiLookupTypeId"],
                ["docs", "Vec<Text>"],
            ],
        },
        "RuntimeApiMetadataV15": {
            "type": "struct",
            "base_class": "GenericRuntimeApiMetadata",
            "type_mapping": [
                ["name", "Text"],
                ["methods", "Vec<RuntimeApiMethodMetadataV15>"],
                ["docs", "Vec<Text>"],
            ],
        },
        "MetadataV15": {
            "type": "struct",
            "type_mapping": [
                ["types", "PortableRegistry"],
                ["pallets", "Vec<PalletMetadataV15>"],
                ["extrinsic", "ExtrinsicMetadataV15"],
                ["runtime_type", "SiLookupTypeId"],
                ["apis", "Vec<RuntimeApiMetadataV15>"],
                ["outer_enums", "OuterEnums15"],
                ["custom", "Vec<CustomMetadata15>"],
            ],
        },
        "PalletMetadataV15": {
            "type": "struct",
            "base_class": "ScaleInfoPalletMetadata",
            "type_mapping": [
                ["name", "Text"],
                ["storage", "Option<StorageMetadataV14>"],
                ["calls", "Option<PalletCallMetadataV14>"],
                ["event", "Option<PalletEventMetadataV14>"],
                ["constants", "Vec<PalletConstantMetadataV14>"],
                ["error", "Option<PalletErrorMetadataV14>"],
                ["index", "u8"],
                ["docs", "Vec<Text>"],
            ],
        },
        "ExtrinsicMetadataV15": {
            "type": "struct",
            "type_mapping": [
                ["version", "u8"],
                ["address_type", "SiLookupTypeId"],
                ["call_type", "SiLookupTypeId"],
                ["signature_type", "SiLookupTypeId"],
                ["extra_type", "SiLookupTypeId"],
                ["signed_extensions", "Vec<SignedExtensionMetadataV14>"],
            ],
        },
        "OuterEnums15": {
            "type": "struct",
            "type_mapping": [
                ["call_type", "SiLookupTypeId"],
                ["event_type", "SiLookupTypeId"],
                ["error_type", "SiLookupTypeId"],
            ],
        },
        "CustomMetadata15": "BTreeMap<Text, CustomValueMetadata15>",
        "CustomValueMetadata15": "Bytes",
    }
