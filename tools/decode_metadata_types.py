#!/usr/bin/env python3
"""Extract SCALE type definitions using turbobt's type registry."""

import json
import sys
from pathlib import Path

from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from turbobt.substrate.client import load_type_registry_v15_types


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python decode_metadata_types.py <metadata.json> [output.md]")
        print("   or: uv run python decode_metadata_types.py <metadata.json> --type <type_id>")
        sys.exit(1)

    # Load metadata hex
    hex_data = json.load(open(sys.argv[1]))

    # Setup runtime config like turbobt does
    runtime_config = RuntimeConfigurationObject()
    runtime_config.update_type_registry(load_type_registry_preset("core"))
    runtime_config.update_type_registry_types(load_type_registry_v15_types())
    runtime_config.type_registry["types"]["metadataall"].type_mapping.append(["V15", "MetadataV15"])

    # Decode as Option<Vec<u8>> first (like turbobt does)
    option_decoder = runtime_config.create_scale_object("Option<Vec<u8>>", data=ScaleBytes(hex_data))
    option_decoder.decode()

    if not option_decoder.value:
        print("ERROR: Metadata is None")
        sys.exit(1)

    # option_decoder.value is hex string, convert to bytes
    inner_bytes = (
        bytes.fromhex(option_decoder.value.removeprefix("0x"))
        if isinstance(option_decoder.value, str)
        else option_decoder.value
    )

    # The Vec contains: magic (4 bytes "meta") + version (1 byte) + MetadataV15 data
    # Skip first 5 bytes
    if inner_bytes[:4] == b"meta":
        version_byte = inner_bytes[4]
        metadata_bytes = inner_bytes[5:]
        print(f"Metadata version from magic: V{version_byte}", file=sys.stderr)
    else:
        metadata_bytes = inner_bytes
        print("No magic found, treating as raw metadata", file=sys.stderr)

    # Decode as MetadataV15
    decoder = runtime_config.create_scale_object("MetadataV15", data=ScaleBytes(f"0x{metadata_bytes.hex()}"))
    decoder.decode()
    variant_data = decoder.value

    print("Metadata decoded successfully", file=sys.stderr)

    # Extract types from V15 structure
    types_list = variant_data["types"]["types"]
    types_by_id = {t["id"]: t for t in types_list}
    version = "V15"

    # Check if --type flag is used
    if len(sys.argv) > 2 and sys.argv[2] == "--type":
        if len(sys.argv) < 4:
            print("Error: --type requires a type ID")
            sys.exit(1)

        type_id = int(sys.argv[3])
        max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 5

        visited = set()

        def display_type_recursive(tid, indent=0, depth=0):
            if depth > max_depth or tid in visited:
                return [" " * indent + f"Type#{tid} (...)"]

            visited.add(tid)
            prefix = " " * indent
            lines = []

            if tid not in types_by_id:
                return [prefix + f"Type#{tid} <primitive or undefined>"]

            t = types_by_id[tid]
            tdef = t["type"]["def"]
            path = "::".join(t["type"]["path"]) or f"Type{tid}"

            if "primitive" in tdef:
                prim = tdef["primitive"]
                lines.append(prefix + f"Type#{tid} = {prim}")
            elif "composite" in tdef:
                lines.append(prefix + f"Type#{tid}: {path} {{")
                for f in tdef["composite"].get("fields", []):
                    fname = f.get("name", "unnamed")
                    ftype = f["type"]
                    ftypename = f.get("typeName", "")
                    type_comment = f" // {ftypename}" if ftypename else ""
                    lines.append(prefix + f"  {fname}: Type#{ftype}{type_comment}")
                    # Recursively expand field types
                    lines.extend(display_type_recursive(ftype, indent + 4, depth + 1))
                lines.append(prefix + "}")
            elif "variant" in tdef:
                lines.append(prefix + f"Type#{tid}: {path} (enum) {{")
                for v in tdef["variant"].get("variants", [])[:5]:  # Show first 5 variants
                    vfields = v.get("fields", [])
                    if vfields:
                        vtype = vfields[0]["type"]
                        lines.append(prefix + f"  {v['name']}(#{v['index']}): Type#{vtype}")
                        if depth < max_depth - 1:
                            lines.extend(display_type_recursive(vtype, indent + 4, depth + 1))
                    else:
                        lines.append(prefix + f"  {v['name']}(#{v['index']})")
                if len(tdef["variant"].get("variants", [])) > 5:
                    lines.append(prefix + f"  ... ({len(tdef['variant']['variants'])} total variants)")
                lines.append(prefix + "}")
            elif "sequence" in tdef:
                seq_type = tdef["sequence"]["type"]
                lines.append(prefix + f"Type#{tid}: Vec<Type#{seq_type}>")
                lines.extend(display_type_recursive(seq_type, indent + 2, depth + 1))
            elif "array" in tdef:
                arr_type = tdef["array"]["type"]
                arr_len = tdef["array"]["len"]
                lines.append(prefix + f"Type#{tid}: [Type#{arr_type}; {arr_len}]")
                lines.extend(display_type_recursive(arr_type, indent + 2, depth + 1))
            elif "compact" in tdef:
                comp_type = tdef["compact"]["type"]
                lines.append(prefix + f"Type#{tid}: Compact<Type#{comp_type}>")
                lines.extend(display_type_recursive(comp_type, indent + 2, depth + 1))
            else:
                lines.append(prefix + f"Type#{tid}: {path} <unknown def>")

            return lines

        result = "\n".join(
            [f"# Recursive Type Display for Type#{type_id}", "=" * 80, ""] + display_type_recursive(type_id)
        )
        print(result)
        return

    # Format output (existing code for full listing)
    lines = [f"# SCALE Types ({version})", f"Total: {len(types_list)}", "=" * 80, ""]

    for t in types_list:
        tid, tdef = t["id"], t["type"]["def"]
        path = "::".join(t["type"]["path"]) or f"Type{tid}"

        lines.append(f"\n## [{tid}] {path}")
        for doc in t["type"].get("docs", []):
            lines.append(f"/// {doc}")

        if "composite" in tdef:
            lines.append(f"{path} {{")
            for f in tdef["composite"].get("fields", []):
                fname = f.get("name", "unnamed")
                ftype = f["type"]
                ftypename = f.get("typeName", "")
                lines.append(f"  {fname}: Type#{ftype}  // {ftypename}" if ftypename else f"  {fname}: Type#{ftype}")
            lines.append("}")
        elif "variant" in tdef:
            lines.append(f"{path} (enum) {{")
            for v in tdef["variant"].get("variants", []):
                lines.append(f"  {v['name']}(#{v['index']})")
            lines.append("}")
        elif "sequence" in tdef:
            lines.append(f"{path} = Vec<Type#{tdef['sequence']['type']}>")
        elif "array" in tdef:
            lines.append(f"{path} = [Type#{tdef['array']['type']}; {tdef['array']['len']}]")

        lines.append("-" * 80)

    result = "\n".join(lines)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(result)
        print(f"Wrote to {sys.argv[2]}")
    else:
        print(result)


if __name__ == "__main__":
    main()
