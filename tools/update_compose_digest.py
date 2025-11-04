#!/usr/bin/env python3
"""
Update docker-compose image digests after a deploy-build run.

Example:
    python tools/update_compose_digest.py prod
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_PREFIX = "ghcr.io/backend-developers-ltd/infinitehash-subnet"
TAG = "v0-latest"


def run(cmd: list[str]) -> str:
    """Run command and return stdout or raise on failure."""
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def fetch_digest(image: str) -> str:
    """Return manifest digest for the given image tag."""
    try:
        output = run(["docker", "manifest", "inspect", "--verbose", image])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - bubble context
        raise SystemExit(f"docker manifest inspect failed: {exc.stderr or exc}") from exc

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed
        raise SystemExit(f"Cannot parse docker manifest output: {exc}") from exc

    digest = data.get("Descriptor", {}).get("digest")
    if not digest:
        raise SystemExit("Unable to find digest in docker manifest output.")
    return digest


def update_compose(compose_path: Path, env: str, digest: str) -> int:
    """Replace image references in docker-compose with new digest."""
    text = compose_path.read_text()
    pattern = re.compile(rf"(ghcr\.io/backend-developers-ltd/infinitehash-subnet-{re.escape(env)})(?:[:@][^\s]+)")
    new_text, count = pattern.subn(rf"\1@{digest}", text)
    if count > 0:
        compose_path.write_text(new_text)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "environment",
        nargs="?",
        default="prod",
        choices=["prod", "staging"],
        help="target environment (determines image namespace)",
    )
    default_path = Path(__file__).resolve().parents[1] / "envs" / "deployed" / "docker-compose.yml"
    parser.add_argument(
        "--compose-path",
        type=Path,
        default=default_path,
        help=f"path to docker-compose.yml to update (default: {default_path})",
    )
    args = parser.parse_args()

    image_tag = f"{REPOSITORY_PREFIX}-{args.environment}:{TAG}"
    digest = fetch_digest(image_tag)

    compose_path = args.compose_path
    if not compose_path.exists():
        raise SystemExit(f"docker-compose file not found: {compose_path}")

    replacements = update_compose(compose_path, args.environment, digest)
    if replacements == 0:
        raise SystemExit(f"No image references for environment '{args.environment}' found in {compose_path}")

    print(f"Pinned {replacements} reference(s) to {image_tag}@{digest}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:  # Allow non-zero exit with message
        if exc.code not in (None, 0):
            print(exc, file=sys.stderr)
        raise
