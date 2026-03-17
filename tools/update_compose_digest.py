#!/usr/bin/env python3
"""
Update docker-compose image digests after a deploy-build run.

Examples:
    python tools/update_compose_digest.py validator
    python tools/update_compose_digest.py miner --environment staging
    python tools/update_compose_digest.py ihp
    python tools/update_compose_digest.py all --environment prod
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
IHP_IMAGE_TAGS = {
    "ihp-proxy": "backenddevelopersltd/infinitehash-proxy-server:v1-latest",
    "ihp-api": "backenddevelopersltd/infinitehash-proxy-api:v1-latest",
}


def run(cmd: list[str]) -> str:
    """Run command and return stdout or raise on failure."""
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def fetch_digest_buildx(image: str) -> str | None:
    """Return top-level image/index digest using docker buildx imagetools inspect."""
    try:
        output = run(["docker", "buildx", "imagetools", "inspect", image])
    except subprocess.CalledProcessError:
        return None

    match = re.search(r"(?m)^Digest:\s+(\S+)\s*$", output)
    if match:
        return match.group(1)
    return None


def fetch_digest(image: str) -> str:
    """Return manifest digest for the given image tag."""
    digest = fetch_digest_buildx(image)
    if digest:
        return digest

    try:
        output = run(["docker", "manifest", "inspect", "--verbose", image])
    except subprocess.CalledProcessError as exc:  # pragma: no cover - bubble context
        raise SystemExit(f"docker manifest inspect failed: {exc.stderr or exc}") from exc

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed
        raise SystemExit(f"Cannot parse docker manifest output: {exc}") from exc

    if isinstance(data, dict):
        digest = data.get("Descriptor", {}).get("digest")
        if digest:
            return digest
    elif isinstance(data, list):
        digests = [item.get("Descriptor", {}).get("digest") for item in data if isinstance(item, dict)]
        digests = [digest for digest in digests if digest]
        if len(digests) == 1:
            return digests[0]
        raise SystemExit(
            "Unable to determine a single digest from docker manifest output. "
            "Install/enable docker buildx or use an image with a single manifest."
        )

    raise SystemExit("Unable to find digest in docker manifest output.")


def update_compose(compose_path: Path, env: str, digest: str) -> int:
    """Replace image references in docker-compose with new digest."""
    text = compose_path.read_text()
    pattern = re.compile(rf"(ghcr\.io/backend-developers-ltd/infinitehash-subnet-{re.escape(env)})(?:[:@][^\s]+)")
    new_text, count = pattern.subn(rf"\1@{digest}", text)
    if count > 0:
        compose_path.write_text(new_text)
    return count


def update_compose_image(compose_path: Path, image_repository: str, digest: str) -> int:
    """Replace image references for a specific repository with a pinned digest."""
    text = compose_path.read_text()
    pattern = re.compile(rf"({re.escape(image_repository)})(?:[:@][^\s]+)")
    new_text, count = pattern.subn(rf"\1@{digest}", text)
    if count > 0:
        compose_path.write_text(new_text)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["validator", "miner", "ihp", "all"],
        help="which docker-compose file(s) to update",
    )
    parser.add_argument(
        "-e",
        "--environment",
        choices=["prod", "staging"],
        default="prod",
        help="target environment (determines image namespace, default: prod)",
    )
    default_path = Path(__file__).resolve().parents[1] / "envs" / "deployed" / "docker-compose.yml"
    parser.add_argument(
        "--compose-path",
        type=Path,
        default=default_path,
        help=f"path to validator docker-compose.yml (default: {default_path})",
    )
    miner_default_path = Path(__file__).resolve().parents[1] / "envs" / "miner" / "docker-compose.yml"
    parser.add_argument(
        "--miner-compose-path",
        type=Path,
        default=miner_default_path,
        help=f"path to Braiins miner docker-compose.yml (default: {miner_default_path})",
    )
    miner_ihp_default_path = Path(__file__).resolve().parents[1] / "envs" / "miner-ihp" / "docker-compose.yml"
    parser.add_argument(
        "--miner-ihp-compose-path",
        type=Path,
        default=miner_ihp_default_path,
        help=f"path to IHP miner docker-compose.yml (default: {miner_ihp_default_path})",
    )
    args = parser.parse_args()

    targets: list[tuple[str, Path, str, str]] = []
    if args.target in ("validator", "all"):
        image_tag = f"{REPOSITORY_PREFIX}-{args.environment}:{TAG}"
        targets.append(("validator", args.compose_path, image_tag, "app"))
    if args.target in ("miner", "all"):
        image_tag = f"{REPOSITORY_PREFIX}-{args.environment}:{TAG}"
        targets.append(("miner-braiins", args.miner_compose_path, image_tag, "app"))
        targets.append(("miner-ihp", args.miner_ihp_compose_path, image_tag, "app"))
    if args.target in ("ihp", "all"):
        for label, image_tag in IHP_IMAGE_TAGS.items():
            targets.append((label, args.miner_ihp_compose_path, image_tag, "direct"))

    total_replacements = 0
    digest_cache: dict[str, str] = {}
    for label, compose_path, image_tag, kind in targets:
        if not compose_path.exists():
            raise SystemExit(f"docker-compose file not found for {label}: {compose_path}")
        digest = digest_cache.get(image_tag)
        if digest is None:
            digest = fetch_digest(image_tag)
            digest_cache[image_tag] = digest
        if kind == "app":
            replacements = update_compose(compose_path, args.environment, digest)
        else:
            image_repository = image_tag.rsplit(":", 1)[0]
            replacements = update_compose_image(compose_path, image_repository, digest)
        if replacements == 0:
            print(f"No image references for {image_tag} found in {compose_path}")
        else:
            print(f"{label.capitalize()}: pinned {replacements} reference(s) to {image_tag}@{digest}")
        total_replacements += replacements

    if total_replacements == 0:
        raise SystemExit("No image references were updated.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:  # Allow non-zero exit with message
        if exc.code not in (None, 0):
            print(exc, file=sys.stderr)
        raise
