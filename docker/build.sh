#!/usr/bin/env bash
set -euo pipefail
docker_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(dirname -- "$docker_dir")
# The context is the workspace root; docker/Dockerfile.dockerignore requires BuildKit.
DOCKER_BUILDKIT=1 docker build --build-arg "BUILD_JOBS=${BUILD_JOBS:-4}" \
    -f "$docker_dir/Dockerfile" \
    -t "${RAGNAROK_IMAGE:-ragnarok:humble-cu121}" "$@" "$workspace_dir"
