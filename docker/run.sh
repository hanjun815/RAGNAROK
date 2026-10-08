#!/usr/bin/env bash
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -lt 1 || $1 == --help ]]; then
    echo "Usage: $0 /path/to/RAGNAROK_dataset [command ...]"
    echo "Mounts the workspace and dataset root; default command: bash."
    exit 0
fi
dataset_dir=$(realpath -e -- "$1")
shift
[[ -d $dataset_dir ]] || { echo "Dataset root must be a directory." >&2; exit 1; }
container_name=${RAGNAROK_CONTAINER:-ragnarok}
if docker container inspect "$container_name" >/dev/null 2>&1; then
    echo "Container $container_name already exists. Use: docker exec -it $container_name bash" >&2
    exit 1
fi
options=(--rm --init --name "$container_name" --gpus all --network host --ipc host
    --workdir /root/code/RAGNAROK
    --mount "type=bind,src=$workspace_dir,dst=/root/code/RAGNAROK"
    --mount "type=bind,src=$dataset_dir,dst=/root/code/RAGNAROK_dataset,readonly"
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}")
[[ -t 0 && -t 1 ]] && options+=(-it)
xauth_file=
cleanup() { [[ -z $xauth_file ]] || rm -f -- "$xauth_file"; }
trap cleanup EXIT
if [[ -n ${DISPLAY:-} && -d /tmp/.X11-unix ]]; then
    options+=(--env "DISPLAY=$DISPLAY" --env QT_X11_NO_MITSHM=1
        --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly)
    if command -v xauth >/dev/null 2>&1; then
        xauth_file=$(mktemp /tmp/ragnarok-xauth.XXXXXX)
        xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
        options+=(--env XAUTHORITY=/tmp/ragnarok.xauth
            --mount "type=bind,src=$xauth_file,dst=/tmp/ragnarok.xauth,readonly")
    fi
fi
[[ $# -gt 0 ]] || set -- bash
docker run "${options[@]}" "${RAGNAROK_IMAGE:-ragnarok:humble-cu121}" "$@"
