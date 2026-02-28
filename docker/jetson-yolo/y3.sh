#!/bin/bash
set -e
echo "launching yolo container"

IMG="my_saved_yolo:updated"
NAME="yolo_container"

COMMON_ARGS=(
  --runtime nvidia
  --gpus all
  --network host
  --ipc host
  --privileged
  -v /dev:/dev
  -v "$HOME/yolo_workspace:/workspace"
  -e SHOW_WINDOW=0
)

# Only add X11 when a display actually exists
if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local:docker
  COMMON_ARGS+=(
    -e DISPLAY="$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix
  )
fi

if sudo docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "starting existing container: $NAME"
  sudo docker start -ai "$NAME"
else
  echo "creating new persistent container: $NAME"
  sudo docker run -it --name "$NAME" "${COMMON_ARGS[@]}" "$IMG"
fi
