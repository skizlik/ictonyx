#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Define the image name
IMAGE_NAME="ictonyx-gpu-docker"

# Get parent directory of your project (where other projects are)
HOST_PROJECTS_DIR=$(dirname "$(pwd)")
CONTAINER_PROJECTS_DIR="/home/appuser/projects"

# Get current user and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "--- Launching container for Ictonyx development ---"
echo "Running as user $USER_ID:$GROUP_ID to maintain proper file ownership"
echo "Mounting $HOST_PROJECTS_DIR to $CONTAINER_PROJECTS_DIR"
echo "To exit, stop the process by pressing Ctrl+C"

# Determine the command to run. Default to Jupyter Lab.
# If a custom command is provided as arguments, use that.
if [ -z "$@" ]; then
    # Default command is Jupyter Lab
    TARGET_CMD="su - appuser -c 'cd /home/appuser/projects && JUPYTER_CONFIG_DIR=/home/appuser/.jupyter JUPYTER_DATA_DIR=/home/appuser/.jupyter JUPYTER_RUNTIME_DIR=/home/appuser/.jupyter/runtime jupyter lab --ip=0.0.0.0 --port=8888 --no-browser'"
    echo "Access JupyterLab at http://localhost:8888"
else
    # Custom command provided, run it as appuser
    TARGET_CMD="su - appuser -c 'cd /home/appuser/projects && $@'"
    echo "Running custom command: $@"
fi

# Create user setup command to run inside container
# (Moved the pip install -e . here, for reasons explained below)
USER_SETUP_CMD="
# Create group if it doesn't exist
if ! getent group $GROUP_ID >/dev/null 2>&1; then
   groupadd -g $GROUP_ID appuser
fi

# Create user if it doesn't exist
if ! id -u appuser >/dev/null 2>&1; then
   useradd -u $USER_ID -g $GROUP_ID -d /home/appuser -m -s /bin/bash appuser
   mkdir -p /home/appuser/.jupyter/runtime
   mkdir -p /home/appuser/.cache/pip
   chown -R appuser:appuser /home/appuser
fi

# Change ownership of projects directory to the user
chown -R appuser:appuser /home/appuser/projects

# Install ictonyx library from the ictonyx subdirectory in editable mode
# This must happen AFTER appuser is created and ownership is set, 
# and before the main command (Jupyter or bash) is executed as that user.
echo 'Installing Ictonyx library in editable mode...'
cd /home/appuser/projects/ictonyx
pip install -e .

# Finally, execute the target command
$TARGET_CMD
"

# Run container with the setup command
docker run \
   --gpus all \
   -it \
   --rm \
   -p 8888:8888 \
   -v "$HOST_PROJECTS_DIR":"$CONTAINER_PROJECTS_DIR" \
   "$IMAGE_NAME" \
   bash -c "$USER_SETUP_CMD"
