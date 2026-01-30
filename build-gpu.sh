#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Define the image name
IMAGE_NAME="ictonyx-gpu-docker"

echo "--- Building Docker image for Ictonyx development ---"
echo "This will create a container that runs as your user ID to prevent root file ownership"

docker build -t $IMAGE_NAME .

echo "--- Image '$IMAGE_NAME' built successfully ---"
echo "Run ./run-gpu.sh to start the container with GPU support"
