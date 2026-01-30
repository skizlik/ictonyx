#!/bin/bash
# GPU testing script for Ictonyx Docker container

IMAGE_NAME="ictonyx-gpu-docker"
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Get parent directory (projects directory)
HOST_PROJECTS_DIR=$(dirname "$(pwd)")
CONTAINER_PROJECTS_DIR="/home/appuser/projects"

echo "=== Testing GPU Access in Container ==="
echo "Testing with projects directory: $HOST_PROJECTS_DIR"

# Test 1: Basic GPU visibility
echo "1. Testing basic GPU access..."
docker run --rm --gpus all $IMAGE_NAME nvidia-smi

echo -e "\n2. Testing TensorFlow GPU detection..."
docker run --rm --gpus all -v "$HOST_PROJECTS_DIR":"$CONTAINER_PROJECTS_DIR" $IMAGE_NAME bash -c "
# Create user for consistent testing
groupadd -g $GROUP_ID appuser 2>/dev/null || true
useradd -u $USER_ID -g $GROUP_ID -d /home/appuser -m -s /bin/bash appuser 2>/dev/null || true
chown -R appuser:appuser /home/appuser

# Test as the user
su - appuser -c 'cd /home/appuser/projects && python3 -c \"
import tensorflow as tf
print(\\\"TensorFlow version:\\\", tf.__version__)
print(\\\"Built with CUDA:\\\", tf.test.is_built_with_cuda())

# List physical devices
physical_devices = tf.config.list_physical_devices()
print(\\\"All physical devices:\\\", physical_devices)

# List GPU devices specifically
gpu_devices = tf.config.list_physical_devices(\\\"GPU\\\")
print(\\\"GPU devices found:\\\", len(gpu_devices))
for i, gpu in enumerate(gpu_devices):
    print(f\\\"  GPU {i}: {gpu}\\\")

# Test GPU availability
print(\\\"GPU available for TensorFlow:\\\", len(gpu_devices) > 0)

# Test basic tensor operation on GPU if available
if gpu_devices:
    print(\\\"Testing tensor operation on GPU...\\\")
    try:
        with tf.device(\\\"/GPU:0\\\"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print(\\\"GPU tensor operation successful!\\\")
            print(\\\"Result shape:\\\", c.shape)
    except Exception as e:
        print(\\\"GPU tensor operation failed:\\\", e)
else:
    print(\\\"No GPU available for tensor operations\\\")
\"'
"

echo -e "\n3. Testing Ictonyx installation..."
docker run --rm --gpus all -v "$HOST_PROJECTS_DIR":"$CONTAINER_PROJECTS_DIR" $IMAGE_NAME bash -c "
# Create user
groupadd -g $GROUP_ID appuser 2>/dev/null || true
useradd -u $USER_ID -g $GROUP_ID -d /home/appuser -m -s /bin/bash appuser 2>/dev/null || true
chown -R appuser:appuser /home/appuser

# Test Ictonyx installation
echo 'Testing Ictonyx library installation...'
su - appuser -c 'export PATH=/home/appuser/.local/bin:\$PATH && cd /home/appuser/projects/ictonyx && pip install --user -e . --no-deps && python3 -c \"import ictonyx; print(\\\"Ictonyx imported successfully!\\\")\"'
"

echo -e "\n4. File ownership test..."
echo "Creating a test file to verify ownership..."
docker run --rm --gpus all -v "$HOST_PROJECTS_DIR":"$CONTAINER_PROJECTS_DIR" $IMAGE_NAME bash -c "
# Create user
groupadd -g $GROUP_ID appuser 2>/dev/null || true
useradd -u $USER_ID -g $GROUP_ID -d /home/appuser -m -s /bin/bash appuser 2>/dev/null || true
chown -R appuser:appuser /home/appuser

# Test file creation as user
su - appuser -c 'cd /home/appuser/projects && echo \"test file\" > docker_test_file.txt'
"

if [ -f "$HOST_PROJECTS_DIR/docker_test_file.txt" ]; then
    FILE_OWNER=$(stat -c '%U' "$HOST_PROJECTS_DIR/docker_test_file.txt")
    echo "Test file created with owner: $FILE_OWNER"
    if [ "$FILE_OWNER" = "$(whoami)" ]; then
        echo "✓ File ownership test PASSED"
    else
        echo "✗ File ownership test FAILED - file owned by $FILE_OWNER instead of $(whoami)"
    fi
    rm -f "$HOST_PROJECTS_DIR/docker_test_file.txt"
else
    echo "✗ File creation test FAILED"
fi

echo -e "\n=== Testing Complete ==="
echo "If all tests passed, run ./run-gpu.sh to start JupyterLab"
