#!/bin/bash

# set timeout duration
timeout=15

# Source the .bashrc file to load environment variables
source "$HOME/.bashrc"

# Stop all running containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

echo "All Docker containers have been stopped and deleted, ready to restart services"

# Change directory to the VISUALIZER_DIR_PATH
cd "$VISUALIZER_DIR_PATH"

# Start viewer after launching orthanc server
echo "Starting dev:orthanc..."
yarn run dev:orthanc --verbose > orthanc_run.log &

echo "Waiting for $timeout seconds to allow dev:orthanc to initialize properly..."
# Wait for another a bit
sleep $timeout

# Start the orthanc service in the background and detach it
nohup yarn orthanc:up > orthanc_up.log 2>&1 &

echo "Starting orthanc:up, waiting for $timeout seconds to initialize..."
# Wait for a bit
sleep $timeout

# Execute the sync and infer script
echo "Executing the Sync and infer script..."
"$PYTHON_PATH" "$SYNC_AND_INFER_SCRIPT_PATH" &

echo "Server services are up and ready."
