#!/bin/bash

# set timeout duration
timeout=15

# Source the .bashrc file to load environment variables
source "$HOME/.bashrc"

# Stop all running containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

# Remove all images
docker rmi $(docker images -q) -f

echo "All Docker containers, images have been stopped and deleted, ready to restart services"

# Change directory to the VISUALIZER_DIR_PATH
cd "$VISUALIZER_DIR_PATH"

# Make sure all to be used ports are available
# close all processes in port 80
sudo lsof -i :80 | grep LISTEN | awk '{print $2}' | xargs -r sudo kill -9
# close all processes in port 3000
sudo lsof -i :3000 | grep LISTEN | awk '{print $2}' | xargs -r sudo kill -9
# close all processes in port 5000
sudo lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs -r sudo kill -9
# close all processes in port 8042
sudo lsof -i :8042 | grep LISTEN | awk '{print $2}' | xargs -r sudo kill -9
# close all processes in port 9990
sudo lsof -i :9990 | grep LISTEN | awk '{print $2}' | xargs -r sudo kill -9

# Start viewer after launching orthanc server
echo "Starting dev:orthanc..."
# Start the orthanc service in the background and detach it
nohup yarn orthanc:up > orthanc_up.log 2>&1 &

echo "Waiting for $timeout seconds to allow dev:orthanc to initialize properly..."
# Wait for another a bit
sleep $timeout

yarn run dev:orthanc --verbose > orthanc_run.log &

echo "Starting orthanc:up, waiting for $timeout seconds to initialize..."
# Wait for a bit
sleep $timeout

# Execute the sync and infer script
# echo "Executing the Sync and infer script..."
# "$PYTHON_PATH" "$SYNC_AND_INFER_SCRIPT_PATH" &

echo "Server services are up and ready."
