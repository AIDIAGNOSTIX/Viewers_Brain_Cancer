#!/bin/bash

# Define the service name
SERVICE_NAME="WakeupServerService"

# Get the full path of the current script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Full path to the target script
TARGET_SCRIPT="$SCRIPT_DIR/start_server.sh"

# Create the systemd service file content
SERVICE_FILE_CONTENT="[Unit]
Description=$SERVICE_NAME

[Service]
Type=simple
ExecStart=/bin/bash $TARGET_SCRIPT

[Install]
WantedBy=multi-user.target"

# Path where the systemd service file will be placed
SYSTEMD_SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME.service"

# Write the service file content to the systemd service file
echo "$SERVICE_FILE_CONTENT" | sudo tee "$SYSTEMD_SERVICE_PATH"

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start at boot
sudo systemctl enable "$SERVICE_NAME.service"

# Optional: start the service immediately to test
echo "Starting the service..."
sudo systemctl start "$SERVICE_NAME.service"

echo "$SERVICE_NAME service is set up and started."
