#!/bin/bash

CONFIG_FILE="env_variables.conf"
BASHRC="$HOME/.bashrc"
TEMP_BASHRC="$BASHRC.tmp"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "$CONFIG_FILE does not exist. Please create it based on env_variables.conf.skeleton."
    exit 1
fi

# Backup the original .bashrc file
cp "$BASHRC" "$BASHRC.backup"

# Prepare a temp bashrc without the old env variables
grep -v '# Appending environment variables from env_variables.conf' "$BASHRC" > "$TEMP_BASHRC"

# Append environment variables to the temp .bashrc
echo "# Appending environment variables from env_variables.conf to $BASHRC" >> "$TEMP_BASHRC"
while IFS='=' read -r key value
do
  # Remove any leading or trailing whitespace from the key
  key=$(echo $key | xargs)
  # Remove leading and trailing whitespace from the value
  value=$(echo $value | xargs)
  # Check if the line is not empty and not a comment
  if [[ -n $key && $key != \#* ]]; then
    # Properly quote the value and append the export statement to temp .bashrc
    echo "export $key=\"$value\"" >> "$TEMP_BASHRC"
  fi
done < "$CONFIG_FILE"

# Overwrite the original .bashrc with the temp file
mv "$TEMP_BASHRC" "$BASHRC"

echo "Environment variables have been added to $BASHRC. Please restart your terminal or source your .bashrc file."
source ~/.bashrc
