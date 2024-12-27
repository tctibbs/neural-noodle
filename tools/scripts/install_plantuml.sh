#!/bin/bash

# ==============================================================================
# Script: install_plantuml.sh
# Purpose: Installs PlantUML and its dependencies, including Java and Graphviz,
#          for rendering UML diagrams in a Linux environment.
# Requirements: Requires root privileges to install system dependencies.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Function: Print a message with a timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting PlantUML and Graphviz installation..."

# Update the package list
log "Updating package list..."
apt-get update -qq

# Install Java for PlantUML
log "Installing Java.."
apt-get install -y default-jre-headless > /dev/null

# Install Graphviz for diagram rendering
log "Installing Graphviz.."
apt-get install -y graphviz > /dev/null

# Install PlantUML
log "Installing PlantUML.."
apt-get install -y plantuml > /dev/null

# Set the GRAPHVIZ_DOT environment variable
log "Configuring environment variable GRAPHVIZ_DOT..."
echo "export GRAPHVIZ_DOT=/usr/bin/dot" >> /etc/environment
export GRAPHVIZ_DOT=/usr/bin/dot

# Verify PlantUML installation
log "Verifying PlantUML installation..."
plantuml -version

# Clean up unused package lists
log "Cleaning up unused package lists..."
rm -rf /var/lib/apt/lists/*

log "PlantUML and Graphviz installation complete!"
