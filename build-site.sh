#!/bin/bash
# Build script for Physical AI & Humanoid Robotics textbook site

echo "Building Physical AI & Humanoid Robotics textbook site..."

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
fi

echo "Building the site..."
npm run build

if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "The site is available in the build/ directory"
    echo "To serve locally: npx serve build"
else
    echo "Build failed!"
    exit 1
fi