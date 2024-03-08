#!/bin/bash

# Ensure we're in the directory of the package to check
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in the current directory."
    exit 1
fi

# Get the package name and current version from poetry
PACKAGE_NAME=$(poetry version | cut -d' ' -f1)
CURRENT_VERSION=$(poetry version --short)

# Fetch the published package metadata from PyPI
REMOTE_METADATA=$(curl -s https://pypi.org/pypi/$PACKAGE_NAME/$CURRENT_VERSION/json)

# Check if the package has been published
if echo "$REMOTE_METADATA" | grep -q "Not Found"; then
    echo "$PACKAGE_NAME $CURRENT_VERSION has not been published yet."
    exit 0
fi

# Build the package and calculate the md5 digest of the distribution file
rm -rf dist
poetry build
DIST_FILE=$(ls dist/*.whl)
LOCAL_DIGEST=$(md5sum "$DIST_FILE" | cut -d' ' -f1)

# Extract the remote digest and compare it with the local digest
REMOTE_DIGEST=$(echo "$REMOTE_METADATA" | jq -r '.urls[0].digests.md5')
if [ "$REMOTE_DIGEST" != "$LOCAL_DIGEST" ]; then
    echo "$PACKAGE_NAME has changed and needs a version bump."
    exit 3
else
    echo "$PACKAGE_NAME is up to date."
fi

# Clean up the build artifacts
rm -rf dist