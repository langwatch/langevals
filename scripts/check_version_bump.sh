#!/bin/bash

# Ensure we're in the directory of the package to check
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in the current directory."
    exit 1
fi

# Get the package name and current version from poetry
PACKAGE_NAME=$(poetry version | cut -d' ' -f1)
CURRENT_VERSION=$(poetry version --short)

# Build the package and calculate the md5 digest of the distribution file
poetry build
DIST_FILE=$(ls dist/*.whl)
LOCAL_DIGEST=$(md5sum "$DIST_FILE" | cut -d' ' -f1)

# Fetch the published package metadata from PyPI
REMOTE_METADATA=$(curl -s https://pypi.org/pypi/$PACKAGE_NAME/$CURRENT_VERSION/json)

# Check if the package has been published
if echo "$REMOTE_METADATA" | grep -q "HTTP/2 404"; then
    echo "$PACKAGE_NAME has not been published yet."
    exit 0
else
    # Extract the remote digest and compare it with the local digest
    REMOTE_DIGEST=$(echo "$REMOTE_METADATA" | jq -r '.urls[0].digests.md5')
    if [ "$REMOTE_DIGEST" != "$LOCAL_DIGEST" ]; then
        echo "$PACKAGE_NAME has changed and needs a version bump."
        exit 3
    else
        echo "$PACKAGE_NAME is up to date."
    fi
fi

# Clean up the build artifacts
rm -rf dist