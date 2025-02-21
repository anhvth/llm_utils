#!/bin/bash

# Exit on error
set -e

# Bump version
VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <new-version>"
    exit 1
fi

# Stage current changes
git add -A
git commit -m "Pre-version bump commit" || true

# Bump version (simplified command)
bump2version --new-version ${VERSION} --allow-dirty patch

# Build and install
pip install .

# Commit version bump
git add -A
git commit -m "Bump version to ${VERSION}"
git push