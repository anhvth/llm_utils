#!/bin/bash

# Exit on error
set -e

# Stage current changes
git add -A
git commit -m "Pre-version bump commit" || true

# Ensure the working directory is clean
git diff-index --quiet HEAD --

# Bump version
bump2version patch --allow-dirty

# Build and install
pip install .

# Commit version bump
git add -A
git commit -m "Bump version to $(bump2version --allow-dirty --dry-run --list patch | grep new_version | sed -r s,"^.*=",,)"
git push
