#!/bin/bash

# Exit on error
set -e

# Stage current changes
git add -A
git commit -m "Pre-version bump commit" || true

# Bump version
# --allow-dirty permits version bumping even with uncommitted changes in the working directory
bump2version --allow-dirty patch

# Build and install
pip install .

# Commit version bump
git add -A
git commit -m "Bump version to $(bump2version --dry-run --list patch | grep new_version | sed -r s,"^.*=",,)"
git push
