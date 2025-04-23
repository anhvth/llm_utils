#!/bin/bash

# Get the current version before making any changes
CURRENT_VERSION=$(bump2version --dry-run --list patch --allow-dirty | grep current_version | cut -d'=' -f2 | xargs)
NEW_VERSION=$(bump2version --dry-run --list patch --allow-dirty | grep new_version | cut -d'=' -f2 | xargs)

echo "Attempting to bump version from $CURRENT_VERSION to $NEW_VERSION..."

# Ensure all changes are committed first
echo "Committing any pending changes..."
git add -A && git commit -m "Pre-release commit" || echo "No changes to commit or commit failed"

# Now bump the version (allow dirty working directory)
if ! bump2version patch --allow-dirty; then
    echo "Error: bump2version failed"
    exit 1
fi

echo "Successfully bumped version to $NEW_VERSION"

# Push the changes and tag to remote
git push && git push --tags

echo "Successfully pushed version $NEW_VERSION to GitHub"