if ! bump2version --allow-dirty patch; then
    echo "Error: bump2version failed"
    exit 1
fi

git add -A && git commit -m "Bumped version to $(bump2version --dry-run --list patch | grep new_version | cut -d'=' -f2 | xargs)"
git push