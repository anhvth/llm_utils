rm .env

echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to .gitignore"

git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch .env' --prune-empty --tag-name-filter cat -- --all

git push origin --force --all
git push origin --force --tags
