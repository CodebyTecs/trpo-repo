#!/bin/bash

rm -rf trpo-repo
mkdir trpo-repo
cd trpo-repo

git init

echo "# TRPO Repo" > README.md
touch first.py
git add .
git commit -m "Initial commit"

echo "print('first change')" >> first.py
git add .
git commit -m "feat: add first changes"

echo "print('second change')" >> first.py
git add .
git commit -m "feat: add second changes"

git checkout -b branch1

echo "Branch1 change" >> README.md
touch second.py
git add .
git commit -m "feat: add third changes in branch1"

git tag branch1_start

git checkout main

echo "Main change" >> README.md
touch second_main.py
git add .
git commit -m "feat: add fourth changes"

git merge branch1 -X ours -m "Merge branch 'branch1'"

git checkout branch1
git reset --hard HEAD~1

git remote add origin https://github.com/CodebyTecs/trpo-repo.git

git push origin main
git push origin branch1
git push origin --tags

git checkout main
git pull origin main

echo "Fifth change" >> README.md
git add .
git commit -m "feat: add fifth changes"

git push origin main

git pull origin main

git log --oneline --graph --all --decorate