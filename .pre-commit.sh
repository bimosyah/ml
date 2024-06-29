#!/bin/sh
# pre-commit hook to update requirements.txt

# Activate virtual environment
source env/bin/activate

# Update requirements.txt
pip3 freeze > requirements.txt

# Add requirements.txt to the commit
git add requirements.txt