#!/bin/bash

# Step a: Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step b: Run necessary parts of the codebase
#python src/document_term_matrix.py &
#python src/cosine_similarity.py &
python src/synthetic_user_data.py &

wait