#!/bin/bash

# Step a: Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 src/data_loader.py &

wait