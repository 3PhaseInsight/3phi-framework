#!/usr/bin/env bash

echo "Starting 3PHI Streamlit app"
# load env
while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    export "$key=$value"
done < .env
# activate .venv & make sure deps are installed
source .venv/bin/activate
.venv/bin/python -m pip install -q -r requirements.txt
# run the streamlit app
streamlit run src/threephi_framework/streamlit/app.py
