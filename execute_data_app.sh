#!/usr/bin/env bash

data_app="$1"

if [[ -z "$data_app" ]]; then
    echo "Usage: $0 <data_app_name>"
    exit 1
fi

# Accept app names with optional .py suffix
data_app="${data_app%.py}"

echo "Executing $data_app"
# setting up env
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    echo "$key=$value"
    export "$key=$value"
done < .env
# activate .venv & make sure deps are installed
source .venv/bin/activate
.venv/bin/python -m pip install -q -r requirements.txt
# go into package root
cd src
# execute data app
if [[ "$data_app" == "streamlit_app" ]]; then
    streamlit run threephi_framework/data_apps/streamlit_app.py
else
    python -m threephi_framework.data_apps.$data_app
fi
# go back up
cd ..
