#!/usr/bin/env bash

data_app="$1"

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
.venv/bin/pip install -q -r requirements.txt
# go into package root
cd src
# execute data app
python -m threephi_framework.data_apps.$data_app
# go back up
cd ..
