#!/usr/bin/env bash

token="$1"
address="$2"
env="$3"

if [ -z "$env" ]; then
  env="prod"
fi

docker build -t "slplus/frontend-react-template:0.0.1" \
  --build-arg VAULT_TOKEN="$token" \
  --build-arg VAULT_ADDR="$address" \
  --build-arg ENVIRONMENT="$env" .
