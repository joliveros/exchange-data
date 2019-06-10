#!/bin/bash

: ${ENV_SECRETS_DIR:=/run/secrets}

env_secrets() {
    for secret in $ENV_SECRETS_DIR/*; do
        env_var_name=$(basename $secret)
        export $env_var_name=$(cat $secret);
        echo "Env variable $env_var_name set."
    done
}
