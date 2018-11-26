#!/usr/bin/env bash

conda_install="conda install --yes"

while read requirement; do
    conda_install+=" $requirement"
done < ./requirements-conda.txt

${conda_install}

pip install -r requirements.txt -r requirements-test.txt