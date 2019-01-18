#!/usr/bin/env bash

set -e
ENV=pytorch-pretrained-BERT

conda create -n $ENV -c pytorch pytorch==1.0 cuda90 tqdm requests boto3 -y
echo "source activate $ENV" > .env
