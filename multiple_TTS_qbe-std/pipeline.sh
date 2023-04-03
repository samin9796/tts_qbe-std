#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate inference
echo -n "Type the query: "
read query
python run_tts.py $query
conda deactivate
bash format.sh
conda activate qbe-std
python make_csv.py
docker-compose run --rm dev
rm -r output; mkdir output
python generate_output.py
