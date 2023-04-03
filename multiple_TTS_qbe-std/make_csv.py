'''
This script creates the result.csv file that is the output of the TTS + QbE-STD System
'''

import os
import glob
import pandas as pd

# These two lists will contain all the filenames from the queries and references directories
queries = []
references = []

# Get the current location
dir_path = os.path.dirname(os.path.realpath(__file__))

# All files and directories ending with .wav and that don't begin with a dot:
dir_queries = glob.glob(dir_path + "/data/raw/datasets/gos-kdl/synthesized_queries/*.wav")
dir_references = glob.glob(dir_path + "/data/raw/datasets/gos-kdl/references/*.wav")

# Get only the filenames, full paths are not required
for path in dir_queries:
    filename = path.split('/')[-1][:-4]
    queries.append(filename)

for path in dir_references:
    filename = path.split('/')[-1][:-4]
    references.append(filename)

# Initialize an empty dataframe with two columns
df = pd.DataFrame(columns=['query', 'reference'])

# Write all the query-reference pairs
for query in queries:
    for reference in references:
        dic = {'query': query, 'reference': reference}
        df = df.append(dic, ignore_index = True)

df.to_csv(dir_path + "/data/raw/datasets/gos-kdl/labels.csv", index=False)

