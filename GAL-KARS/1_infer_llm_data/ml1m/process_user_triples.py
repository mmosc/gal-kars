import pandas as pd
import json
import os
from tqdm import tqdm

# read item propos
user_files = [f'users/{file}' for file in os.listdir('users/')]

counter = 0
triples = []


key_map = {'genre': 'genre',
        'genres': 'genre',
        'kind_of_plot': 'kind_of_plot',
        'kind_of_musical_score': 'kind_of_musical_score',
        'kind mood_for_the_musical_score plot': 'mood_for_the_musical_score',
        'visual_style': 'visual_style',
        'writing_style': 'writing_style',
        'runtime': 'runtime',
        'theme': 'themes',
        'themes': 'themes'
}

for user_file in tqdm(user_files, total=len(user_files)):

    # get the id of the user
    user_id = user_file.split('/')[-1].split('_')[0]

    # read the json file
    json_prop = json.load(open(user_file, 'r'))

    # read each prop
    for key in json_prop:

        # if the prop is in the dict, then use that unified name
        if key in key_map:
            name_key = key_map[key]
            
            # get each value and build the triple
            for value in json_prop[key]:

                if value.strip() == '':
                    continue

                # replace blank in value
                value = value.replace(' ','_')

                # build the triple and ignore duplicates
                triple = f'{user_id}\tllama_generated/{name_key}\tllama_generated/{value}\n'
                if triple not in triples:
                    triples.append(triple)
                    counter += 1

# save the KG
with open(f'LLM_user_triples.tsv', 'w') as fout:
    for triple in triples:
        fout.write(triple)

print(counter)