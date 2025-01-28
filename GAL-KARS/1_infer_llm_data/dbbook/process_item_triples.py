import pandas as pd
import json
import os

# read item mapping
items = pd.read_csv('in/mapping_items.tsv', names=['id','title'], sep='\t')

# read item propos
item_files = [f'items/{file}' for file in os.listdir('items/')]

# map for keys of dict
key_map = {'genre': 'genre',
    'plot': 'plot',
    'subject': 'subject',
    'writing style': 'writing_style',
    'genres': 'genre',
    'author': 'author',
    'mood': 'mood',
    'topic': 'topic',
    'country': 'country',
    'kind': 'kind_book',
    'kind of book': 'kind_book',
    'kind of plot': 'kind_plot',
    'country in which the book has been written': 'country'
}

counter = 0
triples = []

for i, item_file in enumerate(item_files):

    # get the name of the item
    item_id = item_file.split('item')[-1].replace('.json','')
    name_item = items[items['id'] == int(item_id)]['title'].values[0].split(';')[1]

    # read the json file
    json_prop = json.load(open(item_file, 'r'))

    # read each prop
    for key in json_prop:

        # if the prop is in the dict, then use that unified name
        if key in key_map:
            name_key = key_map[key]
            
            # get each value and build the triple
            for value in json_prop[key]:

                # replace blank in value
                value = value.replace(' ','_')

                # build the triple and ignore duplicates
                triple = f'{name_item}\tllama_generated/{name_key}\tllama_generated/{value}\n'
                if triple not in triples:
                    triples.append(triple)
                    counter += 1

# save the KG
with open(f'LLM_item_triple.tsv', 'w') as fout:
    for triple in triples:
        fout.write(triple)

print(counter)