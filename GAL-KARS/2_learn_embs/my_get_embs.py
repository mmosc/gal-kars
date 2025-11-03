import pandas as pd 
import pickle as pkl
import numpy as np

dataset = 'lfm-2b'

dataset_columns = {
    'lfm-2b': ['user_row_index', 'edge', 'artist']
}

target_columns = ['user', 'item', 'rating']
dataset_columns_rename = {
    'lfm-2b': {
        'user_row_index': 'user',
        'edge': 'rating',
        'artist': 'item',
    }
}
# mapping_ents = pd.read_csv(f'{dataset}/mapping_items.tsv', sep='\t', names=['id', 'name'])


# Convert artists to sequential ids
with open(f'{dataset}/top50_artist_to_unique_id.pkl', 'rb') as f:
  top50_artist_to_unique_id = pkl.load(f)

map_prep_ents = {i: artist for artist, i in top50_artist_to_unique_id.items()}
# print(f"items in mapping: {len(mapping_ents)}")
# map_prep_ents = mapping_ents.set_index('id').to_dict()['name']
print(f"items in mapping dict: {len(map_prep_ents)}")

id_name = dict()
name_id = dict()
for k,v in map_prep_ents.items():
  if ';' in v:
    id_name[k] = v.split(';')[1]
    name_id[v.split(';')[1]] = k
  else:
    id_name[k] = v
    name_id[v] = k

# ID items -> name
print(f"id -> name items: {len(id_name.keys())}")

# read ratings to get user and item IDs
# train also includes the interactions used to train pykeen and to test pykeen
# val and test interactions are never included in pykeen data (neither in pykeen train nor in pykeen test)
tr = pd.read_csv(f'{dataset}/train.tsv', sep='\t')
va = pd.read_csv(f'{dataset}/valid.tsv', sep='\t')
te = pd.read_csv(f'{dataset}/test.tsv', sep='\t')

tr = dataset_columns_rename
va = va.rename(columns=dataset_columns_rename[dataset])
te = te.rename(columns=dataset_columns_rename[dataset])

tr['rating'] = 1
va['rating'] = 1
te['rating'] = 1

tr = tr[['user', 'item', 'rating']]
va = va[['user', 'item', 'rating']]
te = te[['user', 'item', 'rating']]

tr['item'] = tr['item'].map(top50_artist_to_unique_id)
va['item'] = va['item'].map(top50_artist_to_unique_id)
te['item'] = te['item'].map(top50_artist_to_unique_id)

tr['user'] = tr['user'].astype(str)
va['user'] = va['user'].astype(str)
te['user'] = te['user'].astype(str)

tr['item'] = tr['item'].astype(str)
va['item'] = va['item'].astype(str)
te['item'] = te['item'].astype(str)


tr['user'] = tr['user'].apply(lambda x: 'user' + x)
va['user'] = va['user'].apply(lambda x: 'user' + x)
te['user'] = te['user'].apply(lambda x: 'user' + x)

ratings = pd.concat([tr,te,va])

# print(ratings.user.nunique())

users = [u for u in set(ratings['user'])]
print(f"total users from ratings: {len(users)}")
print(f"total items from ratings: {len(set(ratings['item']))}")
print(f"total entities: {len(set(ratings['item'])) + len(users)}")

# add user IDs
for u in users:
  id_name[u] = u
print(f"id -> name total: {len(id_name.keys())}")
print(f"id -> name total: {len([int(id) for id in id_name.keys()])}")

# setting
for s in [1,2,3]:
  # layers
  for l in [1,2,3]:

    embs = dict()

    ent_mat_id = pd.read_csv(f"results/{dataset}_setting_{s}_CompGCN_k=64_l={l}/entities_to_id.tsv", sep='\t', names=['entity', 'id'])
    # print(ent_mat_id.dtypes)
    map_ent_mat_id = ent_mat_id.set_index('entity').to_dict()['id']

    with open(f"results/{dataset}_setting_{s}_CompGCN_k=64_l={l}/embeddings.tsv", 'r') as emb_file:
        data = [np.fromstring(line.strip(), sep='\t') for line in emb_file]
        data = np.array(data, dtype=np.float64)

        # distribution for the few random embeddings
        mean = np.mean(data.flatten())
        std = np.std(data.flatten())
        
        missing_ids = set()
        # print(len(set(id_name.keys())))
        for dataset_id, entity_name in id_name.items():
          name = (entity_name)
          if name not in map_ent_mat_id:
            print(f'{name} not in map_ent_mat_id')
            rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
            # embs[int(dataset_id)] = rand_emb
            embs[entity_name] = rand_emb
            missing_ids.add(dataset_id)
          else:
            matrix_id = map_ent_mat_id[name]
            # embs[int(dataset_id)] = data[matrix_id]
            embs[entity_name] = data[matrix_id]

        total_missing = set()
        for i, row in te.iterrows():
          if row['user'] not in embs:
            print(type(row['user']))
            total_missing.add(row['user'])
            # rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
            # embs[int(row['item'])] = rand_emb
          if row['item'] not in embs:
            print(type(row['item']))
            total_missing.add(row['item'])
            # rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
            # embs[int(row['item'])] = rand_emb
        # print(embs.keys())
        print(f'{s}, {l}: without emb: {len(missing_ids)}, te without emb: {len(total_missing)}, total emb: {len(embs)}')
        pkl.dump(embs, open(f'{dataset}/_embeddings/{dataset}_s={s}_CompGCN_k=64_l={l}.pkl', 'wb'))