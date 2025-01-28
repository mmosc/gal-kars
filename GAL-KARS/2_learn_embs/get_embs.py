import pandas as pd 
import pickle as pkl
import numpy as np

dataset = 'ml1m'

mapping_ents = pd.read_csv(f'{dataset}/mapping_items.tsv', sep='\t', names=['id', 'name'])
print(f"items in mapping: {len(mapping_ents)}")

map_prep_ents = mapping_ents.set_index('id').to_dict()['name']
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
print(f"id -> name: {len(id_name.keys())}")

# read ratings to get user and item IDs
tr = pd.read_csv(f'{dataset}/train_sorted.tsv', sep='\t', names=['user','item','rating'])
te = pd.read_csv(f'{dataset}/test_sorted.tsv', sep='\t', names=['user','item','rating'])
ratings = pd.concat([tr,te])

users = [u for u in set(ratings['user'])]
print(f"total users from ratings: {len(users)}")
print(f"total items from ratings: {len(set(ratings['item']))}")
print(f"total entities: {len(set(ratings['item'])) + len(users)}")

# add user IDs
for u in users:
  id_name[u] = f'user{u}'

for s in [8]:

  for l in [1,2,3]:

    embs = dict()

    ent_mat_id = pd.read_csv(f"results/setting_{s}_CompGCN_k=64_l={l}/entities_to_id.tsv", sep='\t', names=['entity', 'id'])
    map_ent_mat_id = ent_mat_id.set_index('entity').to_dict()['id']

    with open(f"results/setting_{s}_CompGCN_k=64_l={l}/embeddings.tsv", 'r') as emb_file:
        data = [np.fromstring(line.strip(), sep='\t') for line in emb_file]
        data = np.array(data, dtype=np.float64)

        # distribution for the few random embeddings
        mean = np.mean(data.flatten())
        std = np.std(data.flatten())
        
        missing_ids = set()

        for dataset_id in id_name:
          name = (id_name[dataset_id])
          if name not in map_ent_mat_id:
            rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
            embs[int(dataset_id)] = rand_emb
            missing_ids.add(dataset_id)
          else:
            matrix_id = map_ent_mat_id[name]
            embs[int(dataset_id)] = data[matrix_id]


        for i, row in te.iterrows():
          if row['user'] not in embs:
            rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
            embs[int(row['item'])] = rand_emb
          if row['item'] not in embs:
            rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
            embs[int(row['item'])] = rand_emb

        print(s, l, len(missing_ids), len(embs))
        pkl.dump(embs, open(f'out_embs/dbbook_s={s}_CompGCN_k=64_l={l}.pkl', 'wb'))