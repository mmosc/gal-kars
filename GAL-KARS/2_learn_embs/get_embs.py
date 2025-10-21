import pandas as pd 
import pickle as pkl
import numpy as np

dataset = 'lfm-2b'

# mapping_ents = pd.read_csv(f'{dataset}/mapping_items.tsv', sep='\t', names=['id', 'name'])
# Convert artists to sequential ids
with open(f'{dataset}/top50_artist_to_unique_id.pkl', 'rb') as f:
  top50_artist_to_unique_id = pkl.load(f)

map_prep_ents = {i: artist for artist, i in top50_artist_to_unique_id.items()}
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
tr = pd.read_csv(f'{dataset}/usertrain.tsv', sep='\t')
va = pd.read_csv(f'{dataset}/uservalid.tsv', sep='\t')
te = pd.read_csv(f'{dataset}/usertest.tsv', sep='\t')

ratings = pd.concat([tr,va,te])

print(f"total users from ratings: {ratings['user_row_index'].nunique()}")
print(f"total items from ratings: {ratings['artist'].nunique()}")
print(f"total entities: {ratings['user_row_index'].nunique() + ratings['artist'].nunique()}")


for s in [1,2,3]:
  for l in [1,2,3]:
    embs = dict()
    ent_mat_id = pd.read_csv(f"results/{dataset}_setting_{s}_CompGCN_k=64_l={l}/entities_to_id.tsv", sep='\t', names=['entity', 'id'])
    map_ent_mat_id = ent_mat_id.set_index('entity').to_dict()['id']
    with open(f"results/{dataset}_setting_{s}_CompGCN_k=64_l={l}/embeddings.tsv", 'r') as emb_file:
        data = [np.fromstring(line.strip(), sep='\t') for line in emb_file]
        data = np.array(data, dtype=np.float64)

        # distribution for the few random embeddings
        mean = np.mean(data.flatten())
        std = np.std(data.flatten())
        
        missing_ids = set()


        for i, row in ratings.iterrows():
          user = row['user_row_index']
          item = row['artist']
          for name in [user, item]:
              if name not in map_ent_mat_id:
                rand_emb = np.random.normal(loc=mean, scale=std, size=data[0].shape)
                embs[name] = rand_emb
                missing_ids.add(name)
              else:
                matrix_id = map_ent_mat_id[name]
                embs[name] = data[matrix_id]

        print(s, l, len(missing_ids), len(embs))
        pkl.dump(embs, open(f'../3_recsys/data/embeddings/lfm-2b/lfm-2b_setting_{s}_CompGCN_k=64_l={l}.pkl', 'wb'))