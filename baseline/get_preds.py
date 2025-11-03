from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model
from logging import getLogger
from hyperopt import tpe
import os
import pandas as pd
from tqdm.contrib import tzip
import torch
import itertools


def get_preds(dataset_name, model_name):
    _, model, dataset, _, _, test_data = load_data_and_model(
        model_file=f'saved_{dataset_name}/{model_name}'
    )

    # get test data user-item pair recommendation score
    users = test_data.dataset.inter_feat['user_id']
    items = test_data.dataset.inter_feat['item_id']
    scores = model.predict(Interaction({'user_id': users,
                                        'item_id': items}).to('cuda'))

    # save to pandas
    predictions = pd.DataFrame()
    predictions['users'] = list(map(int, dataset.id2token(dataset.uid_field, users)))
    predictions['items'] = list(map(int, dataset.id2token(dataset.iid_field, items)))
    predictions['scores'] = scores.tolist()
    predictions = predictions[predictions['scores'] != 0]
    predictions = predictions.sort_values(by=['users', 'scores'], ascending=[True, False])

    pred_name = f'preds_{dataset_name}/test_rating/{model_name.replace(".pth", ".tsv")}'
    predictions.to_csv(pred_name, sep='\t', header=None, index=False)



def get_top_k(dataset_name, model_name, k=100):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file = f'saved_{dataset_name}/{model_name}'
    )

    # Set up device (GPU if available, else CPU) and put model in evaluation mode
    device = torch.device(config["device"])
    model.to(device)
    model.eval()

    # Get field names and user count from dataset
    user_field = dataset.uid_field  # "user" - field name for user IDs
    item_field = dataset.iid_field  #  "item" - field name for item IDs
    user_num = dataset.user_num  # total number of users in the dataset

    # List to store top-k items for each user
    all_user_items = []
    all_user_scores = []
    all_train_items, _, all_history_len = train_data.dataset.history_item_matrix()

    # Generate recommendations for each user
    for uid in range(user_num):
        # Create interaction object for current user
        # We need to format the input properly for the model
        u_tensor = torch.tensor([uid], device=device)
        interaction = Interaction({user_field: u_tensor}).to(device)

        # Get prediction scores for ALL items for this user
        # full_sort_predict returns scores for every item in the catalog
        scores = model.full_sort_predict(interaction)  # returns 1D version
        train_items = all_train_items[uid]
        # history_len = all_history_len[uid]
        # # the final entries are padded with 0's
        # which is good so we can exclude the 0 padding item from recommendations
        # print(uid, train_items[-10:], history_len)

        # Train items of user uid

        # Ensure scores have the right shape [1, n_items]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # make it [1, n_items]

        scores[0, train_items] = - 9.
        # Select top-k items with highest scores
        # torch.topk returns both values and indices - we only need indices
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)  # shape: [1, k]
        all_user_items.append(topk_idx.cpu())  # Move to CPU and store
        all_user_scores.append(topk_scores.cpu())  # Move to CPU and store


        # Progress tracking - print every 500 users
        if (uid + 1) % 500 == 0:
            print(f"processed {uid + 1}/{user_num} users")


    # Combine all user recommendations into single tensor
    # Result shape: [user_num, k]
    topk_indices = torch.cat(all_user_items, dim=0)
    topk_scores = torch.cat(all_user_scores, dim=0)

    # Convert internal IDs back to original tokens/IDs
    # Map user indices to their original user IDs
    user_tokens = dataset.id2token(user_field, list(range(user_num)))
    # Map item indices to their original item IDs for all recommendations
    item_tokens = dataset.id2token(item_field, topk_indices)

    # Build DataFrame for output
    rows = []
    dfs  = []
    for u_tok, items, scores in tzip(user_tokens, item_tokens.tolist(), topk_scores.tolist()):
        current_df = pd.DataFrame()
        current_df['user:token'] = [u_tok] * k
        current_df['item:token'] = items
        current_df['label:float'] = scores
        dfs.append(current_df)

    # Save results to TSV file
    out_path = f'preds_{dataset_name}/test_rating/{model_name.replace(".pth", f"_top_{k}.tsv")}'

    recs = pd.concat(dfs)
    recs.to_csv(out_path, sep="\t", index=False)
    print(f"Saved top-100 per user to {out_path}")




datasets = ['lfm-2b']

for dataset in datasets:
    print(dataset)
    # pths = [x for x in os.listdir(f'saved_{dataset}') if '.pth' in x and 'MultiVAE' in x]
    pths = [x for x in os.listdir(f'saved_{dataset}') if '.pth' in x]

    for pth in pths:
        if not os.path.exists(f'preds_{dataset}/test_rating'):
            os.makedirs(f'preds_{dataset}/test_rating')

        pred_name = f'preds_{dataset}/test_rating/{pth.replace(".pth",".tsv")}'

        if os.path.exists(pred_name):
            print(f'{pred_name} existing')
            continue
        else:
            print(f'{pred_name} not existing. Training', 'yellow')

        try:
            get_top_k(dataset, pth)
            with open('_report.txt', 'a') as fout:
                fout.write(f'OK {pth} \n')
        except Exception as e:
            with open('_report.txt', 'a') as fout:
                fout.write(f'ERROR {pth}')
                fout.write(str(e))
                fout.write('\n')