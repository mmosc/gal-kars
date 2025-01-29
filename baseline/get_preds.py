from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model
from logging import getLogger
from hyperopt import tpe
import os
import pandas as pd
from tqdm import tqdm


def get_preds(dataset_name, model_name): 

    _, model, dataset, _, _, test_data = load_data_and_model(
        model_file = f'saved_{dataset_name}/{model_name}'
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
    predictions = predictions[predictions['scores'] != 0 ]
    predictions = predictions.sort_values(by=['users', 'scores'],ascending=[True, False])

    pred_name = f'preds_{dataset_name}/test_rating/{model_name.replace(".pth",".tsv")}'
    predictions.to_csv(pred_name, sep='\t', header=None, index=False)


datasets = ['movielens','dbbook', 'lastfm']

for dataset in datasets:

    print(dataset)
    pths = [x for x in os.listdir(f'saved_{dataset}') if '.pth' in x and 'MultiVAE' in x]

    for pth in pths:

        pred_name = f'preds_{dataset}/test_rating/{pth.replace(".pth",".tsv")}'

        if os.path.exists(pred_name):
            print(f'{pred_name} existing')
            continue
        else:
            print(f'{pred_name} not existing. Training', 'yellow')
            

        try:
            get_preds(dataset, pth)
            with open('_report.txt', 'a') as fout:
                fout.write(f'OK {pth} \n')
        except Exception as e:
            with open('_report.txt', 'a') as fout:
                fout.write(f'ERROR {pth}')
                fout.write(str(e))
                fout.write('\n')