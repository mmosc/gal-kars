import clayrs.content_analyzer as ca
import clayrs.evaluation as eva
import pandas as pd
import os
import warnings
from tqdm import tqdm


warnings.filterwarnings("ignore")

def eval(dataset='', prediction_type_model='', ks=5, relevant_threshold=1):

    if prediction_type_model == 'baseline':
        preds_fold = f'preds/{dataset}/baselines/'
        prediction_dest_results = 'results/baselines/'
    else:
        preds_fold = f'preds/{dataset}/uni/'
        prediction_dest_results = f'results/{dataset}/uni/'

    # files to be evaluated
    # preds_fold = f'{prediction_type_model}/{dataset}'
    prediction_list = [x for x in os.listdir(f'{preds_fold}') if f'top{ks}' in x]
    print('Predictions to be evaluated:', prediction_list)

    # train data
    train_data = ca.CSVFile(os.path.join('data', f"{dataset}", "train.tsv"), separator="\t")
    train_ratings = ca.Ratings(train_data)

    # test data
    test_data = ca.CSVFile(os.path.join('data', f"{dataset}", "test.tsv"), separator="\t")
    test_ratings = ca.Ratings(test_data)
    
    dict_results = {}

    # read existing result dataframe
    existing = False
    if os.path.exists(f'{prediction_dest_results}/{dataset}_{prediction_type_model}_results_top{ks}.tsv'):
        results = pd.read_csv(f'{prediction_dest_results}/{dataset}_{prediction_type_model}_results_top{ks}.tsv', sep='\t')
        existing = True

    for _, prediction in enumerate(prediction_list):

        if existing:
            if len(results[results['model'] == prediction]) > 0:
                print(f'skipping {prediction}')
                continue

        metric_list = []

        # eval at each k
        k = ks
        metric_list.extend([
            eva.PrecisionAtK(k=k, relevant_threshold=relevant_threshold),
            eva.RecallAtK(k=k, relevant_threshold=relevant_threshold),
            eva.FMeasureAtK(k=k, relevant_threshold=relevant_threshold),
            eva.NDCGAtK(k=k),
            eva.GiniIndex(k=k),
            eva.EPC(k=k, original_ratings=train_ratings, ground_truth=test_ratings),
            eva.APLT(k=k, original_ratings=train_ratings),
            
        ])

        eval_summary = ca.CSVFile(os.path.join(f'{preds_fold}', f"{prediction}"), separator="\t")

        truth_list = [test_ratings]
        rank_list = [ca.Rank(eval_summary)]

        em = eva.EvalModel(
            pred_list=rank_list,
            truth_list=truth_list,
            metric_list=metric_list
        )
        
        # compute metrics and save user results for statistical tests
        sys_result, users_result = em.fit()
        sys_result = sys_result.loc[['sys - mean']]
        sys_result.reset_index(drop=True, inplace=True)
        sys_result['model'] = prediction
        sys_result.columns = [x.replace(" - macro", "") for x in sys_result.columns]
        cols = list(sys_result.columns)
        cols = cols[-1:] + cols[:-1]
        sys_result = sys_result.loc[:, cols]
        dict_results[prediction] = sys_result
        

    new_results = pd.concat([v for v in dict_results.values()]).reset_index(drop=True).sort_values(by=['model'], ascending=[True])
    print(new_results)

    if existing:
        results = pd.concat([results, new_results])
    else:
        results = new_results

    results = results.sort_values(by=['model'], ascending=[True])
    results.to_csv(f'{prediction_dest_results}/new_{dataset}_{prediction_type_model}_results_top{ks}.tsv', index=False, sep='\t')

    


relevant_threshold=1

# dataset ranges in ['ml1m', 'dbbook']
# prediction_type_model ranges in ['getall', 'baselines']

for dataset in ['dbbook']:
    for prediction_type_model in ['getall']:

        eval(dataset, prediction_type_model, ks=5, relevant_threshold=1)

