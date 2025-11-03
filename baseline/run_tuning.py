from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

from recbole.quick_start import run_recbole

import os

def run_hyper(dataset, model_name):

    config_file_path = f'settings_{dataset}/{model_name}.yaml'
    params_file_path = f'settings_{dataset}/{model_name}.hyper'

    hp = HyperTuning(objective_function=objective_function, 
        algo='exhaustive', 
        early_stop=10,
        max_evals=100,
        params_file=params_file_path,
        fixed_config_file_list=[config_file_path])

    hp.run()



with open('log.txt', 'a') as fout:
    fout.write('\n----\n\n')

models = ['BPR']
            # 'BPR', 'EASE', 'ItemKNN', 'LightGCN', 'MultiVAE',
            # 'NeuMF', 'NGCF', 'SLIMElastic', 'SGL',
            #
            # 'CFKG_inner', 'CFKG_transe', 'CKE', 'KGCN',
            # 'KGNNLS', 'KTUP', 'MKR', 'KGAT', 'KGIN',
            #
            # ]
            
datasets = ['lfm-2b']

for model in models:  
    for dataset in datasets:
        with open('log.txt', 'a') as fout:
            fout.write(f'Starting {model} on {dataset}... ')
            fout.flush()
            try:
                run_hyper(dataset, model)
                fout.write(f'completed.\n')
                fout.flush()
            except Exception as e:
                fout.write(f'error: {str(e)}.\n')
                fout.flush()