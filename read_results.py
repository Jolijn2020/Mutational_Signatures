import json
import pandas as pd
import numpy as np

def load_data(path):
"""
This method takes a Results*.json file and reads it into a dictionary
parsed_data['params'] has the parameters used for the run

'D', 'lambda', 'D_nmf', 'lambda_nmf', 'objectives_wnmf', 'objectives_klnmf', 'reconstructed_data', 'reconstructed_data_nmf':
contains a pd.DataFrame of the respective data

'all_Dw', 'all_lambdaw', 'all_reconstructed_dataw', 'all_Dkl', 'all_lambdakl', 'all_reconstructed_datakl'
contains a dictionary with a key for the each of the runs, so
parsed_data['all_Dw'][0] will contain the first D 
parsed_data['all_Dw'][0] will contain the second D 
etc

"""
    with open(path, 'r') as file:
        data = json.load(file)

    parsed_data = dict()
    for key in data.keys():
        if subfield.ndim == 0:
            parsed_data[key] = subfield
            continue
        if subfield.ndim in [1, 2]:
            parsed_data[key] = pd.DataFrame(subfield)
        else: # 3 dim
            parsed_data[key] = dict()
            for i, subarray in enumerate(subfield):
                parsed_data[key][i] = pd.DataFrame(subarray)
    return parsed_data
