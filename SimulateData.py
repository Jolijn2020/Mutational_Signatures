import pandas as pd
import numpy as np
import os
import random

def simulate_data(signatures_file_path, signatures_to_extract, n_samples, average_noise, save_path=None):
    # Read in the signatures
    signatures = get_signatures(signatures_file_path, signatures_to_extract)


    sample_distributions = get_distribution_of_samples(signatures, n_samples)
    simulated_data = calculate_counts(signatures, sample_distributions, average_noise)


    if save_path!= None:
        simulated_data.to_csv(save_path, index=True)
        print('Sucessfully saved simulated data in ' + save_path)
    else:
        print('save_path not specified, so simulated data not saved')

    return simulated_data

def get_signatures(file_path, signatures_to_extract):

    df = pd.read_csv(file_path, sep='\t')
    
    # Set Type  as the index 
    if 'Type' in df.columns:
        df.set_index('Type', inplace=True)
    
    signatures = df[signatures_to_extract]
    
    return signatures

def get_distribution_of_samples(signatures, n_samples):
    df_sparse = pd.DataFrame()
    for i in range(n_samples):
        # TODO: find what distribution of signatures to use (Article uses 5 out of 10 for each sample)
        # Right now, use 0.4 percent chance of signatures being present, with the strength of it being between 0.5 and 2

        # Get a distribution of the counts
        distribution = [random.random()*1.5+0.5 if random.uniform(0, 1) > 0.6 else 0 for x in range(0, signatures.shape[1])]
        total = sum(distribution)
        while(sum(distribution)==0):
            distribution = [random.random()*1.5+0.5 if random.uniform(0, 1) > 0.6 else 0 for x in range(0, signatures.shape[1])]
            total = sum(distribution)

        # normalize
        total = sum(distribution)
        distribution = [x/total for x in distribution]
        df_sparse[i] = distribution

    df_sparse = df_sparse.set_index(signatures.columns)
    return df_sparse

def calculate_counts(signatures, sample_distributions, average_noise):
    simulated_data = signatures.dot(sample_distributions)
    for i in range(simulated_data.shape[1]):
        distribution = simulated_data[i]
        # Get the number of counts between 1001 and 50119 in logscale (50119 for easier numbers in formula)
        n_counts = 10 ** (random.uniform(3, 4.7))
        counts = [int(x*n_counts) for x in distribution]

        # Add Poisson noise
        noisy_counts = [x+np.random.poisson(average_noise) for x in counts]
        simulated_data[i] = noisy_counts

    return simulated_data



# Old methods, not used anymore, but might come in handy later
def get_signatures_from_old_files(folder_path):
    arr = os.listdir(folder_path)
    signatures = []
    signature_names = []
    for file in arr:
        if file.startswith("SBS"):
            signatures.append(pd.read_csv(os.path.join(folder_path, file)))
            signature_names.append(file.removesuffix('.csv'))
    return signatures, signature_names

def get_specific_signature_from_old_signatures(signatures_list, signature_names, signature='_GRCh37'):
    signatures = pd.DataFrame()
    for i in range(len(signature_names)):
        column = signatures_list[i][signature_names[i] + signature]
        signatures = pd.concat([signatures, column], axis = 1)

    return signatures