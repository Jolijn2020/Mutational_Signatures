import pandas as pd
import numpy as np
import os
import random
import math
import json

def simulate_data(config):
    # signatures_file_path, signatures_to_extract, n_samples, average_noise, save_dir=None

    # Read in the signatures
    signatures = get_signatures(config["signatures_file_path"], config["signatures_to_extract"])

    # create the sample distributions
    sample_distr_config = config["sample_distribution"]
    sample_distr_function = get_distribution_function(sample_distr_config)

    sample_distributions = get_distribution_of_samples(signatures, config["n_samples"],  sample_distr_config['use_sign_active_prob'], sample_distr_config.get('sign_active_prob'), sample_distr_config.get('n_sign_active'), sample_distr_function)

    # Simulate the data (including noise)
    noise_function = get_distribution_function(config['noise_distribution'])
    counts_function = get_distribution_function(config['counts_distribution'])
    simulated_data = calculate_counts(signatures, sample_distributions, noise_function, counts_function)

    data_file, config_file = create_file_names(config['save_dir'], config['signatures_to_extract'])
    simulated_data.to_csv(data_file, index=True)
    print('Sucessfully saved simulated data in ' + data_file)

    with open(config_file, 'w') as f:
        json.dump(config, f)
    print('Sucessfully saved meta-data in ' + config_file)

    return simulated_data

def get_distribution_function(config):
    failed = False
    match config['distribution'].lower():
        case "poisson":
            if config.get('avg') != None:
                def func():
                    return np.random.poisson(config.get('avg'))
            else:
                failed = True
        case "uniform":
            if config.get('min') != None and config.get('max') != None:
                def func():
                    return random.uniform(config.get('min'), config.get('max'))
            else:
                failed = True
        case "logscale":
            if config.get('min') != None and config.get('max') != None:
                def func():
                    return 10 ** (random.uniform(math.log(config.get('min'), 10), math.log(config.get('max'), 10)))
            else:
                failed = True
        case "normal":
            if config.get('avg') != None and config.get('sigma') != None:
                def func():
                    return random.normalvariate(config.get('avg'), config.get('sigma'))
            else:
                failed = True

    if failed:
        raise RuntimeError("The right value(s) were not specified for the " + config.get('distribution') + " distribution.")
    return func


def create_file_names(dir_path, signatures):
    # get current files
    files = os.listdir(dir_path)
    num = int(len(files)/2)
    num_used = True
    while num_used:
        num += 1
        substring = 'data_v' + str(num) + '_'
        use_num_files = [f for f in files if substring in f]
        if len(use_num_files) == 0:
            num_used = False

    name = dir_path + "/data_v" + str(num) +  "_sign_"
    for signature in signatures:
        name += str(signature[3:]) + "_"
    name += ".csv"

    name_config = dir_path + "/config_v" + str(num) + ".json"
    return name, name_config

def get_signatures(file_path, signatures_to_extract):

    df = pd.read_csv(file_path, sep='\t')
    
    # Set Type  as the index 
    if 'Type' in df.columns:
        df.set_index('Type', inplace=True)
    
    signatures = df[signatures_to_extract]
    
    return signatures


def get_distribution_of_samples(signatures, n_samples, use_sign_active_prob, sign_active_prob, n_sign_active, sign_distribution):
    df_sparse = pd.DataFrame()
    for i in range(n_samples):

        # Get a distribution of the counts
        # Use the probability of a signature being present
        if use_sign_active_prob:
            distribution = [sign_distribution() if random.uniform(0, 1) < sign_active_prob else 0 for x in range(0, signatures.shape[1])]
            total = sum(distribution)
            while(sum(distribution)==0):
                distribution = [sign_distribution() if random.uniform(0, 1) < sign_active_prob else 0 for x in range(0, signatures.shape[1])]
                total = sum(distribution)

        # Use the predetermined amount of signatures to be used and select randomly from the signatures list
        else:
            if signatures.shape[1] < n_sign_active:
                distribution = [sign_distribution() for x in range(0, signatures.shape[1])]
            else:
                distribution = [0] * n_sign_active
                sign_count = 0
                # Keep adding signatures until have enough
                while sign_count < n_sign_active:
                    index = random.randint(0, n_sign_active)
                    if distribution[index] == 0:
                        distribution[index] = sign_distribution()
                        sign_count += 1

        # normalize
        total = sum(distribution)
        distribution = [x/total for x in distribution]
        df_sparse[i] = distribution

    df_sparse = df_sparse.set_index(signatures.columns)
    return df_sparse

# def get_noise_function(func_string, min=0, max=20, avg=10):
#     match func_string.lower():
#         case "poisson":
#             def noise_func():
#                 return np.random.poisson(avg)
#         case "uniform":
#             def noise_func():
#                 return random.uniform(min, max)
#         case _:
#             def noise_func():
#                 return np.random.poisson(10)
#     return noise_func

# def get_n_counts_function(func_string, min=1000, max=50000, avg=10000, sigma=5000):
#     match func_string.lower():
#         case "logscale":
#             def n_counts_func():
#                 return 10 ** (random.uniform(math.log(min, 10), math.log(max, 10)))
#         case "uniform":
#             def n_counts_func():
#                 return random.uniform(min, max)
#         case "normal":
#             def n_counts_func():
#                 return random.normalvariate(avg, sigma)
#         case _:
#             def n_counts_func():
#                 return 10 ** (random.uniform(math.log(1000, 10), math.log(50000, 10)))
#     return n_counts_func

# def get_sample_distribution_function(distribution, min, max, avg, sigma):

#     match distribution.lower():
#         case "poisson":
#             def func():
#                 return np.random.poisson(avg)
#         case "uniform":
#             def func():
#                 return random.uniform(min, max)
#         case "normal":
#             def func():
#                 return random.normalvariate(avg, sigma)
#         case _:
#             def func():
#                 return random.uniform(min, max)
#     return func

# def get_noise_function(distribution, min=0, max=20, avg=10, sigma=1):
#     match distribution.lower():
#         case "poisson":
#             def func():
#                 return np.random.poisson(avg)
#         case "uniform":
#             def func():
#                 return random.uniform(min, max)
#         case _:
#             def func():
#                 return np.random.poisson(10)
#     return func

# def get_n_counts_function(distribution, min=1000, max=50000, avg=10000, sigma=5000):
#     match distribution.lower():
#         case "logscale":
#             def func():
#                 return 10 ** (random.uniform(math.log(min, 10), math.log(max, 10)))
#         case "uniform":
#             def func():
#                 return random.uniform(min, max)
#         case "normal":
#             def func():
#                 return random.normalvariate(avg, sigma)
#         case _:
#             def func():
#                 return 10 ** (random.uniform(math.log(1000, 10), math.log(50000, 10)))
#     return func

def calculate_counts(signatures, sample_distributions, noise_func, n_counts_func):
    simulated_data = signatures.dot(sample_distributions)
    for i in range(simulated_data.shape[1]):
        distribution = simulated_data[i]

        # The total number of mutations in a sample
        n_counts = n_counts_func()
        counts = [int(x*n_counts) for x in distribution]

        # Add noise
        noisy_counts = [x+noise_func() for x in counts]
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