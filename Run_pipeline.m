params = struct();
params.file_path = 'simulated_data_01_14/s_8_n_0.02_GRCh37_10a_56_10d_52_36_91_45_38.csv';
params.num_iter = 5;
params.visu = 1;
params.verb = 2;
params.Alpha = 0.05;
params.Km = 'symmetric';
params.GPU = 0;
params.k = 8;
params.Gamma = 1/50;
params.wO = 1;
params.rho1 = 0.2;
params.rho2 = 0.2;
params.stop = 1e-3;
params.Dss = 0.0007;
params.lss = 0.0004;
params.experiment_name = 'with_hamming';
params.M_path = 'distances/hamming_distances.tsv';


all_lambda = Pipeline(params)