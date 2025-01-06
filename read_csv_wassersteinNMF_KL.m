%% Import the simulated data from a CSV file
simulated_data_path = '/Users/benjamingauthier/Desktop/Mutational_Signatures/simulated_data/test1.csv'; % Path to your CSV file
data_table = readtable(simulated_data_path); % Read the CSV as a table

%% Convert the table to a numeric matrix
data = table2array(data_table);
data(1, :) = []; % Remove the first row
data(:, 1) = []; % Remove the first column

%% Normalize the data
data = bsxfun(@rdivide, data, sum(data, 1)); % Normalize columns to sum to 1

data_min = min(data(:)); % Minimum value in the entire data matrix
data_max = max(data(:)); % Maximum value in the entire data matrix
fprintf('Data range: %.4f to %.4f\n', data_min, data_max);

%% Generate bin centers
minVal = data_min;
maxVal = data_max;
num_bins = size(data, 1); % Number of rows in your data
x = linspace(minVal, maxVal, num_bins); % Generate bin centers

%% Build the cost matrix
M = abs(bsxfun(@minus, x', x)); % Compute pairwise distances
M = M / median(M(:));           % Normalize the cost matrix

%% Set parameters for Wasserstein NMF
options.stop = 1e-3;
options.verbose = 2;
options.D_step_stop = 5e-5;
options.lambda_step_stop = 5e-4;
options.alpha = 0.5;
options.Kmultiplication = 'symmetric';
options.GPU = 0;

k = 3; % Number of components (dictionary size)
gamma = 1 / 50;
wassersteinOrder = 1;
rho1 = 0.1; % Regularization parameter for coefficients
rho2 = 0.1; % Regularization parameter for dictionary

%% Perform Wasserstein NMF
fprintf('Performing Wasserstein NMF...\n');
[D, lambda, objectives] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);

% Plot the objective values to check convergence
figure;
plot(objectives);
xlabel('Number of outer iterations');
ylabel('Objective Function');
title('Wasserstein NMF Objective Convergence');

%% Compare data and reconstruction
% Reconstruct the data
reconstructed_data = D * lambda;

% Quantitative evaluation
reconstruction_error = norm(data - reconstructed_data, 'fro'); % Frobenius norm
explained_variance = 1 - (norm(data - reconstructed_data, 'fro')^2 / norm(data, 'fro')^2);
fprintf('Wasserstein NMF Reconstruction Error (Frobenius norm): %.4f\n', reconstruction_error);
fprintf('Wasserstein NMF Explained Variance: %.4f\n', explained_variance);

%% Visualize original data and reconstruction
sample_idx = 1; % Choose a sample to visualize
figure;
subplot(1, 2, 1);
plotDictionary(x, data(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Original Data');
title('Original Data');

subplot(1, 2, 2);
plotDictionary(x, reconstructed_data(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Reconstructed Data');
title('Wasserstein NMF Reconstruction');

%% Compare with regular NMF using KL divergence
fprintf('Performing Regular NMF with KL Divergence...\n');
[D_nmf, lambda_nmf] = nnmf(data, k, 'algorithm', 'mult', 'replicates', 5);

% Reconstruct the data using regular NMF
reconstructed_data_nmf = D_nmf * lambda_nmf;

% Quantitative evaluation
reconstruction_error_nmf = norm(data - reconstructed_data_nmf, 'fro'); % Frobenius norm
explained_variance_nmf = 1 - (norm(data - reconstructed_data_nmf, 'fro')^2 / norm(data, 'fro')^2);
fprintf('Regular NMF Reconstruction Error (Frobenius norm): %.4f\n', reconstruction_error_nmf);
fprintf('Regular NMF Explained Variance: %.4f\n', explained_variance_nmf);

%% Visualize comparison between Wasserstein NMF and Regular NMF
figure;
subplot(1, 3, 1);
plotDictionary(x, data(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Original Data');
title('Original Data');

subplot(1, 3, 2);
plotDictionary(x, reconstructed_data(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Wasserstein NMF Reconstruction');
title('Wasserstein NMF Reconstruction');

subplot(1, 3, 3);
plotDictionary(x, reconstructed_data_nmf(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Regular NMF Reconstruction');
title('Regular NMF Reconstruction');
