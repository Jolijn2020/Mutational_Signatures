%% Import the simulated data from a CSV file
simulated_data_path = './simulated_data/s_8_n_0.04_GRCh37_17b_86_98_39_22a_43_17a_13.csv'; 
data_table = readtable(simulated_data_path); % Read the CSV as a table



%% Convert the table to a numeric matrix
numericColumns = varfun(@isnumeric, data_table, 'OutputFormat', 'uniform'); 
data = table2array(data_table(:, numericColumns));
data(1, :) = []; % Remove the first row

%% Normalize the data
data = bsxfun(@rdivide, data, sum(data, 1)); % Normalize columns to sum to 1

data_min = min(data(:)); % Minimum value in the entire data matrix
data_max = max(data(:)); % Maximum value in the entire data matrix

minVal = data_min;
maxVal = data_max;
num_bins = size(data, 1); % Number of rows in your data

x = linspace(minVal, maxVal, num_bins); % Generate bin centers

%% Visualize the data
minY = 0;
maxY = .1;
YtickStep = .02;
indices = 1:3;
fontSize = 30;
lineWidth = 4;
axisValues = [minVal, maxVal, minY, maxY];

%% Build the cost matrix
M = abs(bsxfun(@minus, x', x)); % Compute pairwise distances
M = M / median(M(:));           % Normalize the cost matrix

%% Set the parameters of wasserstein_DL
options.stop = 1e-3;
options.verbose = 2;
options.D_step_stop = 5e-5;
options.lambda_step_stop = 5e-4;
options.alpha = 0.5;
options.Kmultiplication = 'symmetric';
options.GPU = 0;

gamma = 1/50;
wassersteinOrder = 1;

rho1 = 0;
rho2 = 0;

%% Set up the output folder
outputFolder = './decomposed_matrices_s_8_n_0.04/';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


num_repetitions = 80; % Adjust this number as needed

% Iterate over the range of k values (e.g., 4 to 20)
for k = 4:30
    fprintf('Processing k = %d...\n', k);
    
    for rep = 1:num_repetitions
        fprintf('  Repetition %d of %d...\n', rep, num_repetitions);
        
        % Perform Wasserstein NMF
        [D, lambda, objectives] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);
        
        % Generate unique filenames for each repetition
        D_filename = fullfile(outputFolder, sprintf('D_k%d_rep%d.csv', k, rep));
        lambda_filename = fullfile(outputFolder, sprintf('lambda_k%d_rep%d.csv', k, rep));
        
        % Save the matrices to CSV files
        writematrix(D, D_filename);
        writematrix(lambda, lambda_filename);
        
    end
end

%% Optionally, visualize the dictionaries after the loop (last value of k)
k = 20; % Last value of k
fprintf('Performing Wasserstein NMF for k = %d...\n', k);
[D, lambda, objectives] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);

% Plot the last set of objectives
figure;
plot(objectives);
xlabel('Number of outer iterations');
ylabel('Objective');

% Visualize the dictionary for the last k value
dictionaryLegendArray = cell(k, 1);
for i = 1:k
    dictionaryLegendArray{i} = ['$d_{', num2str(i), '}$'];
end
plotDictionary(x, D, axisValues, lineWidth, fontSize, YtickStep, [], dictionaryLegendArray, 'Wasserstein NMF');