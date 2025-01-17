%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Script to run multiple Wasserstein DL runs and store dictionary matrices
%% along with their objective values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define folder containing simulated data
folderPath = "simulated_data_01_16";
options.stop = 1e-3;
options.verbose = 1;
options.D_step_stop = 7e-4;
options.lambda_step_stop = 4e-4;
rho1 = 0.03;
rho2 = 0.03;
nRuns = 30; % Number of runs    
k_range = 28; % Set the maximum number of components

% Get a list of all CSV files in the folder
fileList = dir(fullfile(folderPath, '*.csv'));

% Loop through each file
for pathIdx = 1:length(fileList)
    % Get the file name
    fileName = fileList(pathIdx).name;
    filePath = fullfile(folderPath, fileName);
    
    fprintf('Processing file: %s\n', fileName);
    
    % Read the CSV file
    data_table = readtable(filePath, 'PreserveVariableNames', true);
    
    % Remove the first column and row if they are metadata/headers
    data_table(:, 1) = [];
    data_table(1, :) = [];
    
    % Convert to numeric array
    data = table2array(data_table);
    
    % Normalize each column to sum to 1
    data = bsxfun(@rdivide, data, sum(data, 1));
    
    % Gather min and max of the data
    data_min = min(data(:)); 
    data_max = max(data(:));
    
    % Create bin centers
    num_bins = size(data, 1);
    x = linspace(data_min, data_max, num_bins);
    
    %% (1) Construct cost matrix M
    M = abs(bsxfun(@minus, x', x)); % Pairwise distances
    M = M / median(M(:));           % Normalize the cost matrix
    
    %% (2) Set the parameters for Wasserstein_DL
    options.alpha = 0.5;
    options.Kmultiplication = 'symmetric';
    options.GPU = 0;
    gamma = 1 / 50;
    wassersteinOrder = 1;
    
    %% (3) Define output directory
    % Extract the file name (without extension) for directory creation
    [~, baseFileName, ~] = fileparts(fileName);
    
    % Define the output directory for storing results
    output_dir = fullfile(pwd, 'outputs/signature_matrices/16_01', baseFileName);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    %% (4) Initialize a table to store objective values
    objectiveTable = table([], [], [], 'VariableNames', {'k', 'RunIndex', 'FinalObjective'});
    
    %% (5) Perform Wasserstein DL for multiple runs and save results
    for k = 2:k_range
        for runIdx = 1:nRuns
            fprintf('-------Running Wasserstein DL: k = %d, Run = %d\n--------', k, runIdx);
            
            % Perform Wasserstein NMF (DL)
            try
                [D_tmp, lambda_tmp, objectives_tmp] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);
                
                % Save the results for this run
                output_path_D = fullfile(output_dir, sprintf('D_k%d_run%d.csv', k, runIdx));
                output_path_lambda = fullfile(output_dir, sprintf('lambda_k%d_run%d.csv', k, runIdx));
                
                writematrix(D_tmp, output_path_D);
                writematrix(lambda_tmp, output_path_lambda);
                
                % Record the final objective value
                finalObjective = objectives_tmp(end);
                newRow = {k, runIdx, finalObjective};
                objectiveTable = [objectiveTable; newRow];
            catch ME
                % Handle any errors in the Wasserstein DL run
                fprintf('Error during run %d for k = %d: %s\n', runIdx, k, ME.message);
            end
        end
    end
    
    %% (6) Save the objective values table to a CSV file
    objectiveFilePath = fullfile(output_dir, 'objective_values.csv');
    writetable(objectiveTable, objectiveFilePath);
end

fprintf('\nAll Wasserstein DL runs are complete, and results have been saved.\n');