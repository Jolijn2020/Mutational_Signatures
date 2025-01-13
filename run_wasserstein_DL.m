%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Script to run multiple Wasserstein DL runs, apply RTOL filtering, 
%  and store dictionary matrices that pass the filter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% (1) Import the simulated data from a CSV file
simulated_data_path = '/Users/lorispodevyn/Documents/COURSES/PROJECT_MASTER_1/Code/Mutational_Signatures/simulated_data/s_8_n_0.02_GRCh37_17b_86_98_39_22a_43_17a_13.csv';
data_table = readtable(simulated_data_path, 'PreserveVariableNames', true);  % Read the CSV file

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

minVal = data_min;
maxVal = data_max;
num_bins = size(data, 1);

x = linspace(minVal, maxVal, num_bins); % Generate bin centers

%% (2) Construct cost matrix M
M = abs(bsxfun(@minus, x', x)); % pairwise distances
M = M / median(M(:));          % normalize the cost matrix

%% (3) Set the parameters of wasserstein_DL
options.stop = 1e-3;
options.verbose = 2;
options.D_step_stop = 5e-5;
options.lambda_step_stop = 5e-4;
options.alpha = 0.5;
options.Kmultiplication = 'symmetric';
options.GPU = 0;
k = 8;
gamma = 1/50;
wassersteinOrder = 1;
rho1 = 0.1;
rho2 = 0.1;

%% (4) For demonstration, define plotting variables
minY = 0;
maxY = 0.1;
YtickStep = 0.02;
indices = 1:3;
fontSize = 30;
lineWidth = 4;
axisValues = [minVal, maxVal, minY, maxY];

dictionaryLegendArray = cell(k,1);
for i = 1:k
    dictionaryLegendArray{i} = ['$d_{', num2str(i),'}$'];
end

%% (5) Create folders for passing and failing runs
output_dir_passing = fullfile(pwd, 'signature_matrices_passing');
output_dir_failing = fullfile(pwd, 'signature_matrices_failing');
if ~exist(output_dir_passing, 'dir')
    mkdir(output_dir_passing);
end
if ~exist(output_dir_failing, 'dir')
    mkdir(output_dir_failing);
end

%% (6) Define multiple runs and RTOL threshold
nRuns = 10;               % number of runs
rtol_threshold = 0.0005;   % relative tolerance threshold

% We will store results in these arrays/cells
allD = cell(nRuns,1);
allLambda = cell(nRuns,1);
allObjectives = cell(nRuns,1);
passed_runs_idx = [];
best_objective = Inf;    % Initialize best objective to infinity

%% (7) Loop over multiple runs
for runIdx = 1:nRuns
    
    fprintf('\n=== Starting run #%d ===\n', runIdx);
    
    % Perform Wasserstein NMF (DL) with the chosen parameters
    [D_tmp, lambda_tmp, objectives_tmp] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);
    
    % Store the results for this run
    allD{runIdx} = D_tmp;
    allLambda{runIdx} = lambda_tmp;
    allObjectives{runIdx} = objectives_tmp;
    
    % Update the best objective value
    best_objective = min(best_objective, min(objectives_tmp));
end

%% (8) RTOL filtering
fprintf('\n=== Applying RTOL filtering ===\n');
for runIdx = 1:nRuns
    % Get the last objective value for this run
    obj_last = allObjectives{runIdx}(end);
    
    % Compute RTOL
    rtol_measure = abs(obj_last - best_objective) / abs(best_objective);
    fprintf('RTOL for run #%d = %.4e\n', runIdx, rtol_measure);
    
    % Check if this run passes the RTOL threshold
    if rtol_measure <= rtol_threshold
        fprintf('Run #%d PASSED the filter!\n', runIdx);
        passed_runs_idx = [passed_runs_idx, runIdx];
        
        % Save the dictionary and lambda for this run in the "passing" folder
        output_path_D = fullfile(output_dir_passing, sprintf('learned_dictionary_D_run%d.csv', runIdx));
    else
        fprintf('Run #%d FAILED the filter.\n', runIdx);
        
        % Save the dictionary and lambda for this run in the "failing" folder
        output_path_D = fullfile(output_dir_failing, sprintf('learned_dictionary_D_run%d.csv', runIdx));
    end
    
    % Write matrices to the appropriate folder
    writematrix(allD{runIdx}, output_path_D);
end

%% (9) Summary of results
fprintf('\n=== Summary ===\n');
fprintf('Total runs: %d\n', nRuns);
fprintf('Best objective value: %.4e\n', best_objective);
fprintf('Number of runs passing the RTOL filter: %d\n', numel(passed_runs_idx));
fprintf('Runs that passed: %s\n', mat2str(passed_runs_idx));

%% (10) Visualize objective curves for all runs
figure('Name', 'Objective Curves for All Runs', 'NumberTitle', 'off');
hold on;
for runIdx = 1:nRuns
    plot(allObjectives{runIdx}, 'DisplayName', sprintf('Run #%d', runIdx), 'LineWidth', 2);
end
xlabel('Number of iterations');
ylabel('Objective');
title('Objective Curves for All Runs');
legend('show');
grid on;
hold off;

%% (11) Visualize dictionary for a selected passing run (if any)
if ~isempty(passed_runs_idx)
    runToShow = passed_runs_idx(1); % Select the first passing run for visualization
    
    D_pass = allD{runToShow};
    lambda_pass = allLambda{runToShow};
    
    % Plot the dictionary
    figure('Name', ['Dictionary for Run #', num2str(runToShow)], 'NumberTitle', 'off');
    for i = 1:size(D_pass, 2)
        subplot(1, size(D_pass, 2), i);
        plot(x, D_pass(:, i), 'LineWidth', 2);
        title(['Component ', num2str(i)]);
        xlabel('Mutation Categories');
        ylabel('Values');
    end
end

fprintf('\nRTOL-filtered results have been saved in the respective folders:\n');
fprintf('  Passing runs: %s\n', output_dir_passing);
fprintf('  Failing runs: %s\n', output_dir_failing);
