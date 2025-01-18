function Pipeline_test_2(file_path)
    % Process the data
    [data, maxVal, minVal, x] = processData(file_path);
    
    % Generate cost matrix M
    M = generate_cost_matrix_M(x);
    
    % Wasserstein NMF parameters
    options.stop = 1e-8;
    options.verbose = 2;
    options.D_step_stop = 5e-5;
    options.lambda_step_stop = 5e-4;
    options.alpha = 0.05;
    options.Kmultiplication = 'symmetric';
    options.GPU = 0;
    k = 5; % Number of components (dictionary size)
    gamma = 1 / 50;
    wassersteinOrder = 1;
    rho1 = 0.0;
    rho2 = 0.0;

    % Perform Wasserstein NMF
    [D, lambda, objectives, reconstructed_data] = perform_WNMF(data, M, ...
        options.stop, options.verbose, options.D_step_stop, ...
        options.lambda_step_stop, options.alpha, ...
        options.Kmultiplication, options.GPU, k, gamma, ...
        wassersteinOrder, rho1, rho2);

    % Plot convergence of objectives
    plotConvergence(objectives);

    % Perform KL NMF
    [D_nmf, lambda_nmf, reconstructed_data_nmf] = perform_KLNMF(data, k);

    % Visualize and compare original, Wasserstein, and KL reconstructions
    sample_idx = 1; % Choose a sample to visualize
    plotComparaison_KL_W(x, data, minVal, maxVal, sample_idx, reconstructed_data, reconstructed_data_nmf);

    % Visualize learned dictionary and coefficient matrices
    visualize_D_and_lambda(D, lambda);
end


function [data, maxVal, minVal, x] = processData(file_path)
    % processData imports a CSV file, processes the data, and returns the results

    % Read the CSV file
    data_table = readtable(file_path); 
    
    % Convert the table to a numeric matrix
    data = table2array(data_table);
    data(1, :) = []; % Remove the first row
    data(:, 1) = []; % Remove the first column
    
    % Normalize the data
    data = bsxfun(@rdivide, data, sum(data, 1)); % Normalize columns to sum to 1
    
    % Calculate minimum and maximum values
    minVal = min(data(:)); % Minimum value in the entire data matrix
    maxVal = max(data(:)); % Maximum value in the entire data matrix
    
    % Generate bin centers
    num_bins = size(data, 1); % Number of rows in the data
    x = linspace(minVal, maxVal, num_bins); % Generate bin centers
end

function [M] = generate_cost_matrix_M(x)
    % Generate the cost Matrix M
    M = abs(bsxfun(@minus, x', x)); % Compute pairwise distances
    M = M / median(M(:));           % Normalize the cost matrix
end

function [D, lambda, objectives, reconstructed_data] = perform_WNMF(data, M, stop, verbose, D_step_stop, lambda_step_stop, alpha, Kmultiplication, GPU, k, gamma, wassersteinOrder, rho1, rho2)
    % Perform Wasserstein NMF

    % Define optimization options
    options.stop = stop;
    options.verbose = verbose;
    options.D_step_stop = D_step_stop;
    options.lambda_step_stop = lambda_step_stop;
    options.alpha = alpha;
    options.Kmultiplication = Kmultiplication;
    options.GPU = GPU;

    % Perform Wasserstein NMF
    fprintf('Performing Wasserstein NMF...\n');
    [D, lambda, objectives] = wasserstein_DL(data, k, M.^wassersteinOrder, gamma, rho1, rho2, options);
    reconstructed_data = D * lambda;
end

function plotConvergence(objectives)
    % plotConvergence Plots the objective values to check convergence
    figure;
    plot(objectives, 'LineWidth', 1.5);
    xlabel('Number of outer iterations', 'FontSize', 12);
    ylabel('Objective Function', 'FontSize', 12);
    title('Wasserstein NMF Objective Convergence', 'FontSize', 14);
    grid on; % Add grid for better visualization
end

function [D_nmf, lambda_nmf, reconstructed_data_nmf] = perform_KLNMF(data, k)
    % Perform KL NMF
    fprintf('Performing Regular NMF with KL Divergence...\n');
    [D_nmf, lambda_nmf] = nnmf(data, k, 'algorithm', 'mult', 'replicates', 5);
    reconstructed_data_nmf = D_nmf * lambda_nmf;
end

function plotComparaison_KL_W(x, data, minVal, maxVal, sample_idx, reconstructed_data, reconstructed_data_nmf)
    % Visualize Original, Wasserstein NMF, and KL NMF reconstructions
    figure;
    subplot(1, 3, 1);
    plotDictionary(x, data(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Original Data');
    title('Original Data');

    subplot(1, 3, 2);
    plotDictionary(x, reconstructed_data(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Wasserstein NMF Reconstruction');
    title('W NMF Reconstruction');

    subplot(1, 3, 3);
    plotDictionary(x, reconstructed_data_nmf(:, sample_idx), [minVal, maxVal, 0, 0.1], 4, 30, 0.02, [], 'x', 'Regular NMF Reconstruction');
    title('KL NMF Reconstruction');
end

function [] = visualize_D_and_lambda(D, lambda)
    % Visualize the learned dictionary and coefficient matrices
    figure;
    subplot(1, 2, 1);
    imagesc(D);  % Display matrix D as an image
    colormap(parula);
    colorbar;
    title('Learned Dictionary Matrix D');
    xlabel('Components');
    ylabel('Data Dimensions');

    subplot(1, 2, 2);
    imagesc(lambda);  % Display matrix lambda as an image
    colormap(parula);
    colorbar;
    title('Coefficient Matrix Lambda');
    xlabel('Data Samples');
    ylabel('Components');
end
