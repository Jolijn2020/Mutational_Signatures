function[ all_lambda_init] = Pipeline(params)
    
    fields = fieldnames(params);
    for i = 1:numel(fields)
        eval([fields{i}, ' = params.(fields{i});'])
    end
 
    if exist ('M_table', 'var') && ~isempty(M_table)
        M_table = readtable(m_path, 'ReadVariableNames', true, 'ReadRowNames', true, 'FileType', 'text', VariableNamingRule='preserve' );
        M = table2array(M_table);
        disp(['Read M from: ', m_path]);
    end

    date_str = datestr(datetime('now'), 'yyyymmdd_HHMMss');
    if exist('experiment_name', 'var')
        prefix = ['Results_', date_str, '_', experiment_name, '_'];
    else
        prefix = ['Results_', date_str, '_'];
    end
    % Dynamically determine output file path
    [parent_dir, file_name, ext] = fileparts(file_path); % Extract path components
    output_dir = fullfile(parent_dir, [prefix, file_name, '.json']); % Modify filename

    % Define options struct for both WNMF and KLNMF
    options.stop = stop;
    options.verbose = verb;
    options.D_step_stop = Dss;
    options.lambda_step_stop = lss;
    options.alpha = Alpha;
    options.Kmultiplication = Km;
    options.GPU = GPU;

    % Process the data
    [data, maxVal, minVal, x] = processData(file_path);

    % Generate cost matrix M
    M = generate_cost_matrix_M(x);

    % Perform Wasserstein NMF
    [ all_H_lambda_init, D, lambda, objectives_wnmf, reconstructed_data, all_Dw, all_lambdaw, all_reconstructed_dataw, all_D_init, all_lambda_init] = perform_WNMF(data, M, num_iter, ...
        stop, verb, Dss, lss, Alpha, Km, GPU, k, Gamma, wO, rho1, rho2);

    % Plot convergence of objectives for Wasserstein NMF
    if visu
        plotConvergence(objectives_wnmf);
    end

    % Perform KL NMF with shared initialization and replicates
    [D_nmf, lambda_nmf, objectives_klnmf, reconstructed_data_nmf, all_Dkl, ...
    all_lambdakl, all_reconstructed_datakl] = perform_KLNMF(data, k, num_iter, ...
    all_D_init, all_lambda_init, stop, verb);


    to_export = struct();
    to_export.D = D;
    to_export.lambda = lambda;
    to_export.D_nmf = D_nmf;
    to_export.lambda_nmf = lambda_nmf;
    to_export.objectives_wnmf = objectives_wnmf;
    to_export.objectives_klnmf = objectives_klnmf;
    to_export.options = options;
    to_export.params = params;
    to_export.all_Dw = all_Dw;
    to_export.all_lambdaw = all_lambdaw;
    to_export.all_reconstructed_dataw = all_reconstructed_dataw;
    to_export.all_Dkl = all_Dkl;
    to_export.all_lambdakl = all_lambdakl;
    to_export.all_reconstructed_datakl = all_reconstructed_datakl;
    to_export.reconstructed_data = reconstructed_data;
    to_export.reconstructed_data_nmf = reconstructed_data_nmf;  

    % Export D, lambda, D_nmf, lambda_nmf, objectives, and options to json
    exportStruct(to_export, output_dir)


    % Visualize and compare original, Wasserstein, and KL reconstructions
    if visu
        sample_idx = 1; % Choose a sample to visualize
        plotComparaison_KL_W(x, data, minVal, maxVal, sample_idx, reconstructed_data, reconstructed_data_nmf);
    end

    if visu
        % Visualize learned dictionary and coefficient matrices
        visualize_D_and_lambda(D, lambda);
    end
    disp(all_Dw)
end


function [data, maxVal, minVal, x] = processData(file_path)
    % processData imports a CSV file, processes the data, and returns the results

    % Read the CSV file
    data_table = readtable(file_path, 'ReadVariableNames', true, 'ReadRowNames', true, VariableNamingRule='preserve'); 
    
    % Convert the table to a numeric matrix
    data = table2array(data_table);
    
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

function [all_H_lambda_init, best_D, best_lambda, all_objectives, best_reconstructed_data, all_D, ...
    all_lambda, all_reconstructed_data, all_D_init, all_lambda_init] = perform_WNMF(data, ...
    M, num_iter, stop, verb, Dss, lss, Alpha, Km, GPU, k, Gamma, wO, rho1, rho2)

    % Initialize the best results
    best_objective_value = Inf; % Start with the worst possible objective value
    best_D = [];
    best_lambda = [];
    all_objectives = cell(1, num_iter); % Store objectives for each iteration
    best_reconstructed_data = [];
    all_D = cell(1, num_iter); % Store all D matrices
    all_D_init = cell(1, num_iter); % Store all D_init matrices
    all_lambda = cell(1, num_iter); % Store all lambda matrices
    all_lambda_init = cell(1, num_iter); % Store all lambda matrices
    all_H_lambda_init = cell(1, num_iter); 
    all_reconstructed_data = cell(1, num_iter); % Store all reconstructed data matrices

    % Check that the number of provided initializations matches the number of iterations
    % if length(D_init_array) ~= num_iter || length(lambda_init_array) ~= num_iter
    %     error('The number of initial \( D \) and \( \lambda \) matrices must match num_iter.');
    % end

    % Loop for num_iter iterations
    for iter = 1:num_iter
        % Define optimization options
        options.stop = stop;
        options.verbose = verb;
        options.D_step_stop = Dss;
        options.lambda_step_stop = lss;
        options.alpha = Alpha;
        options.Kmultiplication = Km;
        options.GPU = GPU;

        % 
        % % Use the provided initializations for this iteration
        % D_init = D_init_array{iter};
        % lambda_init = lambda_init_array{iter};

        % Perform Wasserstein NMF
        fprintf('Performing Wasserstein NMF iteration %d...\n', iter);
        [D, lambda, objectives, HD, Hlambda, D_init, HD_init, hlambda_init, lambda_init] = wasserstein_DL(data, k, M.^wO, Gamma, rho1, rho2, options);
        reconstructed_data = D * lambda;

        % Save objectives for this iteration
        all_objectives{iter} = objectives;

        % Save the D, lambda, and reconstructed data, init matrices for this iteration
        all_D{iter} = D;
        all_lambda{iter} = lambda;
        all_reconstructed_data{iter} = reconstructed_data;
        all_D_init{iter} = D_init;
        all_lambda_init{iter} = lambda_init;
        all_H_lambda_init{iter} = hlambda_init; 


        % Check if this iteration's final objective is better (smaller)
        final_objective = objectives(end); % Objective value at the last step of this iteration
        if final_objective < best_objective_value
            best_objective_value = final_objective; % Update the best objective value
            best_D = D; % Update best dictionary
            best_lambda = lambda; % Update best coefficients
            best_reconstructed_data = reconstructed_data; % Update best reconstruction
        end
    end

    % Return the best results and all objectives
    fprintf('Best objective value: %f\n', best_objective_value);
end


function plotConvergence(all_objectives)
    % plotConvergence: Plots the objective values to check convergence for all iterations
    %
    % Input:
    %   all_objectives - Cell array where each cell contains the objectives for one iteration

    figure;
    hold on;

    % Iterate over all objectives and plot them
    for i = 1:length(all_objectives)
        plot(all_objectives{i}, 'LineWidth', 1.5, 'DisplayName', ['Iteration ' num2str(i)]);
    end

    xlabel('Number of Outer Iterations', 'FontSize', 12);
    ylabel('Objective Function', 'FontSize', 12);
    title('Wasserstein NMF Objective Convergence (All Iterations)', 'FontSize', 14);
    legend show; % Display a legend for each iteration
    grid on; % Add a grid for better visualization
    hold off;
end

function [D_best, lambda_best, all_objectives_klnmf, ...
reconstructed_data_best, all_D, all_lambda, all_reconstructed_data] = perform_KLNMF(data, k, num_iter, ...
D_init_array, lambda_init_array, stop, verbose)
    % Perform KL NMF with multiple replicates and custom shared parameters
   

    % Validate input sizes
    if length(D_init_array) ~= num_iter || length(lambda_init_array) ~= num_iter
        error('Number of initializations for D and lambda must match num_iter.');
    end

    fprintf('Performing Regular NMF with KL Divergence (with %d replicates)...\n', num_iter);

    % Initialize variables to track the best results
    best_error = inf; % Start with an infinite error
    D_best = [];
    lambda_best = [];
    all_objectives_klnmf = cell(1, num_iter); % Store objectives for each replicate
    reconstructed_data_best = [];
    all_D = cell(1, num_iter); % Store all D matrices
    all_lambda = cell(1, num_iter); % Store all lambda matrices
    all_reconstructed_data = cell(1, num_iter); % Store all reconstructed data matrices

    % Perform KL NMF for each replicate
    for i = 1:num_iter
        fprintf('Replicate %d...\n', i);

        % Use the provided initializations for this replicate
        D_init = D_init_array{i};
        lambda_init = lambda_init_array{i};
        size(lambda_init)

        % Initialize objectives array for this replicate
        iteration_objectives = [];

        % Perform NNMF with the provided initialization
        options = statset('MaxIter', 100, 'TolFun', stop, 'Display', 'final');
        
        [D, lambda] = nnmf(data, k, 'algorithm', 'mult', 'w0', D_init, 'h0', lambda_init, 'options', options);


        % Reconstruct the data and track the objective (reconstruction error) at each step
        reconstructed_data = D * lambda;
        reconstruction_error = norm(data - reconstructed_data, 'fro');
        iteration_objectives = [iteration_objectives, reconstruction_error];

        % Save objectives, D, lambda, and reconstructed data for this replicate
        all_objectives_klnmf{i} = iteration_objectives;
        all_D{i} = D;
        all_lambda{i} = lambda;
        all_reconstructed_data{i} = reconstructed_data;

        % Update the best result if the current replicate is better
        if reconstruction_error < best_error
            best_error = reconstruction_error;
            D_best = D;
            lambda_best = lambda;
            reconstructed_data_best = reconstructed_data;
        end
    end

    fprintf('Best reconstruction error: %.4f\n', best_error);
end



function [D, lambda, objectives] = nnmf_with_objectives(data, k, D_init, lambda_init)
    % Perform NNMF and track objectives
    options = statset('MaxIter', 100); % Adjust max iterations as needed
    [D, lambda] = nnmf(data, k, 'w0', D_init, 'h0', lambda_init, 'options', options);
    objectives = []; % NNMF does not track objectives by default; calculate manually if needed
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

function exportStruct(astruct, output_file)
    jsonStr = jsonencode(astruct, PrettyPrint=true);
    fid = fopen(output_file, 'w');
    disp(['Writing to ', output_file]);
    fwrite(fid, jsonStr, 'char');
    fclose(fid);
end

function exportMatricesToCSV(D, lambda, D_nmf, lambda_nmf, objectives_wnmf, objectives_klnmf, options, all_Dw, all_lambdaw, all_reconstructed_dataw, all_Dkl, all_lambdakl, all_reconstructed_datakl, reconstructed_data_best_w, reconstructed_data_best_kl, output_file)
    % Open the file for writing
    fid = fopen(output_file, 'w');
    if fid == -1
        error('Cannot open file %s for writing.', output_file);
    end

    % 1. Write Options
    fprintf(fid, 'Options:\n');
    fields = fieldnames(options);
    for i = 1:numel(fields)
        field = fields{i};
        value = options.(field);
        if isnumeric(value)
            value_str = num2str(value); % Convert numeric values to string
        elseif ischar(value)
            value_str = value; % Keep strings as is
        else
            value_str = 'N/A'; % For unsupported types
        end
        fprintf(fid, '%s: %s\n', field, value_str);
    end

    % 2. Write Best D for Wasserstein NMF
    fprintf(fid, '\nWasserstein NMF Best (D)\n');
    fclose(fid);
    writematrix(D, output_file, 'WriteMode', 'append');

    % 3. Write All D for Wasserstein NMF
    fid = fopen(output_file, 'a');
    for i = 1:length(all_Dw)
        fprintf(fid, '\nWasserstein NMF (D) Iteration %d\n', i);
        fclose(fid);
        writematrix(all_Dw{i}, output_file, 'WriteMode', 'append');
        fid = fopen(output_file, 'a');
    end
    fclose(fid);

    % 4. Write Best Lambda for Wasserstein NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nWasserstein NMF Best(lambda)\n');
    fclose(fid);
    writematrix(lambda, output_file, 'WriteMode', 'append');

    % 5. Write All Lambda for Wasserstein NMF
    fid = fopen(output_file, 'a');
    for i = 1:length(all_lambdaw)
        fprintf(fid, '\nWasserstein NMF (lambda) Iteration %d\n', i);
        fclose(fid);
        writematrix(all_lambdaw{i}, output_file, 'WriteMode', 'append');
        fid = fopen(output_file, 'a');
    end
    fclose(fid);

    % 6. Write Best Reconstructed Data for Wasserstein NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nWasserstein NMF Best Reconstructed Data\n');
    fclose(fid);
    writematrix(reconstructed_data_best_w, output_file, 'WriteMode', 'append');

    % 7. Write All Reconstructed Data for Wasserstein NMF
    fid = fopen(output_file, 'a');
    for i = 1:length(all_reconstructed_dataw)
        fprintf(fid, '\nWasserstein NMF Reconstructed Data Iteration %d\n', i);
        fclose(fid);
        writematrix(all_reconstructed_dataw{i}, output_file, 'WriteMode', 'append');
        fid = fopen(output_file, 'a');
    end
    fclose(fid);

    % 8. Write Objectives for Wasserstein NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nWasserstein NMF Objectives\n');
    fclose(fid);
    fid = fopen(output_file, 'a');
    for i = 1:length(objectives_wnmf)
        fprintf(fid, 'Iteration %d: %s\n', i, num2str(objectives_wnmf{i}));
    end
    fclose(fid);

    % 9. Write Best D for KL NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nKL NMF Best (D)\n');
    fclose(fid);
    writematrix(D_nmf, output_file, 'WriteMode', 'append');

    % 10. Write All D for KL NMF
    fid = fopen(output_file, 'a');
    for i = 1:length(all_Dkl)
        fprintf(fid, '\nKL NMF (D) Iteration %d\n', i);
        fclose(fid);
        writematrix(all_Dkl{i}, output_file, 'WriteMode', 'append');
        fid = fopen(output_file, 'a');
    end
    fclose(fid);

    % 11. Write Best Lambda for KL NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nKL NMF Best (lambda)\n');
    fclose(fid);
    writematrix(lambda_nmf, output_file, 'WriteMode', 'append');

    % 12. Write All Lambda for KL NMF
    fid = fopen(output_file, 'a');
    for i = 1:length(all_lambdakl)
        fprintf(fid, '\nKL NMF (lambda) Iteration %d\n', i);
        fclose(fid);
        writematrix(all_lambdakl{i}, output_file, 'WriteMode', 'append');
        fid = fopen(output_file, 'a');
    end
    fclose(fid);

    % 13. Write Best Reconstructed Data for KL NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nKL NMF Best Reconstructed Data\n');
    fclose(fid);
    writematrix(reconstructed_data_best_kl, output_file, 'WriteMode', 'append');

    % 14. Write All Reconstructed Data for KL NMF
    fid = fopen(output_file, 'a');
    for i = 1:length(all_reconstructed_datakl)
        fprintf(fid, '\nKL NMF Reconstructed Data Iteration %d\n', i);
        fclose(fid);
        writematrix(all_reconstructed_datakl{i}, output_file, 'WriteMode', 'append');
        fid = fopen(output_file, 'a');
    end
    fclose(fid);

    % 15. Write Objectives for KL NMF
    fid = fopen(output_file, 'a');
    fprintf(fid, '\nKL NMF Objectives\n');
    fclose(fid);
    fid = fopen(output_file, 'a');
    for i = 1:length(objectives_klnmf)
        fprintf(fid, 'Iteration %d: %s\n', i, num2str(objectives_klnmf{i}));
    end
    fclose(fid);

    fprintf('All matrices, objectives, and options exported to %s\n', output_file);
end
