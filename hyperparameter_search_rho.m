s = 1e-3;
Dss = 0.0007;
lss = 0.0004;



%and then rho
rho1 = [3e-1 2e-1 linspace(1e-1, 1e-5, 10)];
rho2 = [3e-1 2e-1 linspace(1e-1, 1e-5, 10)];
smallest_error = inf;
errors = zeros(1,3);

params=struct();

params.file_path = 'simulated_data_01_14/s_8_n_0.02_GRCh37_10a_56_10d_52_36_91_45_38.csv';
params.num_iter = 1;
params.visu = 0;
params.verb = 1;
params.Alpha = 0.05;
params.Km = 'symmetric';
params.GPU = 0;
params.k = 8;
params.Gamma= 1/50;
params.wO = 1;
params.experiment_name = 'hyperparameter_search_rho';
params.M_path = 'distances/hamming_distances.tsv';
params.stop = s;
params.Dss = Dss;
params.lss = lss;


% Loop through all combinations
for r1 = rho1
    for r2 = rho2

        [rowExists, rowIndex] = ismember([r1 r2], errors(:,1:2), 'rows');
        if ~rowExists
            disp("new condition");
            params.rho1 = r1;
            params.rho2 = r1;

            disp([s, Dss, lss]);

            % disp(params);

            tic;
            results = Pipeline(params);
            toc;

            results.params.Kmultiplication = 'symmetric';
            K=exp(-results.transport_plan/results.params.Gamma);
            [multiplyK, multiplyKt]=buildMultipliers(K,results.params.Gamma,results.params,size(results.source_data));
            pX=matrixEntropy(results.source_data);


            % Evaluate the model
            error = computeWassersteinLegendre(results.source_data,results.H,results.params.Gamma,pX,multiplyK,multiplyKt)


            % Update the best hyperparameters
            errors = [errors ; s, Dss, lss, error];

            if error < smallest_error
                smallest_error = error
                best_params = [s Dss lss]
                % Create a structure
            end
        end
    end
end

disp(['Best Params: ', mat2str(best_params)]);
