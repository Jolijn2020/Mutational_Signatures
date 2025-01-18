% Define the hyperparameter ranges
% Do two rounds, for these params
% stop = [1e-3, 1e-4, 1e-5];
params=struct();

% stop = [1e-2 1e-3 1e-4];
stop = [1e-3 1e-4];

D_step_stop = [ linspace(1e-3, 1e-5, 10) ] % 1.0000e-04    1.0000e-05  1e-6 1e-7];
% D_step_stop = [1.0000e-03   9.0000e-04   8.0000e-04   7.0000e-04  ...
%    6.0000e-04   5.0000e-04   4.0000e-04   3.0000e-04 ...
%    2.0000e-04   1.0000e-04   1.0000e-04   9.0000e-05 ...
%    8.0000e-05   7.0000e-05   6.0000e-05   5.0000e-05 ...
%    4.0000e-05   3.0000e-05   2.0000e-05   1.0000e-05 ];
lambda_step_stop = [  linspace(1e-3, 1e-5, 10)];
% lambda_step_stop = [1.0000e-03   9.0000e-04   8.0000e-04   7.0000e-04  ...
%    6.0000e-04   5.0000e-04   4.0000e-04   3.0000e-04 ...
%    2.0000e-04   1.0000e-04   1.0000e-04   9.0000e-05 ...
%    8.0000e-05   7.0000e-05   6.0000e-05   5.0000e-05 ...
%    4.0000e-05   3.0000e-05   2.0000e-05   1.0000e-05 ];

forbidden = [1.0000e-03   2.0000e-04   8.0000e-04;
            1.0000e-03   1.0000e-04   8.0000e-04];


%and then rho
rho1 = [0 linspace(1e-1, 1e-2, 10)];
rho2 = [0 linspace(1e-1, 1e-2, 10)];
smallest_error = inf;
% errors = [];
%
% options.verbose = 2;
%   % options.stop = 1e-8;
%   % options.D_step_stop = 5e-5;
%   % options.lambda_step_stop = 5e-4;
%   options.stop=5e-3;
%   options.verbose=2;
%   options.D_step_stop=1e-4%3e-5; % 4e-5
%   options.lambda_step_stop=1e-3%3e-4; %4e-4
%   options.Kmultiplication='symmetric';
%   options.GPU=0;
%
%   % options.alpha = 0.05;
%   options.alpha = 0.05;
%
%   % k = 5; % Number of components (dictionary size)
%   gamma = 1 / 50;
%   wassersteinOrder = 1;
%   rho1 = .03;
%   rho2 = .03;

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
params.rho1 = 0.1;
% rho1 = 0;0
params.rho2 = 0.1;
% rho2 = 0;
params.experiment_name = 'with_hamming';
params.M_path = 'distances/hamming_distances.tsv';



% Loop through all combinations
for s = stop
    for Dss = D_step_stop
        for lss = lambda_step_stop
            [rowExists, rowIndex] = ismember([s Dss lss], errors(:,1:3), 'rows');
            if ~rowExists
                disp("new condition");
                disp([s, Dss, lss]);
                params.stop = s;
                params.Dss = Dss;
                params.lss = lss;
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
end

disp(['Best Params: ', mat2str(best_params)]);
