params = struct();
params.num_iter = 10;
params.visu = 0;
params.verb = 1;
params.Alpha = 0.05;
params.Km = 'symmetric';
params.GPU = 0;
params.k = 8;
params.Gamma = 1/50;
params.wO = 1;
params.rho1 = 0.1;
params.rho2 = 0.1;
params.stop = 1e-3;
params.Dss = 0.0007;
params.lss = 0.0004;
transport_plans = {'';
    'distances/distances_uniform.tsv';
    'distances/distances_overall.tsv';
    'distances/distances_hamming.tsv'};

% transport_plans = {''};


folder = 'simulated_data_01_17';

csvFiles = dir(fullfile(folder, 's*.csv'));

% Loop through each file
for i = 1:length(csvFiles)
    filePath = fullfile(folder, csvFiles(i).name);
    params.file_path = filePath;
    matchedStrings = regexp(filename, '\<s_(\d+)', 'tokens');
    params.k = str2double(matchedStrings{1});

    for i = 1:length(transport_plans)
        plan = transport_plans{i};
        disp(plan);
        if ~isempty(plan)

            [folderPath, fileName, fileExt] = fileparts(plan);
            parts = split(fileName, '_');
            id = parts(end);
        else
            id='smooth'
        end
        params.experiment_name = id;
        params.M_path = plan;

        disp(params);
        tic;
        results = Pipeline(params)
        toc;
    end
end
