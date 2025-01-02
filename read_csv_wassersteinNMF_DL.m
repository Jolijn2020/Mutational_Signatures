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



minVal=data_min;
maxVal=data_max;
num_bins = size(data, 1); % Number of rows in your data

x = linspace(minVal, maxVal, num_bins); % Generate bin centers

%% Visualize the data

minY=0;
maxY=.1;
YtickStep=.02;
indices=1:3;
fontSize=30;
lineWidth=4;
axisValues=[minVal,maxVal,minY,maxY];


%% Build the cost matrix
M = abs(bsxfun(@minus, x', x)); % Compute pairwise distances
M = M / median(M(:));           % Normalize the cost matrix

%% Set the parameters of wasserstein_DL

options.stop=1e-3;
options.verbose=2;
options.D_step_stop=5e-5;
options.lambda_step_stop=5e-4;
options.alpha=0.5;
options.Kmultiplication='symmetric';
options.GPU=0;
k=3;



dictionaryLegendArray=cell(numel(indices),1);
for i=1:k
    dictionaryLegendArray{i}=['$d_{',num2str(i),'}$'];
end

gamma=1/50;

wassersteinOrder=1;

%% Perform Wasserstein NMF
rho1=.1;
rho2=.1;
[D, lambda, objectives]=wasserstein_DL(data,k,M.^wassersteinOrder,gamma,rho1,rho2,options);

plot(objectives);
xlabel('Number of outer iterations')
ylabel('Objective')

%% Visualize the dictionary


plotDictionary(x,D,axisValues,lineWidth,fontSize,YtickStep,[],dictionaryLegendArray,'Wasserstein NMF')

%% Perform Wasserstein DL

options.alpha=0;
options.D_step_stop=1e-7;
options.lambda_step_stop=1e-7;
[D_DL, lambda_DL, objectives]=wasserstein_DL(data,k,M.^wassersteinOrder,gamma,0,0,options);
plot(objectives);
xlabel('Number of outer iterations')
ylabel('Objective')

%% Visualize the dictionary

axisValues(3)=floor(min(D_DL(:)*100))/100;

plotDictionary(x,D_DL,axisValues,lineWidth,fontSize,YtickStep,[],dictionaryLegendArray,'Wasserstein DL')


%% Compare data and reconstruction

width=1200;
height=600;
figure('Position',[1 1 width height])

axisValues(3)=0;
minY=0;
i=1;

subplot(1,2,1)
plotDictionary(x,data(:,i),axisValues,lineWidth,fontSize,YtickStep,[],'x','Data')

subplot(1,2,2)
plotDictionary(x,[D*lambda(:,i),D_DL*lambda_DL(:,i)],axisValues,lineWidth,fontSize,YtickStep,[],{'NMF reconstruction','DL reconstruction'},'Reconstruction')