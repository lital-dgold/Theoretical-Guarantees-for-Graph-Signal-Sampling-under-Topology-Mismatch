clear all 
close all 
clc
set(groot, 'defaultFigurePosition', [400, 400, 560, 420]);
init_output_folder_path ='./Figures/power_data/';
addpath('./sampling_methods');
addpath('./auxiliary_functions');
addpath('./cost_functions');
addpath('./PGD_Sampling/');
addpath('./GreedySampling/');
addpath('./GreedySampling/efficient_update_function/');
addpath('./data_source_files');

% Parameters
n = 118;
number_samples = 10000;
weight_pseudo_data = 0.5;
sigma_pseudo_data = 0.1;
lpf_cutoff = ceil(n/2);

% Simulation parameters:
d = 10;
min_buses = 40;
number_of_measured_buses_array = round((n/100).*[100 : -d : min_buses]);
num_of_iterations_random_sampling = 100;

% Loading the graph data:
str = sprintf('ieee%dcdf.txt',n);    
[Y,~,v,~,~,~,~] = load_cdf(str);%read the data
Y = full(Y);
L = -imag(Y);
slack = 111;
v=v./v(slack);% Very important as there is no normalizing inside the functions
parameter = angle(v);
[V,S] = eig(L);
[V,~] = ensure_incresing_order(V,S);
V_k=V(:,1:lpf_cutoff);
x_0 = zeros(n, number_samples); % MEAN OF THE PARAMETER TO ESTIMATE
% Plotting constants
x_label_mse_vs_num_samples='\% measured nodes ($\tilde{q}$)';
x_data_mse_vs_num_samples = round(100.* number_of_measured_buses_array./n);
y_label_mse_vs_num_samples = 'MSE';
graph_type_mse_vs_num_samples = 1;
methods = getPlottingConstants();
selected_methods_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% MSE versus q %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
slack_idx = slack;
inv_prior = L;
filter = L;
sigma = 0.1;
mu = 0.1/sigma^2;
[data_vec] = create_non_bayseian_data(filter, parameter,sigma,number_samples);


% Call the appropriate sampling function for each method
sampling_sets = zeros(length(methods), n);
mse_data = zeros(length(methods), length(number_of_measured_buses_array));
for idx = selected_methods_idx
    disp(['Calculating for method: ' methods(idx).name]);
    [SamplingSet] = greedy_sampling_switch_method(V_k, L, filter, ...
        inv_prior, methods, idx, mu, sigma, slack_idx);
        % Evaluate MSE for this sampling set and store the result
    sampling_sets(idx, :) = SamplingSet;
end
%% Evaluate MSE for this sampling set and store the result 
for i = selected_methods_idx
    switch methods(i).name
        case 'Random'
           mse_data(i, :) = random_sampling(filter, data_vec, n, inv_prior, mu, ...
                            sigma,slack_idx, parameter, x_0, num_of_iterations_random_sampling, ...
                            number_of_measured_buses_array);
        otherwise
           mse_data(i,:) = evalute_sampling_set(number_of_measured_buses_array, sampling_sets(i, :), filter, data_vec, n,inv_prior,mu,sigma,slack_idx, parameter, x_0);
    end
end
selected_methods = methods(selected_methods_idx);
selected_methods_mse_data = mse_data(selected_methods_idx,:)';

output_folder_path = fullfile(init_output_folder_path, "error_vs_q");
save(fullfile(output_folder_path, "workspace.mat"))


% Pass the selected methods to the final_plot function
final_plot_struct(x_data_mse_vs_num_samples, x_label_mse_vs_num_samples, ...
    selected_methods_mse_data, y_label_mse_vs_num_samples, graph_type_mse_vs_num_samples, selected_methods);
fig_save_name = fullfile(output_folder_path, "mse");
saveas(gcf,fig_save_name + ".fig")
saveas(gcf,fig_save_name, 'epsc')

mean_norm_param = sum(parameter(:).^2);
final_plot_struct(x_data_mse_vs_num_samples, x_label_mse_vs_num_samples, ...
    selected_methods_mse_data./mean_norm_param, "NMSE", graph_type_mse_vs_num_samples, selected_methods);
fig_save_name = fullfile(output_folder_path, "nmse");
saveas(gcf,fig_save_name + ".fig")
saveas(gcf,fig_save_name, 'epsc')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% MSE versus snr %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set_size = round(0.7*n);
snr_values = 5:5:25;  % Define the range of SNR values (in dB)
mse_data = zeros(length(snr_values), length(methods));  

slack_idx = slack;
inv_prior = L;
filter = L;
for snr_idx = 1:length(snr_values)
    snr = snr_values(snr_idx);
    
    % Adjust noise level (sigma) based on SNR
    sigma = sqrt(10^(-snr / 10));  % Noise power is inversely related to SNR
    mu = 1/sigma^2;
   
    % Generate non-Bayesian data based on SNR-adjusted sigma
    [data_vec] = create_non_bayseian_data(filter, parameter,sigma,number_samples);
  
    % Call the appropriate sampling function for each method
    for idx = selected_methods_idx
        disp(['Calculating for method: ' methods(idx).name]);
        [SamplingSet] = greedy_sampling_switch_method(V_k, L, filter, ...
        inv_prior, methods, idx, mu, sigma, slack_idx, set_size);
        switch methods(idx).name
            case 'Random'
               mse_data(snr_idx, idx) = random_sampling(filter, data_vec, n, inv_prior, mu, ...
                sigma, slack_idx, parameter, x_0, num_of_iterations_random_sampling, ...
                set_size);
            otherwise
               mse_data(snr_idx, idx) = evalute_sampling_set(set_size, SamplingSet, filter, data_vec, n, inv_prior, mu, sigma, slack_idx, parameter, x_0);
        end
    end
end

selected_methods = methods(selected_methods_idx);
selected_methods_mse_data = mse_data(:,selected_methods_idx);

output_folder_path = fullfile(init_output_folder_path, "error_vs_snr");
save(fullfile(output_folder_path, "workspace.mat"))


% Pass the selected methods to the final_plot function
final_plot_struct(snr_values, 'SNR (dB)', selected_methods_mse_data, "MSE", graph_type_mse_vs_num_samples, selected_methods);
fig_save_name = fullfile(output_folder_path, "mse");
saveas(gcf,fig_save_name + ".fig")
saveas(gcf,fig_save_name, 'epsc')

mean_norm_param = sum(parameter(:).^2);
final_plot_struct(snr_values, 'SNR (dB)', selected_methods_mse_data./mean_norm_param, "NMSE", graph_type_mse_vs_num_samples, selected_methods);
fig_save_name = fullfile(output_folder_path, "nmse");
saveas(gcf,fig_save_name + ".fig")
saveas(gcf,fig_save_name, 'epsc')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% MSE versus missmatch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Generate the original graph g
rng(1,"twister");
delta_x_list = 0:0.05:0.3;
num_ters = 10;
sigma = 0.1;
mu = 0.1/sigma^2;
set_size = round(0.7*n);
inv_prior = L;
filter = L;%
[data_vec] = create_non_bayseian_data(filter, parameter, sigma, number_samples);
[V,S]=eig(L);
[V,S] = ensure_incresing_order(V,S);

% Call the appropriate sampling function for each method
mse_data = zeros(length(delta_x_list), length(methods));
for k = 1:num_ters
    mse_data1 = zeros(length(delta_x_list), length(methods));
    for j = 1:length(delta_x_list)
        delta_x = delta_x_list(j);

        % Step 2: Create a perturbation in the graph g_b (perturbed graph)
        % delta_x represents the percentage of edges to perturb
        L_perturbed = perturb_graph(L, delta_x);  % Assume perturb_graph changes delta_x percentage of edges
        [V_perturbed,S_perturbed]=eig(L_perturbed);
        [V_perturbed,S_perturbed] = ensure_incresing_order(V_perturbed,S_perturbed);
        V_perturbed_k=V_perturbed(:,1:lpf_cutoff);
        inv_prior = L_perturbed;
        filter = inv_prior;
        for idx = selected_methods_idx
            disp(['Calculating for method: ' methods(idx).name]);
            [SamplingSet] = greedy_sampling_switch_method(V_perturbed_k, L_perturbed, filter, ...
                inv_prior, methods, idx, mu, sigma, slack_idx, set_size);
            
            % Evaluate MSE for this sampling set and store the result
            switch methods(idx).name
                case 'Random'
                   mse_data1(j, idx) = random_sampling(filter, data_vec, n, inv_prior, mu, ...
                    sigma, slack_idx, parameter, x_0, num_of_iterations_random_sampling, ...
                    set_size);
                otherwise
                mse_data1(j, idx) = evalute_sampling_set(set_size, SamplingSet, filter, data_vec, n, inv_prior, mu, sigma, slack_idx, parameter, x_0);
            end
        end  
    end
    mse_data = mse_data + mse_data1./num_ters;
end
selected_methods = methods(selected_methods_idx);
selected_methods_mse_data = mse_data(:,selected_methods_idx);


output_folder_path = fullfile(init_output_folder_path, "error_vs_delta_L");
save(fullfile(output_folder_path, "workspace.mat"))


% Pass the selected methods to the final_plot function
final_plot_struct(100.*delta_x_list, '$\Delta$ L [\%]', selected_methods_mse_data, "MSE", 1, selected_methods);
fig_save_name = fullfile(output_folder_path, "mse");
saveas(gcf,fig_save_name + ".fig")
saveas(gcf,fig_save_name, 'epsc')

mean_norm_param = sum(parameter(:).^2);
final_plot_struct(100.*delta_x_list, '$\Delta$ L [\%]', selected_methods_mse_data./mean_norm_param, "NMSE", 1, selected_methods);
legend('Location', 'northwest');
ticks = [1, 10^0.5, 10];
yticks(ticks);

% 2. Define the labels using LaTeX syntax
yticklabels({'$10^{0}$', '$10^{0.5}$', '$10^{1}$'});

% 3. Set the interpreter so MATLAB renders the LaTeX
set(gca, 'TickLabelInterpreter', 'latex');

fig_save_name = fullfile(output_folder_path, "nmse");
saveas(gcf,fig_save_name + ".fig")
saveas(gcf,fig_save_name, 'epsc')

