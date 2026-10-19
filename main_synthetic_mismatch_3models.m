% main_synthetic_mismatch_3models.m
%
% PURPOSE:
%   Extend main_synthetic_controlled_mismatch_1.m to THREE different
%   (h_M, h_R+) generative model combinations, using DATA_MODE =
%   'gaussian_low_mu' only.  For each model the experiment mirrors
%   mismatch_1 exactly: data generated from L_true; ALL methods receive
%   the perturbed L_p for BOTH sampling-set selection AND GFR-ML
%   estimation; NMSE vs delta_L is plotted and saved.
%
% THREE MODELS:
%
%   MODEL 1 — f_gmrf_h_L   (replicates mismatch_1, matches power system)
%       h_M(L)  = L   (normalised)
%       h_R+(L) = L   (normalised)
%       x ~ N(0, L_true^† / mu)
%
%   MODEL 2 — f_tikh_h_I   (matches WSN / sensor-reading setup)
%       h_M(L)  = I              (identity: plain sensor readings)
%       h_R+(L) = (I+alpha*L)^2  (normalised Tikhonov smooth prior)
%       x ~ N(0, (I+alpha*L_true)^{-2} / mu)   [smooth signals]
%
%   MODEL 3 — f_diffusion_h_exp  (diffusion measurement + GMRF prior)
%       h_M(L)  = exp(2*tau*L)  (normalised diffusion filter)
%       h_R+(L) = L             (normalised, GMRF prior)
%       x ~ N(0, L_true^† / mu)
%
% Parameters are kept identical to main_synthetic_controlled_mismatch_1.m
% (N=80, sigma=0.1, sampling_ratio=0.7, num_iters=1000) so that MODEL 1
% serves as a direct consistency check.
%
% Output:
%   ./Figures/synthetic_controlled_mismatch_3models/<model_name>/nmse.{fig,eps}
%   ./Figures/synthetic_controlled_mismatch_3models/<model_name>/workspace.mat

clear all; close all; clc;
rng(42, 'twister');
set(groot, 'defaultFigurePosition', [400, 400, 560, 420]);

addpath('./sampling_methods');
addpath('./auxiliary_functions');
addpath('./cost_functions');
addpath('./GreedySampling/');
addpath('./GreedySampling/efficient_update_function/');
addpath('./data_source_files');
addpath('./PGD_Sampling/');

%% =========================================================
%% 1.  Shared parameters  (identical to mismatch_1.m)
%% =========================================================
N               = 80;
sigma           = 0.1;
mu              = 0.1 / sigma^2;   % = 10  (power-system convention)

sampling_ratio  = 0.7;
set_size        = round(sampling_ratio * N);
number_samples  = 3000;
num_iters       = 1000;
num_rand_iters  = 100;
slack_idx       = 1;
lpf_cutoff      = ceil(N / 2);
x_0             = zeros(N, 1);
delta_x_list    = 0 : 0.05 : 0.50;

% Model-specific hyper-parameters
alpha_tikh      = 0.3;   % Tikhonov (model 2): h_R+(L) = (I + alpha*L)^2
tau_diff        = 0.025; % Diffusion (model 3): h_M(L) = exp(2*tau*L)

%% =========================================================
%% 2.  Build a random connected weighted graph
%% =========================================================
connected = false;
attempt   = 0;
while ~connected
    attempt = attempt + 1;
    rng(attempt, 'twister');
    G = erdosRenyi(N, 0.05, 2);
    A = full(G.Adj);
    A = double(A | A');
    A = A - diag(diag(A));
    D_deg = diag(sum(A, 2));
    L_test = D_deg - A;
    connected = (rank(L_test) == N - 1);
end
rng(42, 'twister');
W = A .* (0.5 + rand(N));
W = triu(W, 1);
W = W + W';
L_true = diag(sum(W, 2)) - W;
fprintf('Graph: N=%d, %d edges\n', N, sum(sum(A))/2);

[V_true, S_true] = eig(L_true);
[V_true, ~] = ensure_incresing_order(V_true, S_true);

%% =========================================================
%% 3.  Helper: build (filter, inv_prior, prior_cov) from a Laplacian
%%     for each of the three models.
%%     filter    = h_M(L)   — passed as 'filter_p'   in the main loop
%%     inv_prior = h_R+(L)  — passed as 'prior_p'    in the main loop
%%                             (regularisation term in K(d))
%%     prior_cov             — used to generate data from L_true only
%% =========================================================
function [filt, inv_prior, prior_cov] = build_model_1(L, mu)
    % h_M = L (normalised),  h_R+ = L (normalised),  cov = L^† / mu
    N = size(L, 1);
    filt      = (N / trace(L)) .* L + ((N / trace(L)) .* L)*((N / trace(L)) .* L);
    inv_prior = (N / trace(L)) .* L + ((N / trace(L)) .* L)*((N / trace(L)) .* L);
    prior_cov = pseudo_inverse(inv_prior) / mu;
end

function [filt, inv_prior, prior_cov] = build_model_2(L, mu, alpha)
    N = size(L, 1);
    filt      = (N / trace(L)) .* L + ((N / trace(L)) .* L)*((N / trace(L)) .* L)+ ((N / trace(L)) .* L)*((N / trace(L)) .* L)*((N / trace(L)) .* L);
    inv_prior = ((N / trace(L)) .* L);
    prior_cov = pseudo_inverse(inv_prior) / mu;
end

function [filt, inv_prior, prior_cov] = build_model_3(L, mu, tau)
     N = size(L, 1);
    filt      = ((N / trace(L)) .* L)*((N / trace(L)) .* L)*((N / trace(L)) .* L);
    inv_prior = ((N / trace(L)) .* L);
    prior_cov = pseudo_inverse(inv_prior) / mu;
end

% Tiny helper used inside build_model_3 (ensure_incresing_order may not be
% available as a subfunction, so we duplicate the sort here)
function [Vs, Ss] = sort_eig(V, S)
    [d, ix] = sort(diag(S), 'ascend');
    Vs = V(:, ix);
    Ss = diag(d);
end

%% =========================================================
%% 4.  Loop over the three models
%% =========================================================
model_names  = {'f_gmrf_h_L', 'f_tikh_h_I', 'f_diffusion_h_exp'};
model_labels = {'h_M=L,\;h_R^+=L', ...
                'h_M=I,\;h_R^+=(I+\alpha L)^2', ...
                'h_M=e^{2\tau L},\;h_R^+=L'};

methods              = getPlottingConstants();
selected_methods_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
selected_methods     = methods(selected_methods_idx);

for model_idx = 1 : 3

    fprintf('\n===== MODEL %d: %s =====\n', model_idx, model_names{model_idx});

    %% -------------------------------------------------------
    %% 4a.  Build TRUE-graph filters and generate Gaussian data
    %% -------------------------------------------------------
    switch model_idx
        case 1
            [filt_true, inv_prior_true, prior_cov_true] = ...
                build_model_1(L_true, mu);
        case 2
            [filt_true, inv_prior_true, prior_cov_true] = ...
                build_model_2(L_true, mu, alpha_tikh);
        case 3
            [filt_true, inv_prior_true, prior_cov_true] = ...
                build_model_3(L_true, mu, tau_diff);
    end

    % Data: x ~ N(0, prior_cov_true),  y = filt_true * x + noise
    [data_vec, parameter] = create_bayseian_data( ...
        filt_true, prior_cov_true, sigma, number_samples, x_0, slack_idx);
    fprintf('Data generated.  mu=%.2f, sigma=%.2f\n', mu, sigma);

    norm_x = sum(parameter(:).^2) / size(parameter, 2);

    %% -------------------------------------------------------
    %% 4b.  Main loop: NMSE vs topology perturbation delta_L
    %% -------------------------------------------------------
    mse_data = zeros(length(delta_x_list), length(methods));

    for k = 1 : num_iters
        if mod(k, 100) == 0
            fprintf('  Model %d — iter %d / %d\n', model_idx, k, num_iters);
        end
        mse_iter = zeros(length(delta_x_list), length(methods));

        for j = 1 : length(delta_x_list)
            delta_x = delta_x_list(j);

            % Perturbed Laplacian — given to ALL methods
            L_p = perturb_graph(L_true, delta_x);
            [V_p, S_p] = eig(L_p);
            [V_p, ~]   = ensure_incresing_order(V_p, S_p);
            V_p_k      = V_p(:, 1:lpf_cutoff);

            % Build perturbed (filter, inv_prior) for THIS model
            switch model_idx
                case 1
                    [filter_p, inv_prior_p, ~] = build_model_1(L_p, mu);
                case 2
                    [filter_p, inv_prior_p, ~] = build_model_2(L_p, mu, alpha_tikh);
                case 3
                    [filter_p, inv_prior_p, ~] = build_model_3(L_p, mu, tau_diff);
            end

            for idx = selected_methods_idx
                [SamplingSet] = greedy_sampling_switch_method( ...
                    V_p_k, L_p, filter_p, inv_prior_p, ...
                    methods, idx, mu, sigma, slack_idx, set_size);

                switch methods(idx).name
                    case 'Random'
                        mse_iter(j, idx) = random_sampling( ...
                            filter_p, data_vec, N, inv_prior_p, ...
                            mu, sigma, slack_idx, parameter, x_0, ...
                            num_rand_iters, set_size);
                    otherwise
                        mse_iter(j, idx) = evalute_sampling_set( ...
                            set_size, SamplingSet, filter_p, data_vec, N, ...
                            inv_prior_p, mu, sigma, slack_idx, parameter, x_0);
                end
            end
        end
        mse_data = mse_data + mse_iter / num_iters;
    end

    %% -------------------------------------------------------
    %% 4c.  Save and plot
    %% -------------------------------------------------------
    output_folder = fullfile('./Figures/synthetic_controlled_mismatch_3models', ...
                             model_names{model_idx});
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    save(fullfile(output_folder, 'workspace.mat'));

    mse_sel = mse_data(:, selected_methods_idx);

    % NMSE vs delta_L
    final_plot_struct(100 * delta_x_list, '$\Delta L$ [\%]', ...
        mse_sel ./ norm_x, 'NMSE', 1, selected_methods);
    legend('Location', 'northwest');
    xlabel('$\Delta L$ [\%]', 'Interpreter', 'latex');
    ylabel('NMSE', 'Interpreter', 'latex');
    title(model_labels{model_idx}, 'Interpreter', 'latex');

    saveas(gcf, fullfile(output_folder, 'nmse.fig'));
    saveas(gcf, fullfile(output_folder, 'nmse'), 'epsc');
    fprintf('Done model %d. Saved to %s\n', model_idx, output_folder);
end

fprintf('\nAll three models complete.\n');
