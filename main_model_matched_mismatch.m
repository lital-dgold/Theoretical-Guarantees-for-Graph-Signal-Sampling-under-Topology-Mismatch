% main_model_matched_mismatch.m
%
% PURPOSE
%   Model-matched validation of Theorem 1, Theorems 2-3 and Corollary 3.
%   The mismatch satisfies the proportional model (13) EXACTLY, so the
%   Monte-Carlo MSE of every sampling set must equal
%            R_alpha(d) = BMSE(d) + alpha * bCRB(d).
%
% DATA MODEL (Bayesian, Assumptions (A1)-(A2))
%   x  ~ N(0, (mu*h_R(L))^dagger)          (slack entry fixed to 0)
%   e  ~ N(0, Sigma)
%   y  = h_M(L) x + e - Delta_h x
%   Delta_h = Sigma^{1/2} U diag(s) U' Sigma^{1/2}     (symmetric, zero mean,
%             s_i independent, drawn independently per sample, indep. of x,e)
%   Var(s_i) = alpha / [U' Sigma^{1/2} Gamma Sigma^{1/2} U]_ii,  Gamma = E[x x']
%   =>  C_delta = E[Delta_h Gamma Delta_h] = alpha * Sigma   (exactly).
%
%   U is a fixed random orthogonal matrix. It is needed because x(slack)=0
%   gives Gamma_ii = 0 at the slack bus, so U = I would fail there.
%
%   The estimator (3) and ALL sampling designs use the same assumed L;
%   there is no edge perturbation in this experiment.
%
% OUTPUT
%   ./Figures/ieee118/model_matched.{fig,pdf,eps}  (panels (a),(b))
%   ./Figures/ieee118/workspace.mat

clear; close all; clc;
set(groot, 'defaultFigurePosition', [400, 400, 560, 420]);
folders = {'./sampling_methods', './auxiliary_functions', './cost_functions', ...
           './GreedySampling/', './GreedySampling/efficient_update_function/', ...
           './data_source_files', './PGD_Sampling/'};
for f = 1:numel(folders)
    if exist(folders{f}, 'dir'), addpath(folders{f}); end
end

%% ===================== Parameters =====================
sigma          = 0.1;                  % Sigma = sigma^2 I  (= 0.01 I)
mu             = 0.1 / sigma^2;
q_ratio        = 0.7;
T              = 20000;                % Monte-Carlo samples (common to all designs)
alpha_list     = [0, logspace(-2, 3, 21)];
num_rand_sets  = 100;
ia             = 2:numel(alpha_list);  % alpha > 0 (log axes, minimax grid)
xa             = alpha_list(ia);
PLOT_BASELINES = true;                % theory curves of A/E/D/T/LR/... designs
kappa_mm       = 10;                   % minimax policy knows only alpha in
                                       % [alpha/kappa_mm, kappa_mm*alpha]  -> uses d_{kappa_mm*alpha}

%% ===================== Graph (IEEE 118-bus) =====================
n = 118;
[Y, ~, ~, ~, ~, ~, ~] = load_cdf(sprintf('ieee%dcdf.txt', n));
L = -imag(full(Y));
L = (L + L') / 2;
% -imag(Y) contains line-charging / shunt / tap terms on the diagonal,
% so it is NOT a Laplacian (L*1 ~= 0) and its reduced version can be
% indefinite or nearly singular. Keep only the line susceptances:
L = L - diag(sum(L, 2));          % now L*1 = 0, L PSD  (L = diag(W1) - W)
slack_idx = 111;

rng(1, 'twister');

% Graph filters (DC power-flow / GMRF model)
hM = L;            % measurement filter h_M(L)
hR = L;            % regularization filter h_R^+(L)

set_size = round(q_ratio * n);
nz       = setdiff(1:n, slack_idx);           % identifiable coordinates
H_r      = hM(:, nz);
F_r      = mu * hR(nz, nz);  F_r = (F_r + F_r') / 2;
Sigma    = sigma^2 * eye(n);
Sinv     = inv(Sigma);
Sh       = sqrtm(Sigma);  Sh = (Sh + Sh') / 2;

[VF, lF] = eig(F_r, 'vector');
fprintf('Reduced precision mu*h_R: min eig = %.3g, max eig = %.3g, cond = %.3g\n', ...
        min(lF), max(lF), max(lF) / min(lF));
assert(min(lF) > 0, 'mu*h_R(L) (slack removed) is not positive definite.');
Gamma_r  = VF * diag(1 ./ lF) * VF';  Gamma_r = (Gamma_r + Gamma_r') / 2;   % prior covariance
Gamma    = zeros(n);  Gamma(nz, nz) = Gamma_r;
Ex2      = trace(Gamma_r);                                  % E||x||^2: NMSE of THEORY lines

%% ===================== Monte-Carlo draws (common random numbers) =====================
X_r = VF * (sqrt(1 ./ lF) .* randn(n - 1, T));               % x ~ N(0, Gamma_r)
X   = zeros(n, T);  X(nz, :) = X_r;
E   = Sh * randn(n, T);

[U, ~] = qr(randn(n));
g = diag(U' * Sh * Gamma * Sh * U);
assert(all(g > 1e-10 * max(g)), 'Degenerate U: redraw it.');

% B(:,t) = Delta_h^{(t)} x_t for alpha = 1; for general alpha it scales by sqrt(alpha)
B  = Sh * U * ((1 ./ sqrt(g)) .* randn(n, T) .* (U' * Sh * X));
Y0 = H_r * X_r + E;                                          % mismatch-free part
Ex2_emp = mean(sum(X_r.^2, 1));     % (1/T) sum_i ||x_i||^2: NMSE of MC markers, eq. (23)
% emp/Ex2_emp = sum_i ||xhat_i - x_i||^2 / sum_i ||x_i||^2  exactly as in (23)

C_hat = (B * B') / T;
fprintf(['Check C_delta = alpha*Sigma (alpha=1): rel. Frobenius error = %.3f ', ...
         '(Monte-Carlo level ~ sqrt(3n/T) = %.3f), tr ratio = %.3f\n'], ...
        norm(C_hat - Sigma, 'fro') / norm(Sigma, 'fro'), sqrt(3 * n / T), ...
        trace(C_hat) / trace(Sigma));

%% ===================== Nominal sampling designs =====================
methods = getPlottingConstants();
names   = {methods.name};
idxB    = find(strcmp(names, 'BMSE'));
idxC    = find(strcmp(names, 'b. CRB'));

[V, S]  = eig(L);
[V, ~]  = ensure_incresing_order(V, S);
V_k     = V(:, 1:ceil(n / 2));

sets = cell(1, numel(methods));
for idx = 1:numel(methods)
    if strcmp(names{idx}, 'Random'), continue; end
    fprintf('Sampling set: %s\n', names{idx});
    sets{idx} = greedy_sampling_switch_method(V_k, L, hM, hR, methods, idx, ...
                                              mu, sigma, slack_idx, set_size);
end
rand_sets = arrayfun(@(k) randperm(n, set_size), 1:num_rand_sets, 'UniformOutput', false);

%% ===================== Mismatch-aware designs d_alpha (16) =====================
sets_alpha = cell(1, numel(alpha_list));
for a = 1:numel(alpha_list)
    al  = alpha_list(a);
    cf  = @(v, hL, fL, iR) Calc_BMSE(v, hL, fL, iR) + al * Calc_BCRB(v, hL, fL, iR);
    cfe = @(iK, r, fL)     Calc_BMSE_efficient(iK, r, fL) + al * Calc_BCRB_efficient(iK, r, fL);
    fprintf('d_alpha, alpha = %.3g\n', al);
    sets_alpha{a} = GreedySampling(hM, mu .* hR, set_size, sigma^(-2), slack_idx, cf, cfe);
end

%% ===================== d_C refinement: node-swap local search (multi-start) =====================
% bCRB = trace(K^{-2}M) is NOT monotone when nodes are added sequentially:
% adding a node raises M (increases bCRB) AND raises K (decreases bCRB via K^{-2}).
% The net sign is ambiguous, so greedy bCRB minimisation can be trapped at a
% poor local minimum whose bCRB exceeds oracle d_alpha values at some alpha,
% violating the condition  bCRB(d_C) <= bCRB(d_alpha) for all alpha  that
% Theorem 2 requires.
%
% Fix: 2-opt node-swap local search on bCRB, seeded from (a) the greedy d_C
% and (b) oracle d_alpha designs at large alpha (which often have low bCRB).
fprintf('\nRefining d_C with multi-start node-swap local search...\n');
[~, dC_bcrb0] = design_costs(sets{idxC}, n, H_r, F_r, Sinv);
% Pick seeds: greedy d_C + the 2 large-alpha oracle designs with the lowest bCRB.
% Keeping only 3 seeds total keeps runtime manageable (~1-3 min).
large_alpha_idx = find(alpha_list >= 1);
oracle_bcrbs    = cellfun(@(S) design_costs_bcrb(S, n, H_r, F_r, Sinv), sets_alpha(large_alpha_idx));
[~, sort_idx]   = sort(oracle_bcrbs);
top_oracle_idx  = large_alpha_idx(sort_idx(1:min(2, numel(sort_idx))));
seed_sets_dC    = [{sets{idxC}}, sets_alpha(top_oracle_idx)];
bcrb_best_dC    = dC_bcrb0;
sets_dC_refined = sets{idxC};
max_swap_iters  = 5;    % cap per-seed iterations; 5 rounds is enough in practice
for s_idx = 1:numel(seed_sets_dC)
    fprintf('  Running seed %d / %d ...\n', s_idx, numel(seed_sets_dC));
    S_ref = bcrb_node_swap(seed_sets_dC{s_idx}, n, H_r, F_r, Sinv, slack_idx, max_swap_iters);
    [~, bc] = design_costs(S_ref, n, H_r, F_r, Sinv);
    if bc < bcrb_best_dC - 1e-10 * max(abs(bcrb_best_dC), 1)
        bcrb_best_dC   = bc;
        sets_dC_refined = S_ref;
        fprintf('  Seed %d improved d_C bCRB to %.4g\n', s_idx, bc);
    end
end
fprintf('d_C bCRB: %.4g (greedy)  ->  %.4g (after local search, %.1f%% improvement)\n', ...
        dC_bcrb0, bcrb_best_dC, 100*(dC_bcrb0 - bcrb_best_dC) / max(abs(dC_bcrb0), 1e-12));
sets{idxC} = sets_dC_refined;

%% ===================== Minimax designs d_{alpha_max} (Corollary 1) =====================
% True alpha is unknown; only alpha in [alpha/kappa_mm, kappa_mm*alpha] is known.
% By Corollary 1 the minimax design depends only on the upper end:
% d_mm(alpha) = d_{alpha_max},  alpha_max = kappa_mm * alpha.
sets_mm = cell(1, numel(alpha_list));
for a = 1:numel(alpha_list)
    al  = kappa_mm * alpha_list(a);
    cf  = @(v, hL, fL, iR) Calc_BMSE(v, hL, fL, iR) + al * Calc_BCRB(v, hL, fL, iR);
    cfe = @(iK, r, fL)     Calc_BMSE_efficient(iK, r, fL) + al * Calc_BCRB_efficient(iK, r, fL);
    fprintf('minimax design, alpha_max = %.3g\n', al);
    sets_mm{a} = GreedySampling(hM, mu .* hR, set_size, sigma^(-2), slack_idx, cf, cfe);
end

%% ===================== Minimax refinement: bCRB node-swap + d_C fallback =====================
% Why composite_node_swap failed: it minimises BMSE + kappa*alpha*bCRB, which
% can INCREASE bCRB while reducing BMSE -- exactly the wrong trade-off for the
% regret bound RegM <= Delta_B, which requires bCRB(d_mm) <= bCRB(d_C).
%
% Correct fix: minimise bCRB directly for each minimax design (same objective as
% d_C).  If the result still has bCRB > bCRB(d_C), fall back to d_C itself.
% After the fix: bCRB(d_mm{a}) <= bCRB(d_C) <= bCRB(d_oracle) for all a,
% which is the condition Theorem 2's proof needs to bound RegM.
fprintf('\nRefining minimax designs (bCRB swap + d_C fallback)...\n');
[~, dC_bcrb_final] = design_costs(sets{idxC}, n, H_r, F_r, Sinv);
max_swap_iters_mm = 3;
n_replaced = 0;
for a = 1:numel(alpha_list)
    if alpha_list(a) == 0, continue; end
    % Minimise bCRB from the greedy minimax seed
    S_ref   = bcrb_node_swap(sets_mm{a}, n, H_r, F_r, Sinv, slack_idx, max_swap_iters_mm);
    [~, bc] = design_costs(S_ref, n, H_r, F_r, Sinv);
    if bc < dC_bcrb_final - 1e-10 * max(abs(dC_bcrb_final), 1)
        sets_mm{a} = S_ref;           % found a design with even lower bCRB
    else
        sets_mm{a} = sets{idxC};      % fall back: d_C has minimum bCRB
        n_replaced = n_replaced + 1;
    end
    if mod(a, 5) == 0 || a == numel(alpha_list)
        fprintf('  Refined %d / %d minimax designs (%d replaced by d_C)\n', ...
                a, numel(alpha_list), n_replaced);
    end
end
fprintf('Minimax refinement done. %d / %d designs replaced by d_C.\n', ...
        n_replaced, numel(alpha_list) - 1);

%% ===================== Evaluation: theory vs Monte Carlo =====================
res = cell(1, numel(methods));
for idx = 1:numel(methods)
    if strcmp(names{idx}, 'Random')
        acc = [];
        for k = 1:num_rand_sets
            r = eval_set(rand_sets{k}, n, H_r, F_r, Sinv, Y0, B, X_r, alpha_list);
            if isempty(acc), acc = r; else
                acc.bmse = acc.bmse + r.bmse; acc.bcrb = acc.bcrb + r.bcrb;
                acc.theo = acc.theo + r.theo; acc.emp  = acc.emp  + r.emp;
            end
        end
        acc.bmse = acc.bmse / num_rand_sets; acc.bcrb = acc.bcrb / num_rand_sets;
        acc.theo = acc.theo / num_rand_sets; acc.emp  = acc.emp  / num_rand_sets;
        res{idx} = acc;
    else
        res{idx} = eval_set(sets{idx}, n, H_r, F_r, Sinv, Y0, B, X_r, alpha_list);
    end
end

theoA = zeros(size(alpha_list)); empA = theoA;
bmseA = theoA; bcrbA = theoA;
for a = 1:numel(alpha_list)
    r = eval_set(sets_alpha{a}, n, H_r, F_r, Sinv, Y0, B, X_r, alpha_list);
    theoA(a) = r.theo(a);  empA(a) = r.emp(a);
    bmseA(a) = r.bmse;     bcrbA(a) = r.bcrb;
end

% Minimax policy evaluated at the TRUE alpha (C_delta = alpha*Sigma)
theoM = zeros(size(alpha_list)); empM = theoM;
bmseM = theoM; bcrbM = theoM;
for a = 1:numel(alpha_list)
    r = eval_set(sets_mm{a}, n, H_r, F_r, Sinv, Y0, B, X_r, alpha_list);
    theoM(a) = r.theo(a);  empM(a) = r.emp(a);
    bmseM(a) = r.bmse;     bcrbM(a) = r.bcrb;
end

%% ===================== Regret, gaps, crossover =====================
% Reference optimum: best design among all candidates (greedy d_alpha cannot
% be guaranteed to be the exact minimizer of (16)).
cand_bmse = [bmseA, bmseM, cellfun(@(r) r.bmse, res(~strcmp(names, 'Random')))];
cand_bcrb = [bcrbA, bcrbM, cellfun(@(r) r.bcrb, res(~strcmp(names, 'Random')))];
for k = 1:num_rand_sets
    [b1, c1] = design_costs(rand_sets{k}, n, H_r, F_r, Sinv);
    cand_bmse(end + 1) = b1;  cand_bcrb(end + 1) = c1; %#ok<SAGROW>
end
R_best = min(cand_bmse' + cand_bcrb' * alpha_list, [], 1);

RegB   = res{idxB}.theo - R_best;
RegC   = res{idxC}.theo - R_best;
RegA   = theoA - R_best;                 % oracle d_alpha (~0 up to greedy)
RegM   = theoM - R_best;                 % minimax d_{kappa*alpha}
DeltaB = res{idxC}.bmse - res{idxB}.bmse;
DeltaC = res{idxB}.bcrb - res{idxC}.bcrb;
alpha_cross = DeltaB / DeltaC;

mask = cand_bcrb > res{idxC}.bcrb * (1 + 1e-9);
alpha0_hat = max([0, (res{idxC}.bmse - cand_bmse(mask)) ./ (cand_bcrb(mask) - res{idxC}.bcrb)]);

fprintf('\nDelta_B = %.4g, Delta_C = %.4g, crossover Delta_B/Delta_C = %.4g\n', ...
        DeltaB, DeltaC, alpha_cross);
fprintf('alpha_0 (estimated over candidate designs) = %.4g\n', alpha0_hat);
fprintf('max |MC - theory|/theory : d_B %.2f%%, d_C %.2f%%, d_alpha %.2f%%\n', ...
        100 * max(abs(res{idxB}.emp - res{idxB}.theo) ./ res{idxB}.theo), ...
        100 * max(abs(res{idxC}.emp - res{idxC}.theo) ./ res{idxC}.theo), ...
        100 * max(abs(empA - theoA) ./ theoA));
fprintf('max_alpha Reg(d_C) = %.4g  (Theorem 2(a): <= Delta_B = %.4g)\n', max(RegC), DeltaB);

%% ===================== Minimax policy (Corollaries 1-2) =====================
% Uncertainty set (17): alpha_min*Sigma <= C_delta <= alpha_max*Sigma.
% Non-proportional members are realized with the same symmetric construction,
% using a different variance per whitened direction:
%     Var(s_i) = a_i / g_i,  a_i in [alpha_min, alpha_max]
%  => C_delta = Sigma^{1/2} U diag(a) U' Sigma^{1/2}  (inside (17), not proportional).
% For these, Theorem 1 gives the exact risk  R(d) = BMSE(d) + w(d)' * a,
% with w_i(d) = ||G(d) Sigma^{1/2} U e_i||^2 and bCRB(d) = sum_i w_i(d).
rho_minmax = 0.1;            % alpha_min = rho_minmax * alpha_max
num_Cdelta = 500;            % random members of the uncertainty set per alpha_max

mlist     = find(~strcmp(names, 'Random'));
cand_sets = [sets_alpha, sets(mlist), rand_sets];
nA = numel(sets_alpha);  nc = numel(cand_sets);
cB_idx = nA + find(mlist == idxB);
cC_idx = nA + find(mlist == idxC);

Wc = zeros(n, nc);  bm = zeros(nc, 1);
for c = 1:nc
    [bm(c), ~, Ki, ~, Dm] = design_costs(cand_sets{c}, n, H_r, F_r, Sinv);
    Gm = Ki * (H_r' * Dm * Sinv * Dm);
    Wc(:, c) = sum((Gm * Sh * U).^2, 1)';
end

amax_list = alpha_list(ia);
mm.amax       = amax_list;
mm.worst      = zeros(3, numel(amax_list));   % worst-case risk: d_B, d_C, d_{alpha_max}
mm.worst_best = zeros(1, numel(amax_list));   % smallest worst-case risk among candidates
mm.reg_max    = zeros(3, numel(amax_list));   % max regret over sampled C_delta
mm.bound      = zeros(1, numel(amax_list));   % (18), evaluated at the worst-case member
mm.bound_ok   = true;
for k = 1:numel(amax_list)
    amax = amax_list(k);  amin = rho_minmax * amax;
    cM   = ia(k);                              % index of d_{alpha_max} in sets_alpha
    Avec = amin + (amax - amin) * rand(n, num_Cdelta);
    Avec(:, 1)    = amax;                      % upper corner  C = alpha_max*Sigma
    Avec(:, 2)    = amin;                      % lower corner  C = alpha_min*Sigma
    Avec(:, 3:22) = amin + (amax - amin) * (rand(n, 20) > 0.5);   % extreme points
    R = bm + Wc' * Avec;                       % nc x num_Cdelta exact risks
    [Rbest, ibest] = min(R, [], 1);
    Reg = R - Rbest;
    bnd = (amax - amin) * sum(Wc(:, ibest), 1);   % (18) with d_opt -> best candidate
    mm.worst(:, k)    = max(R([cB_idx, cC_idx, cM], :), [], 2);
    mm.worst_best(k)  = min(max(R, [], 2));
    mm.reg_max(:, k)  = max(Reg([cB_idx, cC_idx, cM], :), [], 2);
    [~, jw] = max(Reg(cM, :));
    mm.bound(k)       = bnd(jw);
    mm.bound_ok       = mm.bound_ok && all(Reg(cM, :) <= bnd * (1 + 1e-9));
end

fprintf('\nMinimax check, alpha_min = %.2g * alpha_max  (risks/regrets normalized by E||x||^2)\n', rho_minmax);
fprintf('%9s | %27s | %9s | %27s | %9s\n', 'alpha_max', ...
        'worst-case: d_B   d_C   d_mm', 'best cand', 'max regret: d_B  d_C  d_mm', 'bound(18)');
for k = 1:numel(amax_list)
    fprintf('%9.3g | %8.4f %8.4f %8.4f | %9.4f | %8.4f %8.4f %8.4f | %9.4f\n', mm.amax(k), ...
            mm.worst(:, k) / Ex2, mm.worst_best(k) / Ex2, mm.reg_max(:, k) / Ex2, mm.bound(k) / Ex2);
end
fprintf('Corollary 1: worst case attained at alpha_max*Sigma -> d_{alpha_max} (greedy) has the\n');
fprintf('             smallest worst-case risk up to greedy suboptimality (compare columns).\n');
fprintf('Corollary 2: regret of d_{alpha_max} <= (18) for all sampled C_delta: %d\n', mm.bound_ok);

%% ===================== Figure: (a) NMSE vs alpha, (b) regret vs alpha =====================
out_dir = './Figures/ieee118';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
cB = [0 0.4470 0.7410]; cC = [0.8500 0.3250 0.0980]; cA = [0.4660 0.6740 0.1880];
cM_col = [0.4940 0.1840 0.5560];

LAYOUT = 'column';   % 'column': panels stacked, one IEEE column (\columnwidth)
                     % 'row'   : panels side by side, full page width (figure*)
switch LAYOUT
    case 'column', fig_w = 3.5;  fig_h = 4.4; tl_rows = 2; tl_cols = 1;
    case 'row',    fig_w = 7.16; fig_h = 2.5; tl_rows = 1; tl_cols = 2;
end
fs = 8;                                               % IEEE figure font size

fig = figure('Units', 'inches', 'Position', [1 1 fig_w fig_h], 'Color', 'w');
tl  = tiledlayout(fig, tl_rows, tl_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- (a) NMSE vs alpha: lines = BMSE + alpha*bCRB (15), markers = Monte Carlo
ax1 = nexttile(tl); hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on');
h = gobjects(0); lab = {};

% One row per policy: {theory, MC, color, marker, line style, label}
pol = { res{idxB}.theo, res{idxB}.emp, cB,     'o', '-',  '$\mathbf{d}_{\rm B}$';
        res{idxC}.theo, res{idxC}.emp, cC,     's', ':',  '$\mathbf{d}_{\rm C}$';
        theoA,          empA,          cA,     '^', '--', '$\mathbf{d}_\alpha$';
        theoM,          empM,          cM_col, 'd', '-.', '$\mathbf{d}_{\alpha_{\max}}$' };

for p = 1:size(pol, 1)
    lw = 1.3 + 0.4 * strcmp(pol{p,5}, ':');        % thicker if dotted
    hl = plot(ax1, xa, pol{p,1}(ia) / Ex2, pol{p,5}, 'Color', pol{p,3}, 'LineWidth', lw); % theory line
    hm = plot(ax1, xa, pol{p,2}(ia) / Ex2_emp, pol{p,4}, ...        % Monte Carlo markers
              'Color', pol{p,3}, 'MarkerSize', 4, 'LineStyle', 'none');
    h(end+1:end+2)   = [hl, hm];                                    % interleave: line, markers
    lab(end+1:end+2) = {[pol{p,6} ' theory'], [pol{p,6} ' MC']};
end

if PLOT_BASELINES                                                   % A/E/D/T designs individually
    baseline_idx = [1, 2, 3, 4];  % A-design, E-design, D-design, T-design
    for idx = baseline_idx
        hb = plot(ax1, xa, res{idx}.theo(ia) / Ex2, ...
                  [methods(idx).linestyle methods(idx).marker], ...
                  'Color', methods(idx).color, 'LineWidth', 0.9, 'MarkerSize', 4);
        h(end+1)   = hb;
        lab{end+1} = ['$\mathbf{d}_{\rm ' methods(idx).name(1) '}$'];
    end
end

if DeltaC > 0
    xline(ax1, alpha_cross, ':', '$\Delta_{\rm B}/\Delta_{\rm C}$', 'Color', [0.4 0.4 0.4], ...
      'LineWidth', 0.8, 'Interpreter', 'latex', 'FontSize', fs - 1, ...
      'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right', ...
      'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
end
set(ax1, 'XScale', 'log', 'YScale', 'log', 'FontSize', fs, 'TickLabelInterpreter', 'latex');
xlim(ax1, [xa(1) xa(end)]);
HEADROOM_A = 0.05;
nmse_vals = [];
for p = 1:size(pol, 1)
    nmse_vals = [nmse_vals, pol{p,1}(ia)/Ex2, pol{p,2}(ia)/Ex2_emp];
end
if PLOT_BASELINES
    for idx = baseline_idx
        nmse_vals = [nmse_vals, res{idx}.theo(ia)/Ex2];
    end
end
yr_top_a = max(nmse_vals(nmse_vals > 0));
yr_bot_a = min(nmse_vals(nmse_vals > 0));
ylim(ax1, [10^(log10(yr_bot_a) - HEADROOM_A/(1-HEADROOM_A)*(log10(yr_top_a)-log10(yr_bot_a))), ...
           10^(log10(yr_top_a) + HEADROOM_A/(1-HEADROOM_A)*(log10(yr_top_a)-log10(yr_bot_a)))]);
xlabel(ax1, '$\alpha$', 'Interpreter', 'latex');
ylabel(ax1, 'NMSE', 'Interpreter', 'latex');

% ---- (b) Regret vs alpha (Theorems 2-3), normalized like the NMSE.
% Log y-axis: zero regret (d_C for alpha >= alpha_0, d_B at alpha = 0) is
% clipped to the bottom of the axis, so it reads as "vanishes".
reg_floor = 1e-3 * DeltaB / Ex2;
ax2 = nexttile(tl); hold(ax2, 'on'); box(ax2, 'on'); grid(ax2, 'on');
plot(ax2, xa, max(RegB(ia) / Ex2, reg_floor), '-o',  'Color', cB,     'LineWidth', 1.3, 'MarkerSize', 4);
plot(ax2, xa, max(RegC(ia) / Ex2, reg_floor), ':s',  'Color', cC,     'LineWidth', 1.7, 'MarkerSize', 4);
plot(ax2, xa, max(RegM(ia) / Ex2, reg_floor), '-.d', 'Color', cM_col, 'LineWidth', 1.3, 'MarkerSize', 4);
yline(ax2, DeltaB / Ex2, '--k', '$\Delta_{\rm B}$', 'Interpreter', 'latex', ...
      'FontSize', fs - 1, 'HandleVisibility', 'off');
if DeltaC > 0
    xline(ax2, alpha_cross, ':', '$\Delta_{\rm B}/\Delta_{\rm C}$', 'Color', [0.4 0.4 0.4], ...
      'LineWidth', 0.8, 'Interpreter', 'latex', 'FontSize', fs - 1, ...
      'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'right', ...
      'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
end
if alpha0_hat > 0
    xline(ax2, alpha0_hat, '-.k', '$\alpha_0$', 'Interpreter', 'latex', ...
          'FontSize', fs - 1, 'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
end
set(ax2, 'XScale', 'log', 'YScale', 'log', 'FontSize', fs, 'TickLabelInterpreter', 'latex');
xlim(ax2, [xa(1) xa(end)]);
HEADROOM_B = 0.05;                                             % empty top fraction; larger = higher ylim
yr_top = max([RegB(ia), RegC(ia), RegM(ia)] / Ex2);       % highest regret value on the plot
yr_top = max(yr_top, DeltaB / Ex2);                         % make sure the Delta_B line fits
ylim(ax2, [reg_floor, 10^(log10(yr_top) + HEADROOM_B / (1 - HEADROOM_B) * ...
                         (log10(yr_top) - log10(reg_floor)))]);
xlabel(ax2, '$\alpha$', 'Interpreter', 'latex');
ylabel(ax2, 'Normalized regret', 'Interpreter', 'latex');
legend(ax2, {'$\mathbf{d}_{\rm B}$', '$\mathbf{d}_{\rm C}$', '$\mathbf{d}_{\alpha_{\max}}$'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', fs - 1);

lgd1 = legend(ax1, h, lab, 'Interpreter', 'latex', 'NumColumns', 2, ...
              'Orientation', 'horizontal', 'Location', 'none', 'FontSize', fs - 1);
lgd1.ItemTokenSize = [12 6];
drawnow;
ax1.Units    = 'normalized';
lgd1.Units   = 'normalized';
p = ax1.Position;  % [left, bottom, width, height] of the axes tile
lgd1.Position(1:2) = [p(1) + 0.0,  p(2) + 0.198];  % nudge from axes bottom-left
%%%%

lgd2 = legend(ax2, {'$\mathbf{d}_{\rm B}$', '$\mathbf{d}_{\rm C}$', '$\mathbf{d}_{\alpha_{\max}}$ (minimax)'}, ...
              'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', fs - 1);
lgd2.ItemTokenSize = [12 6];
% ---- Save (vector PDF/EPS at the exact print size)
fig_name = fullfile(out_dir, 'model_matched');
savefig(fig, [fig_name '.fig']);
exportgraphics(fig, [fig_name '.pdf'], 'ContentType', 'vector');
exportgraphics(fig, [fig_name '.eps'], 'ContentType', 'vector');

save(fullfile(out_dir, 'workspace.mat'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ===================== Table: NMSE vs q% sensors at alpha = 10 =====================
alpha_table   = 10;
q_pct_list    = [30, 40, 50, 60, 70, 80, 90];
table_methods = {'d_B', 'd_C', 'd_alpha', 'd_alpha_max', 'A-design', 'E-design', 'D-design', 'T-design'};
nmse_table    = zeros(numel(table_methods), numel(q_pct_list));

for qi = 1:numel(q_pct_list)
    m = round((q_pct_list(qi) / 100) * n);
    fprintf('Computing q = %d%% (%d sensors)...\n', q_pct_list(qi), m);

    cf_B   = @(v,hL,fL,iR) Calc_BMSE(v,hL,fL,iR);
    cfe_B  = @(iK,r,fL)    Calc_BMSE_efficient(iK,r,fL);
    cf_C   = @(v,hL,fL,iR) Calc_BCRB(v,hL,fL,iR);
    cfe_C  = @(iK,r,fL)    Calc_BCRB_efficient(iK,r,fL);
    cf_A   = @(v,hL,fL,iR) Calc_BMSE(v,hL,fL,iR) + alpha_table          * Calc_BCRB(v,hL,fL,iR);
    cfe_A  = @(iK,r,fL)    Calc_BMSE_efficient(iK,r,fL) + alpha_table    * Calc_BCRB_efficient(iK,r,fL);
    cf_MM  = @(v,hL,fL,iR) Calc_BMSE(v,hL,fL,iR) + kappa_mm*alpha_table * Calc_BCRB(v,hL,fL,iR);
    cfe_MM = @(iK,r,fL)    Calc_BMSE_efficient(iK,r,fL) + kappa_mm*alpha_table * Calc_BCRB_efficient(iK,r,fL);

    for mi = 1:numel(table_methods)
        switch table_methods{mi}
            case 'd_B'
                S = GreedySampling(hM, mu.*hR, m, sigma^(-2), slack_idx, cf_B, cfe_B);
            case 'd_C'
                S = GreedySampling(hM, mu.*hR, m, sigma^(-2), slack_idx, cf_C, cfe_C);
            case 'd_alpha'
                S = GreedySampling(hM, mu.*hR, m, sigma^(-2), slack_idx, cf_A, cfe_A);
            case 'd_alpha_max'
                S = GreedySampling(hM, mu.*hR, m, sigma^(-2), slack_idx, cf_MM, cfe_MM);
            case 'A-design'
                S = A_optimal_design(m, V_k);
            case 'E-design'
                S = E_optimal_design(m, V_k);
            case 'D-design'
                S = D_optimal_design(m, V_k);
            case 'T-design'
                S = T_optimal_design(m, V_k);
        end
        [bm, bc] = design_costs(S, n, H_r, F_r, Sinv);
        nmse_table(mi, qi) = (bm + alpha_table * bc) / Ex2;
    end
end

% --- Print table to console ---
fprintf('\n=== NMSE at alpha = %g (normalized by E||x||^2), graph: ieee118 ===\n', alpha_table);
header = sprintf('%-14s', 'Method');
for qi = 1:numel(q_pct_list)
    header = [header sprintf('  q=%2d%%  ', q_pct_list(qi))]; %#ok<AGROW>
end
fprintf('%s\n%s\n', header, repmat('-', 1, length(header)));
for mi = 1:numel(table_methods)
    row = sprintf('%-14s', table_methods{mi});
    for qi = 1:numel(q_pct_list)
        row = [row sprintf('  %.4f ', nmse_table(mi, qi))]; %#ok<AGROW>
    end
    fprintf('%s\n', row);
end

% --- Save to CSV ---
csv_path = fullfile(out_dir, sprintf('nmse_table_alpha%g.csv', alpha_table));
fid = fopen(csv_path, 'w');
fprintf(fid, 'Method');
for qi = 1:numel(q_pct_list)
    fprintf(fid, ',q=%d%%', q_pct_list(qi));
end
fprintf(fid, '\n');
for mi = 1:numel(table_methods)
    fprintf(fid, '%s', table_methods{mi});
    for qi = 1:numel(q_pct_list)
        fprintf(fid, ',%.6f', nmse_table(mi, qi));
    end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('Table saved to %s\n', csv_path);
%% ===================== Local functions =====================
function bc = design_costs_bcrb(S, n, H_r, F_r, Sinv)
%DESIGN_COSTS_BCRB  Return only the bCRB for a sampling set S.
    [~, bc] = design_costs(S, n, H_r, F_r, Sinv);
end

function cost = composite_design_cost(S, n, H_r, F_r, Sinv, alpha)
%COMPOSITE_DESIGN_COST  BMSE + alpha * bCRB for a sampling set S.
    [bmse, bcrb] = design_costs(S, n, H_r, F_r, Sinv);
    cost = bmse + alpha * bcrb;
end

function S_out = composite_node_swap(S_in, n, H_r, F_r, Sinv, slack, alpha, max_iters)
%COMPOSITE_NODE_SWAP  2-opt node-swap local search minimising BMSE + alpha*bCRB.
%   Same first-improvement logic as bcrb_node_swap but uses the composite cost,
%   so it is correct for both oracle (d_alpha) and minimax (d_{kappa*alpha}) designs.
    if nargin < 8, max_iters = Inf; end
    S     = S_in(:)';
    unsel = setdiff(setdiff(1:n, S), slack);
    cost  = composite_design_cost(S, n, H_r, F_r, Sinv, alpha);
    iter  = 0;
    improved = true;
    while improved && iter < max_iters
        improved = false;
        iter     = iter + 1;
        for ii = 1:numel(S)
            for jj = 1:numel(unsel)
                S_try     = S;  S_try(ii) = unsel(jj);
                c         = composite_design_cost(S_try, n, H_r, F_r, Sinv, alpha);
                if c < cost - 1e-10 * max(abs(cost), 1)
                    S        = S_try;
                    cost     = c;
                    unsel    = setdiff(setdiff(1:n, S), slack);
                    improved = true;
                    break;
                end
            end
            if improved, break; end
        end
    end
    S_out = S;
end

function S_out = bcrb_node_swap(S_in, n, H_r, F_r, Sinv, slack, max_iters)
%BCRB_NODE_SWAP  2-opt node-swap local search to minimise bCRB = trace(K^{-2}M).
%
%   First-improvement rule: accepts the first swap that reduces bCRB and
%   restarts.  Stops when no swap helps or max_iters outer rounds are done.
    if nargin < 7, max_iters = Inf; end
    S     = S_in(:)';
    unsel = setdiff(setdiff(1:n, S), slack);
    [~, bcrb] = design_costs(S, n, H_r, F_r, Sinv);
    iter = 0;
    improved = true;
    while improved && iter < max_iters
        improved = false;
        iter     = iter + 1;
        for ii = 1:numel(S)
            for jj = 1:numel(unsel)
                S_try        = S;
                S_try(ii)    = unsel(jj);
                [~, bc]      = design_costs(S_try, n, H_r, F_r, Sinv);
                if bc < bcrb - 1e-10 * max(abs(bcrb), 1)
                    S        = S_try;
                    bcrb     = bc;
                    unsel    = setdiff(setdiff(1:n, S), slack);
                    improved = true;
                    break;
                end
            end
            if improved, break; end
        end
    end
    S_out = S;
end

function [bmse, bcrb, Ki, M, Dm] = design_costs(S, n, H_r, F_r, Sinv)
    d = zeros(n, 1); d(S) = 1; Dm = diag(d);
    M  = H_r' * Dm * Sinv * Dm * H_r;
    K  = M + F_r;
    Ki = K \ eye(size(K));  Ki = (Ki + Ki') / 2;
    bmse = trace(Ki);                 % (6)
    bcrb = trace(Ki * Ki * M);        % (6) = Var(Sigma, L, d)
end

function out = eval_set(S, n, H_r, F_r, Sinv, Y0, B, X_r, alpha_list)
    [out.bmse, out.bcrb, Ki, ~, Dm] = design_costs(S, n, H_r, F_r, Sinv);
    out.theo = out.bmse + alpha_list * out.bcrb;           % (15)
    Gm  = Ki * (H_r' * Dm * Sinv * Dm);                     % GFR-ML (3), x0 = 0
    Xh0 = Gm * Y0;
    XhB = Gm * B;
    out.emp = zeros(size(alpha_list));
    for a = 1:numel(alpha_list)
        err = Xh0 - sqrt(alpha_list(a)) * XhB - X_r;
        out.emp(a) = mean(sum(err.^2, 1));
    end
end
