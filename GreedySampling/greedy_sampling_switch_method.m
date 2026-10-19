function [SamplingSet] = greedy_sampling_switch_method(V_k, L, hL, fL, ...
    methods, idx, mu, sigma, slack, num_sensors)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    V_k
    L
    hL
    fL
    methods
    idx
    mu
    sigma
    slack
    num_sensors = length(L)
end

arguments (Output)
    SamplingSet
end
n = length(L);
switch methods(idx).name
    case 'A-design'
        SamplingSet = A_optimal_design(num_sensors, V_k);
    case 'E-design'
        SamplingSet = E_optimal_design(num_sensors, V_k);
    case 'D-design'
        SamplingSet = D_optimal_design(num_sensors, V_k);
    case 'T-design'
        SamplingSet = T_optimal_design(num_sensors, V_k);
    case 'LR-design'
        SamplingSet = LR_optimal_design(eye(n), L, num_sensors, mu, sigma, slack);
    case 'Random'% 6
        SamplingSet = zeros(1, num_sensors);
    case 'b. CRB' % 7
        cost_func = @Calc_BCRB;
        cost_func_efficient = @Calc_BCRB_efficient;
        SamplingSet = GreedySampling(hL, mu.*fL, num_sensors, sigma^(-2), slack, cost_func, cost_func_efficient);

    case 'WC-MSE' % 8
        cost_func = @Calc_WMSE;
        cost_func_efficient = @Calc_WMSE_efficient;
        SamplingSet = GreedySampling(hL, mu.*fL, num_sensors, sigma^(-2), slack, cost_func, cost_func_efficient);

    case 'BMSE' % 9
        cost_func = @Calc_BMSE;
        cost_func_efficient = @Calc_BMSE_efficient;
        SamplingSet = GreedySampling(hL, mu.*fL, num_sensors, sigma^(-2), slack, cost_func, cost_func_efficient);

    case 'WC-BMSE' % 10
        cost_func = @Calc_WBMSE;
        SamplingSet = GreedySampling(hL, mu.*fL, num_sensors, sigma^(-2), slack, cost_func);

    otherwise
        % Use error() with a formatted string for debugging
        error('SamplingMethod:UnknownName', ...
              'The method "%s" is not recognized. Please check getPlottingConstants().', ...
              methods(idx).name);
end
end