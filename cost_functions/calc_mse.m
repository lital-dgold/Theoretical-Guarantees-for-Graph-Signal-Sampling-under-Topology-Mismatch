function mse = calc_mse(parameter, est, slack)
    if slack == 0
        est_no_DC = est;
        parameter_no_DC = parameter;
    else
        est_no_DC = est-est(slack, :);
        parameter_no_DC = parameter-parameter(slack, :);
    end
    error = est_no_DC-parameter_no_DC;
    num_samples = size(est, 2);
    mse = sum(error(:).^2) / num_samples;
end
