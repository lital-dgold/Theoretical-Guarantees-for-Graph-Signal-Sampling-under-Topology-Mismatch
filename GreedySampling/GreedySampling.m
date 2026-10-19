function [sampling_set, best_cost_array] = GreedySampling(hL, fL, num_sensors, inv_R, slack, cost_func, cost_func_efficient)
    arguments
        hL 
        fL
        num_sensors
        inv_R
        slack
        cost_func 
        cost_func_efficient = []
    end
n=size(hL,1);
left=1:n;

vector_slack=1:n;
vector_slack=vector_slack~=slack;

fL = fL(vector_slack',vector_slack);
hL = hL(:,vector_slack);

sampling_set = zeros(1, num_sensors);
temp_sampling_set = [];
best_cost_array = zeros(n,1);


[inv_R_cols,inv_R_rows] = size(inv_R);
max_dim_R = max(inv_R_cols,inv_R_rows);
min_dim_R = min(inv_R_cols,inv_R_rows);
if ~isempty(cost_func_efficient) &&  max_dim_R == 1
    inv_R = inv_R.*ones(n,1);
    inv_R_mat = diag(inv_R);
else
    inv_R_mat = inv_R;
end


for i=1:num_sensors
    cost_values = zeros(n-i+1,1);
    if ~isempty(cost_func_efficient) && (i > 1) && (min_dim_R == 1)
        v = samplingSet2vector(temp_sampling_set, n)';
        diag_v = diag(v);
        K = hL' * diag_v * diag(inv_R) *diag_v * hL + fL;
        inv_K = pseudo_inverse(K);
    end
    for j=1:n-i+1
        idx = left(j);
        if ~isempty(cost_func_efficient) &&  (i > 1) && (min_dim_R == 1)
            inv_sigma = sqrt(inv_R(idx));
            r = inv_sigma * hL(idx,:)';
            cost_values(j) = cost_func_efficient(inv_K, r, fL);
        else
            v = samplingSet2vector([temp_sampling_set, idx], n)';
            cost_values(j) = cost_func(v, hL, fL, inv_R_mat);
        end    
    end
    [min_mse,idx] = min(cost_values);
    sampling_set(i) = left(idx);
    temp_sampling_set = sampling_set(1:i);
    left = left(left~=left(idx));
    best_cost_array(i) = min_mse;
end
end
