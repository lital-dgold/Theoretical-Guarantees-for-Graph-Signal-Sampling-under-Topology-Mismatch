function [sampling_set, crb_array] = LR_optimal_design(filter,inv_prior,K,mu,sigma,slack)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
n=size(filter,1);
left=1:n;
sampling_set=[];
vector_slack=1:n;
vector_slack=vector_slack~=slack;
L_squard=inv_prior(vector_slack',vector_slack);
crb_array = zeros(n,1);
for i=1:K
    MSE=zeros(n-i+1,1);
    for j=1:n-i+1
        L_s = filter(samplingSet2vector([sampling_set,left(j)], n)',vector_slack);
        MSE(j)=Calc_worst_Bayesian_CRB(L_s,L_squard,mu,sigma);
    end
    [min_mse,idx]=min(MSE);
    sampling_set=[sampling_set,left(idx)];
    left=left(left~=left(idx));
    crb_array(i) = min_mse;
end

end

function [CRB] = Calc_worst_Bayesian_CRB(H_pnm,L_squard,mu,sigma)
    K=(H_pnm'*H_pnm+mu*(sigma^2)*L_squard);%+0.0001*eye(size(H_pnm,2))
    svd_vec = svd(K);
    stability_subspace = svd_vec > 0.0000001;
    if sum(stability_subspace) > 0
        CRB = 1/min(svd_vec(stability_subspace));
    else
        CRB = inf;
    end
    % CRB= - min(svd(K));%(sigma^2)*trace((H_pnm)*k*k*H_pnm') + lambda_max;
end