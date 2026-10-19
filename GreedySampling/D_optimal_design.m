function [sampling_set] = D_optimal_design(m, V_k)
% implementation of the D-optimal set design 
% m - number of samples
% V_k - first k eigenvectors of the graph Laplacian
% D-optimal design maximizes the determinant of V_k(S,:)' * V_k(S,:)

[n, k] = size(V_k);
left = 1:n;
sampling_set = [];

for i = 1:m
    metric = zeros(length(left), 1);
    
    for j = 1:length(left)
        % Create temporary candidate set
        current_indices = [sampling_set, left(j)];
        
        % Get singular values of the candidate submatrix
        % We use svd because it is stable for nonsquare matrices
        s = svd(V_k(current_indices, :), 'econ');
        
        % D-optimality criterion: Product of the squared singular values
        % This is equivalent to det(V_k(S,:)' * V_k(S,:))
        % We use log-sum if m is very large to avoid numerical overflow, 
        % but for standard graph sampling, the product is fine.
        metric(j) = prod(s.^2);
    end
    
    % Select the index that maximizes the determinant
    [~, idx] = max(metric);
    
    % Update sets
    sampling_set = [sampling_set, left(idx)];
    left(idx) = []; % Remove selected index from candidates
end
end