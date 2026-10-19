function [sampling_set] = T_optimal_design(m, V_k)
%implementation of the E optimal set design from 
%"Discrete Signal Processing on Graphs: Sampling Theory"
%   m - number of samples
%   k - number of nonzero fourier coefitions (m>=k)
%   V - eigenvectors of L, when we sort the eigenvalues of L in a descending order
[n,k]=size(V_k);
norm_v_k = diag(V_k*V_k');
[~, idx] = sort(norm_v_k, 'descend'); 
sampling_set = idx(1:m);
end
