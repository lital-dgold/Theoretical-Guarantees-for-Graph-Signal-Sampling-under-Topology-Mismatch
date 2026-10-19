function [sampling_set] = E_optimal_design(m, V_k)
%implementation of the E optimal set design from 
%"Discrete Signal Processing on Graphs: Sampling Theory"
%   m - number of samples
%   k - number of nonzero fourier coefitions (m>=k)
%   V - eigenvectors of L, when we sort the eigenvalues of L in a descending order
[n,k]=size(V_k);
left=1:n;
sampling_set=[];
for i=1:m
    sigma=zeros(n-i+1,1);
    for j=1:n-i+1
        [~,s,~] = svd(V_k([sampling_set,left(j)],:)); % The singular values are in descending order
        square_s=s(1:min(i,k),1:min(i,k));% Ignoring the zero eigenvalues
        sigma(j) = min(diag(square_s));%s(min(i,k),min(i,k));% minimal sigma
    end
    [~,idx]=max(sigma);
    sampling_set=[sampling_set,left(idx)];
    left=left(left~=left(idx));
end
end

