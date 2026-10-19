function [sampling_set] = A_optimal_design(m ,V_k )
%implementation of the E optimal set design from 
%"Discrete Signal Processing on Graphs: Sampling Theory"
%   m - number of samples
%   k - number of nonzero fourier coefitions (m>=k)
%   V - eigenvectors of L, when we sort the eigenvalues of L in a descending order
n=length(V_k);
left=1:n;
sampling_set=[];
for i=1:m
    sigma=zeros(n-i+1,1);
    for j=1:n-i+1
        sigma(j) = trace(pseudo_inverse(V_k([sampling_set,left(j)],:)'*V_k([sampling_set,left(j)],:)));
    end
    [~,idx]=min(sigma);
    sampling_set=[sampling_set,left(idx)];
    left=left(left~=left(idx));
end
end

