function [V_new, S_new] = ensure_incresing_order(V,S)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[b,ind] = sort(diag(S));
S_new = diag(b);
V_new = zeros(size(V));
for i=1:length(V)
    V_new(:,i)=V(:,ind(i));
end
end

