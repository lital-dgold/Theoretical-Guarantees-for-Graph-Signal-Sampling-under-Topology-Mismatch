function [outputArg1] = pseudo_inverse(inputMat)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[U,S,V] = svd(inputMat);
diag_mat = diag(S);
thr = 10^(-10);
diag_mat(diag_mat > thr) = 1./ diag_mat(diag_mat > thr);
outputArg1 = U*diag(diag_mat)*V';
end