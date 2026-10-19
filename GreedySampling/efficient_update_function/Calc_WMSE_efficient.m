function [outputArg1] = Calc_WMSE_efficient(inv_K, r, fL)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
BCRB = Calc_BCRB_efficient(inv_K, r, fL);
z = inv_K * r;
max_svd_term = (inv_K-(z*z' ./(1+r'*z))) * fL;
outputArg1 = BCRB + svds(max_svd_term, 1)^2;
end