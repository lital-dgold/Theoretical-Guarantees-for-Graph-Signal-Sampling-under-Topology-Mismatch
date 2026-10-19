function [outputArg1] = Calc_BMSE_efficient(inv_K, r, fL)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
z = inv_K * r;
outputArg1 = trace(inv_K)-z'*z /(1+r'*z);
end