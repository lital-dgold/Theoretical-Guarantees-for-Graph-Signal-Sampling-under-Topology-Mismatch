function [outputArg1] = Calc_BCRB_efficient(inv_K, r, fL)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
BMSE = Calc_BMSE_efficient(inv_K, r, fL);
z = inv_K * r;
square_term = (inv_K-(z*z' ./(1+r'*z)))^2;
outputArg1 = BMSE - trace(square_term * fL);
end