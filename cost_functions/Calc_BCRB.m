function biased_crb = Calc_BCRB(v, hL, fL, inv_R)
    diag_v = diag(v);
    % Compute the term (h(L) * diag(v) * h(L) + f(L))
    term1 = hL' * diag_v * inv_R *diag_v * hL + fL;
    inv_term1 = pseudo_inverse(term1);
    squared_inverse = inv_term1 * inv_term1;
    biased_crb = trace(squared_inverse * hL' * diag_v * inv_R *diag_v * hL);
end