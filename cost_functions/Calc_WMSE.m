function wmse = Calc_WMSE(v, hL, fL, inv_R)
    diag_v = diag(v);

    % Compute the term (h(L) * diag(v) * h(L) + f(L))
    term1 = hL' * diag_v * inv_R *diag_v * hL + fL;
    inv_term1 = pseudo_inverse(term1);
    squared_inverse = inv_term1 * inv_term1;
    bias = inv_term1 * hL' * diag_v * inv_R *diag_v * hL - eye(length(fL));
    lambda_max = max(svd(bias'*bias));
    wmse = trace(squared_inverse * hL' * diag_v * inv_R *diag_v * hL) + lambda_max;
end