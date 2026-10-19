function lambda_max = Calc_WBMSE(v, hL, fL, inv_R)
        if size(v,1)< size(v,2)
            v=v';
        end
        diag_v = diag(v);
        term1 = hL' * diag_v * inv_R * diag_v * hL + fL;
        % lambda_max = (min(svd(term1)))^(-1);
        svd_vec = svd(term1);
        stability_subspace = svd_vec > 0.0000001;
        if sum(stability_subspace) > 0
            lambda_max = 1/min(svd_vec(stability_subspace));
        else
            lambda_max = inf;
        end

end
