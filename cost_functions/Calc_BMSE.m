function [BMSE] = Calc_BMSE(v, hL, fL, inv_R)
    if size(v,1)< size(v,2)
        v=v';
    end
        diag_v = diag(v);
        term1 = hL' * diag_v * inv_R *diag_v * hL + fL;
        term1_inv = pseudo_inverse(term1);
        BMSE = trace(term1_inv);
    % if size(v,1)< size(v,2)
    %     v=v';
    % end
    %     diag_v = diag(v);
    %     term1 = hL * diag_v * inv_R * hL + fL;
    %     term1_inv = inv(term1);
    %     BMSE = trace(term1_inv);
end