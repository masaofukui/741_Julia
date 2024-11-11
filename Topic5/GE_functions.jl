

function solve_w(param; calibration=0,tol=1e-8 )
    @unpack_model param
    w_ub = 20;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    targeted_condition = 100
    iter = 0
    v = [];
    ce_calibrate = [];
    HJB_result = [];
    dist_result = [];
    while iter < max_iter && abs(targeted_condition) > tol
        if calibration != 0
            w = 0.78;
        else
            w = (w_ub + w_lb)/2
        end
        HJB_result = solve_HJB_QVI(param,w)
        v = HJB_result.v
        v_vec = reshape(v,Jn*Jz)
        if nu == Inf
            free_entry = sum(v_vec.*tilde_psig_nz) - ce
            targeted_condition = copy(free_entry)
            println("iter: ",iter," w: ",w," free entry error: ",targeted_condition)
        else
            m = (sum(v_vec.*tilde_psig_nz)./ce)^(nu)
            dist_result = solve_stationary_distribution(param,HJB_result,m=m)
            tildeg_nonuniform = dist_result.tildeg_nonuniform
            ng_repeat = repeat(ng,Jz)
            excess_labor_demand = sum(ng_repeat.*tildeg_nonuniform) - L
            targeted_condition = copy(excess_labor_demand)
            println("iter: ",iter," w: ",w," excess labor demand: ",excess_labor_demand)
        end
        if targeted_condition > 0
            w_lb = w
        else
            w_ub = w
        end
        iter += 1
        if calibration != 0 && nu == Inf
            ce_calibrate = sum(v_vec.*tilde_psig_nz)
            break
        end
    end
    @assert iter < max_iter

    if nu == Inf
        dist_result = solve_stationary_distribution(param,HJB_result)
    end
    
    return (w = w, 
            HJB_result = HJB_result,
            ce_calibrate = ce_calibrate,
            dist_result = dist_result)
end