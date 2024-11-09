

function solve_w(param; calibration=0,tol=1e-8 )
    @unpack_model param
    w_ub = 20;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    free_entry = 100
    iter = 0
    v = [];
    ce_calibrate = [];
    HJB_result = []
    while iter < max_iter && abs(free_entry) > tol
        if calibration != 0
            w = 0.78;
        else
            w = (w_ub + w_lb)/2
        end
        HJB_result = solve_HJB_QVI(param,w)
        v = HJB_result.v
        v_vec = reshape(v,Jn*Jz)
        free_entry = sum(v_vec.*tilde_psig_nz) - ce
        if free_entry > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," excess labor demand: ",free_entry)
        iter += 1
        if calibration != 0
            ce_calibrate = sum(v_vec.*tilde_psig_nz)
            break
        end
    end
    @assert iter < max_iter
    
    return (w=w, HJB_result=HJB_result,ce_calibrate=ce_calibrate)
end