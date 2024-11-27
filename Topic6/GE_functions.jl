

function solve_U_theta(param; calibration=0,tol=1e-8, U=10)
    @unpack_model param
    
    targeted_condition = 100
    iter = 0
    S = [];
    ce_calibrate = [];
    HJB_result = [];
    dist_result = [];
    theta_lb = 0;
    theta_ub = 10;
    theta = 1;
    while iter < max_iter && abs(targeted_condition) > tol
        if calibration == 1
            theta = 1;
        else
            theta = (theta_ub + theta_lb)/2
        end
        HJB_result = solve_HJB_QVI(param,theta,U)
        S = HJB_result.S
        S_vec = reshape(S,Jn*Jz)
        if nu == Inf
            free_entry = sum(S_vec.*tilde_psig_nz.*(1-gamma)) - ce
            targeted_condition = copy(free_entry)
            println("iter: ",iter," theta: ",theta," free entry error: ",targeted_condition)
            
        else
            m = (sum(S_vec.*tilde_psig_nz.*(1-gamma))./ce)^(nu)
            dist_result = solve_stationary_distribution(param,HJB_result,m=m)
            tildeg_nonuniform = dist_result.tildeg_nonuniform
            vacancy = HJB_result.vacancy
            employment = sum(ng_repeat.*tildeg_nonuniform);
            vacancy_vec = reshape(vacancy,Jn*Jz)
            agg_vacancy = sum(ng_repeat.*tildeg_nonuniform) - L
            if employment > L
                theta_new = 1e10;
            else
                theta_new = agg_vacancy/(L-employment)
            end
            targeted_condition = copy(theta_new-theta)
            println("iter: ",iter," theta: ",theta," excess labor demand: ",excess_labor_demand)
        end
        if targeted_condition > 0
            theta_lb = theta
        else
            theta_ub = theta
        end

        iter += 1
        if calibration != 0 && nu == Inf
            ce_calibrate = sum(S_vec.*tilde_psig_nz.*(1-gamma))
            break
        end
    end
    @assert iter < max_iter

    if nu == Inf
        dist_result = solve_stationary_distribution(param,HJB_result)
    end

    @unpack exit_or_not,pig = HJB_result
    @unpack tildeg_nonuniform,m = dist_result

    tilde_psig_nz_mat = reshape(tilde_psig_nz,Jn,Jz)
    
    entry_rate = sum(m.*tilde_psig_nz_mat.*( 1 .- exit_or_not))/sum(tildeg_nonuniform)

    tildeg_nonuniform_mat = reshape(tildeg_nonuniform,Jn,Jz)
    tildeg_nonuniform_n = sum(tildeg_nonuniform_mat,dims=2)
    ave_size = sum(ng.*tildeg_nonuniform_n)/sum(tildeg_nonuniform_n)

            
    println("--------- Average Size -----------")
    println(ave_size)
    println("------ Entry Rate ------")
    println(entry_rate)
 
    # compute implied b
    ng_repeat = repeat(ng,Jz)
    S = HJB_result.S
    S_vec = reshape(S,Jn*Jz)
    b = r*U - lambdafun(theta)*gamma*sum(S_vec./ng_repeat.*tildeg_nonuniform)
    
    return (
            theta= theta, 
            HJB_result = HJB_result,
            ce_calibrate = ce_calibrate,
            dist_result = dist_result,
            entry_rate=entry_rate,
            ave_size = ave_size,
            b = b
    )
end