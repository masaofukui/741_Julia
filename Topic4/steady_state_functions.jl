function populate_A(param)
    @unpack_model param
    A = spzeros(length(zg),length(zg))
    
    for (i,z) in enumerate(zg)
        dz_plus = dz[min(i,J-1)]
        dz_minus = dz[max(i-1,1)]
        if mu > 0
            A[i,min(i+1,J)] += mu.*z/dz_plus;
            A[i,i] += -mu.*z/dz_plus;
        else
            A[i,i] += mu.*z/dz_minus;
            A[i,max(i-1,1)] += -mu.*z/dz_minus;
        end
        denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;
        A[i,i] += - 1/2*(dz_plus + dz_minus).*(sig*z)^2/denom;
        A[i,max(i-1,1)] += 1/2*dz_plus.*(sig*z)^2/denom;
        A[i,min(i+1,J)] += 1/2*dz_minus.*(sig*z)^2/denom;
    end 
    return A
end

function Howard_Algorithm(param,B,pig;vinit=0)
    @unpack_model param
    iter = 1;
    if vinit == 0
        vold = zeros(length(zg));
    else
        vold = copy(vinit)
    end
    vnew = copy(vold);
    exit_or_not = []
    while iter < max_iter 
        val_noexit =  (B*vold .- pig);
        val_exit = vold  .- underv
        exit_or_not =val_noexit  .> val_exit;
        D = spdiagm(0 => exit_or_not)
        Btilde = (I-D)*B + D
        q = pig.*(1 .-exit_or_not) + underv.*(exit_or_not)
        vnew = Btilde\q;
        if norm(vnew - vold) < 1e-6
            break
        end
        vold = copy(vnew)
    end
    @assert iter < max_iter "Howard Algorithm did not converge"
    return vnew,exit_or_not
end

function solve_HJB_VI(param,w)
    @unpack_model param
    A = populate_A(param)
    B = (r.*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    v,exit_or_not = Howard_Algorithm(param,B,pig)
    underz_index = findlast(exit_or_not .> 0 )
    if isnothing(underz_index)
        underz_index = 1
    end
    return (v = v,underz_index = underz_index,
    ng = ng,exit_or_not =exit_or_not)
end

function solve_w(param; calibration=0,tol=1e-8 )
    @unpack_model param
    w_ub = 20;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    excess_labor = 100
    iter = 0
    underz_index = [];
    v = [];
    ng = [];
    ce_calibrate = [];
    exit_or_not = [];
    tildeg_nonuniform = [];
    m = []
    while iter < max_iter && abs(excess_labor) > tol
        if calibration != 0
            w = 0.78;
        else
            w = (w_ub + w_lb)/2
        end
        HJB_result = solve_HJB_VI(param,w)
        @unpack v,underz_index,ng,exit_or_not = HJB_result
        m = (sum(v.*tilde_psig)./ce)^(nu)
        dist_result = solve_stationary_distribution(param,HJB_result,m)
        @unpack tildeg_nonuniform = dist_result
        excess_labor = sum(ng.*tildeg_nonuniform) - L
        
        if excess_labor > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," excess labor demand: ",excess_labor)
        iter += 1
        if calibration != 0
            ce_calibrate = sum(v.*tilde_psig)
            break
        end
    end
    @assert iter < max_iter
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,ce_calibrate = ce_calibrate,
    exit_or_not = exit_or_not,
    tildeg_nonuniform=tildeg_nonuniform,
    m=m)
end

function solve_stationary_distribution(param,HJB_result,m)
    @unpack_model param
    @unpack exit_or_not = HJB_result
    D = spdiagm(0 => exit_or_not)
    I_D = I-D;
    A = populate_A(param)
    A = A - spdiagm(0 => eta*ones(J))
    tildeA = A*I_D + D;
    B = -I_D*tilde_psig;  
    tildeg_nonuniform_normalized = (tildeA')\B;
    tildeg_nonuniform = tildeg_nonuniform_normalized.*m

    return (tildeg_nonuniform=tildeg_nonuniform,
            tildeg_nonuniform_normalized=tildeg_nonuniform_normalized)
end

function compute_calibration_targets(param,ss_result)
    @unpack_model param
    @unpack underz_index,ng,tildeg_nonuniform,m,w = ss_result
    
    entry_rate = sum(m.*tilde_psig[underz_index:end])/sum(tildeg_nonuniform)
    Entry = sum(m.*tilde_psig[underz_index:end]);

    entrants_size = sum(ng[underz_index:end].*tilde_psig[underz_index:end])/sum(tilde_psig[underz_index:end])
    ave_size = sum(ng.*tildeg_nonuniform)/sum(tildeg_nonuniform)
    reverse_cumsum_g = reverse(cumsum(reverse(tildeg_nonuniform)))/sum(tildeg_nonuniform)
    emp500_cutoff = findlast(reverse_cumsum_g .> 0.0038)
    share_500 = sum(ng[emp500_cutoff:end].*tildeg_nonuniform[emp500_cutoff:end])/sum(ng.*tildeg_nonuniform)
    Firm_mass = sum(tildeg_nonuniform)
    Exit_rate = entry_rate - eta
    Exit = Exit_rate*Firm_mass


    println("--------- Average Size -----------")
    println(ave_size)
    println("------ Entrants Size / Incumbents Size ------")
    println(entrants_size/ave_size)
    println("------ Entry Rate ------")
    println(entry_rate)
    println("------ Share of 500+ Firms ------")
    println(share_500)
    println("------ cutoff z ------")
    println(zg[underz_index])
    println("------ cutoff n ------")
    println(ng[underz_index])

    ss_stats = Dict{String,Any}()
    ss_stats["Entry_rate"] = entry_rate
    ss_stats["Entry"] = Entry
    ss_stats["entrants_size"] = entrants_size
    ss_stats["Firm_size"] = ave_size
    ss_stats["share_500"] = share_500
    ss_stats["w"] = w
    ss_stats["Firm_mass"] = Firm_mass
    ss_stats["Exit"] = Exit
    ss_stats["Exit_rate"] = Exit_rate


    return ss_stats
end