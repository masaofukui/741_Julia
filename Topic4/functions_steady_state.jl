
function entry_dist(xi,zg,entry_zlow)
    d = Pareto(xi, entry_zlow)
    dz = diff(zg)
    psig = pdf(d,zg)
    tilde_psig = copy(psig)
    J = length(zg)
    tilde_dn = copy(psig)
    for j = 1:J
        if j == 1
            tilde_dn[j] = 1/2*dz[j]
            tilde_psig[j] = psig[j]*tilde_dn[j]
        elseif j == J
            tilde_dn[j] =1/2*dz[j-1]
            tilde_psig[j] = psig[j]*tilde_dn[j] 
        else
            tilde_dn[j] = 1/2*(dz[j-1] + dz[j])
            tilde_psig[j] = psig[j]*tilde_dn[j]
        end
    end
    tilde_psig = tilde_psig/sum(tilde_psig)
    return psig,tilde_psig
end




function populate_A_HJB(param)
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
function solve_HJB_VI(param,w)
    @unpack_model param
    A = populate_A_HJB(param)
    B = (r.*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    q = -pig + underv.*B*ones(length(zg))
    result = LCPsolve.solve!(LCP(B,q),max_iter=1000)
    #println(result.converged)
    x = result.sol
    v = x .+ underv
   
    underz = []
    first_positive = findfirst(x .> 0 )
    if isnothing(first_positive)
        underz_index = J
        underz = zg[J]
    elseif first_positive == 1
        underz_index = 1
        underz = zg[1]
    else
        underz_index = findfirst(x .> 0 )
        v_noexit = B*v - pig
        v_noexit_interp = linear_interpolation(zg, v_noexit - (v .-underv))
        underz = find_zero(v_noexit_interp, (zg[1],zg[end]))
    end
    return v,underz_index,underz,ng
end

function solve_w(param; calibration=0 )
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    excess_labor = 100
    iter = 0
    underz_index = 0
    v = 0
    ng = 0
    underz = 0
    m = 0;
    tildeg = [];
    while iter < 1000 && abs(excess_labor) > 1e-6
        w = (w_ub + w_lb)/2
        v,underz_index,underz,ng = solve_HJB_VI(param,w)
        m = compute_entry(param,v,underz)
        hattildeg = solve_stationary_distribution(param,underz)
        tildeg = m.*hattildeg;
        excess_labor = sum(ng.*tildeg) - L

        if excess_labor > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," Excess labor demand: ",excess_labor)
        iter += 1
    end
    @assert iter < 1000
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,underz=underz,
    m = m,tildeg=tildeg)
end

function compute_entry(param,v,underz)
    @unpack_model param
    underz_grid_down,weight_down,weight_up = closest_index(zg, underz)
    tilde_psig[1:(underz_grid_down-1)] .= 0
    tilde_psig[underz_grid_down] = weight_down*tilde_psig[underz_grid_down]
    m = M*(sum(v.*tilde_psig)./ce)^(nu)

    return m 
end

function populate_A_KFE(param,underz,eta_in)
    A = populate_A_HJB(param)
    A = spdiagm(-eta_in*ones(J)) + A
    B = -(tilde_psig); 
    underz_grid_down,weight_down,weight_up = closest_index(zg, underz) 
    underz_set = 1:(underz_grid_down-1);
    A[underz_set,:] .= 0;
    A[:,underz_set] .= 0;
    A[underz_set,underz_set] = I(length(underz_set));
    A[:,underz_grid_down] = weight_down.*A[:,underz_grid_down]
    B[underz_set] .= 0;
    B[underz_grid_down] = weight_down*B[underz_grid_down];
    return A,B
end

function solve_stationary_distribution(param,underz)
    @unpack_model param
    A,B = populate_A_KFE(param,underz,eta);
    g = (A')\B;
    return g
end