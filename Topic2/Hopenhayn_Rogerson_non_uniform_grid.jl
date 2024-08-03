using SparseArrays 
using Parameters
using LinearAlgebra
using LCPsolve
using Distributions
using Plots

@with_kw mutable struct model
    J = 1000
    sig = 0.3
    zeta = 1.05
    mu = sig^2*(1-zeta)/2
    #zeta = 1 - 2*mu/sig^2
    #mu = -0.000001
    lzg = range(log(0.1),10,length=J)
    #zg = range(0.00001,10,length=J)
    zg = exp.(lzg)
    dz = diff(zg)
    alph = 0.64
    cf = 0.1
    r = 0.05
    L = 1
    underv = 0.0
    xi = 1.5
    psig = entry_dist(xi,zg)[1]
    tilde_psig = entry_dist(xi,zg)[2]
    tilde_dn = entry_dist(xi,zg)[3]
    ce = 1
end
function entry_dist(xi,zg)
    d = Pareto(xi, 0.1)
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
    return psig,tilde_psig,tilde_dn
end
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

function solve_HJB_VI(param,w)
    @unpack_model param
    A = populate_A(param)
    B = (r.*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    q = -pig + underv.*B*ones(length(zg))
    result = solve!(LCP(B,q),max_iter=1000)
    println(result.converged)
    x = result.sol
    v = x .+ underv
    first_positive = findfirst(x .> 0 )
    if isnothing(first_positive)
        underz_index = J
    else
        underz_index = findfirst(x .> 0 )
    end
    return v,underz_index,ng
end

function solve_w(param; calibration=0 )
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = 0
    v = 0
    ng = 0
    ce_calibrate = 0
    while iter < 1000 && abs(err_free_entry) > 1e-6
        if calibration != 0
            w = 1;
        else
            w = (w_ub + w_lb)/2
        end
        v,underz_index,ng = solve_HJB_VI(param,w)
        err_free_entry = sum(v.*tilde_psig) - ce
        if err_free_entry > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," err_free_entry: ",err_free_entry)
        iter += 1
        if calibration != 0
            ce_calibrate = sum(v.*tilde_psig)
            break
        end
    end
    @assert iter < 1000
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,ce_calibrate = ce_calibrate)
end

function solve_stationary_distribution(param,underz_index)
    @unpack_model param
    A = populate_A(param)
    
    A_truncate = A[underz_index:end,underz_index:end]
    B = -tilde_psig[underz_index:end];  
    g = (A_truncate')\B;

    g = [zeros(underz_index-1);g]
    return g
end

param = model()
@unpack_model param
HJB_result = solve_w(param; calibration=1)
v = HJB_result.v
w = HJB_result.w
ng = HJB_result.ng
underz_index = HJB_result.underz_index
ce_calibrate = HJB_result.ce_calibrate
param.ce = ce_calibrate
tilde_hatg = solve_stationary_distribution(param,underz_index)
plot(zg,tilde_hatg)
m = L/sum(tilde_hatg.*ng)
tilde_g = tilde_hatg.*m
entry_rate = sum(m.*tilde_psig[underz_index:end])/sum(tilde_g)


entrants_size = sum(zg[underz_index:end].*tilde_psig[underz_index:end])/sum(tilde_psig[underz_index:end])
ave_size = sum(zg.*tilde_g)/sum(tilde_g)
reverse_cumsum_g = reverse(cumsum(reverse(tilde_g)))/sum(tilde_g)
emp500_cutoff = findlast(reverse_cumsum_g .> 0.0038)
share_500 = sum(ng[emp500_cutoff:end].*tilde_g[emp500_cutoff:end])/sum(ng.*tilde_g)


entrants_size/ave_size
println("------ Entrants Size / Incumbents Size ------")
println(entrants_size/ave_size)
println("------ Entry Rate ------")
println(entry_rate)
println("------ Share of 500+ Firms ------")
println(share_500)
println("------ cutoff z ------")
println(zg[underz_index])


G = reverse(cumsum(reverse(tilde_g)))
plot(log.(zg),log.(G))
center_index = 800;
plot!(log.(zg),log(G[center_index]) .-zeta*(log.(zg) .-log.(zg[center_index])) )


g = tilde_g./tilde_dn
plot(zg,g./sum(g))
gn = tilde_g.*ng./tilde_dn
plot!(zg,gn./sum(gn))
