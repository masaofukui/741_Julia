using SparseArrays 
using Parameters
using LinearAlgebra
using LCPsolve
using Distributions
using Plots
@with_kw mutable struct model
    J = 200
    sig = 2
    mu = -0.00000001
    zg = range(0.1,1,length=J)
    dz = zg[2] - zg[1]
    alph = 0.64
    cf = 0.1
    r = 0.05
    L = 1
    underv = 0.0
    xi = 20
    psig = entry_dist(xi,zg)
    ce = 10
end
function entry_dist(xi,zg)
    d = Pareto(xi, 0.1)
    psig = diff(cdf(d,zg))
    psig = [psig; psig[end]]
    psig = psig./sum(psig)./(zg[2]-zg[1])
    return psig
end
function populate_A(param)
    @unpack_model param
    A = spzeros(length(zg),length(zg))
    for (i,z) in enumerate(zg)
        if mu > 0
            A[i,min(i+1,J)] += mu.*z/dz;
            A[i,i] += -mu.*z/dz;
        else
            A[i,i] += mu.*z/dz;
            A[i,max(i-1,1)] += -mu.*z/dz;
        end
        A[i,i] += - (sig*z)^2/dz^2;
        A[i,max(i-1,1)] += 1/2*(sig*z)^2/dz^2;
        A[i,min(i+1,J)] += 1/2*(sig*z)^2/dz^2;
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

function solve_w(param)
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = 0
    v = 0
    ng = 0
    while iter < 1000 && abs(err_free_entry) > 1e-6
        w = (w_ub + w_lb)/2
        v,underz_index,ng = solve_HJB_VI(param,w)
        err_free_entry = sum(v.*psig.*dz) - ce
        if err_free_entry > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," err_free_entry: ",err_free_entry)
        iter += 1
    end
    @assert iter < 1000
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng)
end

function solve_stationary_distribution(param,underz_index)
    @unpack_model param
    A = populate_A(param)
    
    A_truncate = A[underz_index:end,underz_index:end]
    B = -psig[underz_index:end];  
    g = (A_truncate')\B;

    g = [zeros(underz_index-1);g]
    return g
end

param = model()
@unpack_model param

HJB_result = solve_w(param)
v = HJB_result.v
w = HJB_result.w
ng = HJB_result.ng
underz_index = HJB_result.underz_index
plot(zg,v)
zg[2]
hatg = solve_stationary_distribution(param,underz_index)
m = L/sum(hatg.*ng.*dz)
g = hatg.*m
entry_rate = sum(m.*psig[underz_index:end].*dz)/sum(g.*dz)


entrants_size = sum(zg[underz_index:end].*psig[underz_index:end].*dz)/sum(psig[underz_index:end].*dz)
ave_size = sum(zg.*g.*dz)/sum(g.*dz)

entrants_size/ave_size
reverse_cumsum_g = reverse(cumsum(reverse(g)))/sum(g)
emp500_cutoff = findlast(reverse_cumsum_g .> 0.0038)
sum(ng[emp500_cutoff:end].*g[emp500_cutoff:end].*dz)/sum(ng.*g.*dz)

plot(zg,g./sum(g))
plot!(zg,(g.*ng)./sum(g.*ng))





zg = range(1,20,length=10000)
dz = zg[2] - zg[1]
zeta = 2
analytical_z = zeta.*(zg./zg[1]).^(-zeta-1)
plot!(zg,analytical_z./sum(analytical_z))

emp500_cutoff = 10
sum(zg[emp500_cutoff:end].*analytical_z[emp500_cutoff:end].*dz)/sum(zg.*analytical_z.*dz)


emp500_cutoff = 1
sum(zg[emp500_cutoff:end].*analytical_z[emp500_cutoff:end].*dz)
zeta/(zeta-1)*(zg[1])^zeta.*(zg[emp500_cutoff]).^(1-zeta)


plot(zg,analytical_z)
