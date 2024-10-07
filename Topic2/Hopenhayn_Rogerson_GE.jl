using SparseArrays 
using Parameters
using LinearAlgebra
using Distributions
using Plots
@with_kw mutable struct model
    J = 200
    sig = 0.41
    mu = -0.001
    zg = range(0.001,100,length=J)
    dz = zg[2] - zg[1]
    alph = 0.64
    cf = 1
    r = 0.05
    L = 1
    underv = 0.0
    xi = 10
    psig = entry_dist(xi,zg)
    ce = 0.001
    max_iter = 1e3
end

function entry_dist(xi,zg)
    d = Pareto(xi, 1)
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

function Howard_Algorithm(param,B,pig)
    @unpack_model param
    iter = 1;
    vold = zeros(length(zg));
    vnew = copy(vold);
    exit_or_not = []
    while iter < max_iter 
        val_noexit =  (B*vold .- pig);
        val_exit = vold  .- underv
        exit_or_not =val_noexit  .> val_exit;
        Btilde = B.*(1 .-exit_or_not) + I(J).*(exit_or_not)
        q = pig.*(1 .-exit_or_not) + underv.*(exit_or_not)
        vnew = Btilde\q;
        if norm(vnew - vold) < 1e-6
            break
        end
        vold = copy(vnew)
    end
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
    return v,underz_index,ng,exit_or_not
end

function solve_w(param)
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = 0
    v = [];
    ng = [];
    exit_or_not = [];
    while iter < 1000 && abs(err_free_entry) > 1e-6
        w = (w_ub + w_lb)/2
        v, underz_index,ng,exit_or_not = solve_HJB_VI(param,w)
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
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,exit_or_not=exit_or_not)
end

function solve_stationary_distribution(param,HJB_result)
    @unpack_model param
    @unpack exit_or_not = HJB_result
    D = spdiagm(0 => exit_or_not)
    I_D = I-D;
    A = populate_A(param)
    tildeA = A*I_D + D;
    B = -I_D*psig;  
    hatg = (tildeA')\B;
    m = L/sum(hatg.*ng.*dz)
    g = hatg.*m
    return g
end

param = model()
HJB_result = solve_w(param)
hatg = solve_stationary_distribution(param,HJB_result)



v = HJB_result.v
w = HJB_result.w
ng = HJB_result.ng
underz_index = HJB_result.underz_index
plot(zg,v)
zg[2]
g = solve_stationary_distribution(param,HJB_result)


entry_rate = sum(m.*psig[underz_index:end].*dz)/sum(g.*dz)
plot(zg,g)

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
