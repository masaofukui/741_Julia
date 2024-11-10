using SparseArrays 
using Parameters
using LinearAlgebra
using LCPsolve
using Distributions
using Plots
using Plots.Measures

fig_save = 0;

@with_kw mutable struct model
    Jn = 100
    Jz = 120
    sig = 0.41
    zeta = 1.05
    mu = -0.01
    lng = range(log(0.1),log(2300000),length=Jn)
    lzg = range(log(0.1),log(2300000),length=Jz)
    ng = exp.(lng)
    zg = exp.(lzg)
    Delta_z = diff(zg)
    Delta_n = diff(ng)
    alph = 0.64
    cf = 0.5
    r = 0.05
    L = 1
    underv = 0.0
    xi = 1.1
    ce = 1
    tilde_Delta_z = [ Delta_z[1]; [(Delta_z[i]+Delta_z[i+1])/2 for i in 1:(Jz-2)]; Delta_z[end]]
    n_start = 1
    tilde_psig_nz = entry_dist(xi,ng,zg,tilde_Delta_z,n_start);
    max_iter = 1e3
    eta = 0.00
    dt = 2
    tg = 0:dt:300
    T = length(tg)
    nu = 4
    phi = 2
    s = 0.08
    g_fun = h -> phi/2 .* h.^2
    h_fun = (dv) -> max.(dv,0) ./phi
end


include("plot_functions.jl")
include("HJB_functions.jl")
include("KFE_functions.jl")
include("subfunctions.jl")

param = model()
@unpack_model param
w=1
result_free_entry = solve_w(param,calibration=1)
w = result_free_entry.w
HJB_result = result_free_entry.HJB_result
ce_calibrate = result_free_entry.ce_calibrate
dist_result = solve_stationary_distribution(param,HJB_result)
tildeg_nonuniform = reshape(dist_result.tildeg_nonuniform,Jn,Jz)
tildeg_nonuniform_n = sum(tildeg_nonuniform,dims=2)
plot(log.(ng),log.(tildeg_nonuniform_n))


result = solve_w(param,calibration=0)


@time HJB_result = solve_HJB_QVI(param,w)

