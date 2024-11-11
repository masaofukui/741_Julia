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
    Jz = 60
    sig = 0.6
    zeta = 1.05
    mu = sig^2*(1-zeta)/2
    lng = range(log(0.001),log(1000),length=Jn)
    lzg = range(log(0.001),log(1000),length=Jz)
    ng = exp.(lng)
    zg = exp.(lzg)
    Delta_z = diff(zg)
    Delta_n = diff(ng)
    alph = 0.64
    cf = 0.1
    r = 0.05
    L = 1
    underv = 0.0
    xi = 1.1
    ce = 0.1
    tilde_Delta_z = [ Delta_z[1]; [(Delta_z[i]+Delta_z[i+1])/2 for i in 1:(Jz-2)]; Delta_z[end]]
    n_start = 1
    tilde_psig_nz = entry_dist(xi,ng,zg,tilde_Delta_z,n_start);
    max_iter = 1e3
    dt = 2
    tg = 0:dt:300
    T = length(tg)
    nu = 5
    phi = 10
    s = 0.00
    g_fun = h -> phi/2 .* h.^2
    h_fun = (dv) -> max.(dv,0) ./phi
end


include("plot_functions.jl")
include("HJB_functions.jl")
include("KFE_functions.jl")
include("GE_functions.jl")
include("subfunctions.jl")

param = model()
@unpack_model param
ss_result = solve_w(param,calibration=0,tol=0.01)
HJB_result = ss_result.HJB_result
dist_result = ss_result.dist_result
tildeg_nonuniform = ss_result.dist_result.tildeg_nonuniform
tildeg_nonuniform_mat = reshape(tildeg_nonuniform,Jn,Jz)
tildeg_nonuniform_n = sum(tildeg_nonuniform_mat,dims=2)
tildeg_nonuniform_z = sum(tildeg_nonuniform_mat,dims=1)'
plot(log.(ng),(tildeg_nonuniform_n))
plot(log.(zg),(tildeg_nonuniform_z))

moments = compute_moments(param,ss_result)

plot(moments.dlZg,moments.dlng)
moments.entry_rate
tilde_psig_nz = reshape(tilde_psig_nz,Jn,Jz)
tilde_psig_z = sum(tilde_psig_nz,dims=2)
plot(zg,tilde_psig_z')