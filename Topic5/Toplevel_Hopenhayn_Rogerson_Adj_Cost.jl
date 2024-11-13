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
    sig = 0.41
    zeta = 1.05
    mu = sig^2*(1-zeta)/2
    lng = range(log(0.01),log(2300000),length=Jn)
    lzg = range(log(0.01),log(2300000),length=Jz)
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
    ce = 0.5
    tilde_Delta_z = [ Delta_z[1]; [(Delta_z[i]+Delta_z[i+1])/2 for i in 1:(Jz-2)]; Delta_z[end]]
    n_start = 5
    tilde_psig_nz = entry_dist(xi,ng,zg,tilde_Delta_z,n_start);
    max_iter = 1e3
    dt = 2
    tg = 0:dt:300
    T = length(tg)
    nu = 5
    phi = 10
    s = 0.00
    kappa = 2;
    g_fun = (h,n) -> phi/kappa .* (h./n).^kappa.*n
    h_fun = (dv,n) -> (max.(dv,0) ./phi).^(1/(kappa-1)).*n 
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

tildeg_nonuniform_n = tildeg_nonuniform_n[:]
G = reverse(cumsum(reverse(tildeg_nonuniform_n)))/sum(tildeg_nonuniform_n)
plt_power_law= plot(log.(ng),log.(G),label="Model",lw=3)
  

moments = compute_moments(param,ss_result)

plot(moments.dlZg,moments.dlng)
moments.entry_rate
tilde_psig_nz_mat = reshape(tilde_psig_nz,Jn,Jz)
tilde_psig_z = sum(tilde_psig_nz_mat,dims=1)
plot(log.(zg),tilde_psig_z')