using SparseArrays 
using Parameters
using LinearAlgebra
using LCPsolve
using Distributions
using Plots
using Plots.Measures
using Interpolations
using Roots
fig_save = 1;


@with_kw mutable struct model
    J = 1000
    sig = 0.2
    zeta = 1.05
    mu = sig^2*(1-zeta)/2
    lzg = range(log(0.01),10,length=J)
    zg = exp.(lzg)
    dz = diff(zg)
    alph = 0.64
    cf = 4
    r = 0.05
    L = 1
    M = 1
    underv = 0.0
    xi = 1.2
    entry_zlow = 1;
    psig =  entry_dist(xi,zg,entry_zlow)[1]
    tilde_psig = entry_dist(xi,zg,entry_zlow)[2]
    ce = 10000
    eta = 0.0
    dt = 10
    tg = 0:dt:2000
    T = length(tg)
    nu = 1;
end

include("subfunctions.jl")
include("functions_steady_state.jl")
include("functions_transitions.jl")
include("plot_functions.jl")


param = model()
@unpack_model param
ss_result = solve_w(param; calibration=0 )


w_ss = ss_result.w
v_ss = ss_result.v
underz_ss = ss_result.underz
m = ss_result.m
g_ss = ss_result.tildeg
ng_ss = ss_result.ng
A_ss,B_ss = populate_A_KFE(param,underz_ss,eta)
sum(g_ss)

N_Jacobian_w = Compute_Sequence_Space_Jacobian(param,ss_result,"w")

N_Jacobian_eta = Compute_Sequence_Space_Jacobian(param,ss_result,"eta")


plot(tg,N_Jacobian_w[:,1])

plot(tg,N_Jacobian_eta[:,1])

etapath = zeros(length(tg))
etapath[1:1] .= 1.0;
dwpath = - N_Jacobian_w\N_Jacobian_eta*etapath;

plot(tg,etapath)
plot(tg,dwpath)