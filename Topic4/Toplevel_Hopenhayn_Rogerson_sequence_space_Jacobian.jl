using SparseArrays 
using Parameters
using LinearAlgebra
using LCPsolve
using Distributions
using Plots
using Plots.Measures

fig_save = 0;

@with_kw mutable struct model
    J = 1000
    sig = 0.41
    zeta = 1.05
    mu = sig^2*(1-zeta)/2
    lzg = range(log(0.1),log(2300000),length=J)
    zg = exp.(lzg)
    dz = diff(zg)
    alph = 0.64
    cf = 0.5
    r = 0.05
    L = 1
    underv = 0.0
    xi = 1.1
    ce = 15
    tilde_Delta_z = [ dz[1]; [(dz[i]+dz[i+1])/2 for i in 1:(J-2)]; dz[end]]
    psig = entry_dist(xi,zg,tilde_Delta_z)[1]
    tilde_psig = entry_dist(xi,zg,tilde_Delta_z)[2]
    max_iter = 1e3
    #lag = range(0,log(100),length=Na)
    eta = 0.00
    dt = 2
    tg = 0:dt:300
    T = length(tg)
    nu = 2
end


include("plot_functions.jl")
include("steady_state_functions.jl")
include("subfunctions.jl")
include("transition_functions.jl")


param = model()
@unpack_model param
ss_result = solve_w(param; calibration=0 )
compute_calibration_targets(param,ss_result)



N_Jacobian_w = Compute_Sequence_Space_Jacobian(param,ss_result,"w")

N_Jacobian_eta = Compute_Sequence_Space_Jacobian(param,ss_result,"eta")

N_Jacobian_Z = Compute_Sequence_Space_Jacobian(param,ss_result,"Z")


plot(tg,N_Jacobian_w[:,1])

plot(tg,N_Jacobian_eta[:,5])

etapath = zeros(length(tg))
etapath[1:1] .= 1.0;

rhoZ = 0.9
Zpath = rhoZ.^(tg.-1)
dwpath = - N_Jacobian_w\N_Jacobian_eta*etapath;
dwpath = - N_Jacobian_w\N_Jacobian_Z*Zpath;

plot(tg,etapath)
plot(tg,dwpath)