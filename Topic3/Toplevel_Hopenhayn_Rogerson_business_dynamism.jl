using SparseArrays 
using Parameters
using LinearAlgebra
using LCPsolve
using Distributions
using Plots
using Plots.Measures

fig_save = 1;

@with_kw mutable struct model
    J = 1000
    Na = 50
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
    ce = 0.5
    tilde_Delta_z = [ dz[1]; [(dz[i]+dz[i+1])/2 for i in 1:(J-2)]; dz[end]]
    psig = entry_dist(xi,zg,tilde_Delta_z)[1]
    tilde_psig = entry_dist(xi,zg,tilde_Delta_z)[2]
    psig_na = [psig; zeros((Na-1)*J)]
    tilde_psig_na = [tilde_psig; zeros((Na-1)*J)]
    max_iter = 1e3
    #lag = range(0,log(100),length=Na)
    ag = range(0,100,length=Na)
    da = diff(ag)
    tilde_Delta_a = [da[1]; [(da[i]+da[i+1])/2 for i in 1:(Na-2)]; da[end]]
    eta = 0.00
    dt = 2
    tg = 0:dt:300
    T = length(tg)
end


include("plot_functions.jl")
include("steady_state_functions.jl")
include("subfunctions.jl")
include("transition_functions.jl")




##############################################################################
# Comparative statics across steady state
##############################################################################

param = model()
@unpack_model param
HJB_result = solve_w(param; calibration=1)
ce_calibrate = HJB_result.ce_calibrate
param = model(ce = ce_calibrate)

etag = range(0,0.05,length = 10)
ss_comparative_statics(param,"eta",etag)

ce_g = range(0.5,2.0,length = 10)
ss_comparative_statics(param,"ce",ce_g)

cf_g = range(0.5,2.0,length = 10)
ss_comparative_statics(param,"cf",cf_g)

r_g = range(0.02,0.1,length = 10)
ss_comparative_statics(param,"r",r_g)


##############################################################################
# Transition Dynamics
##############################################################################


@unpack_model param
etapath = max.(0.02 .- 0.02/40 .*tg,0)
param.eta = etapath[1];
@assert param.ce == ce_calibrate
transition_result = solve_transition_distribution(param,etapath)

@unpack entry_rate_path, exit_rate_path, mpath, Average_age_path, Average_size_path = transition_result

plt_eta = plt_fun(param,tg,etapath,lw=lw,xlabel = "Year",title = "Population growth rate")
plt_entry = plt_fun(param,tg,entry_rate_path,lw=lw,xlabel = "Year",title = "Entry & exit rates", y2 = exit_rate_path,label1="Entry rate",label2="Exit rate")
plt_size = plt_fun(param,tg,Average_size_path,lw=lw,xlabel = "Year",title = "Average Firm Size")
plt_age = plt_fun(param,tg,Average_age_path,lw=lw,xlabel = "Year",title = "Average Firm Age")

plt_all = plot(plt_eta,plt_entry,
plt_size,plt_age,layout=(2,2),size=(1200,800))
plot!(margin=5mm)

if fig_save == 1
    savefig(plt_all,"./figure/Karahan_Sahin_Pugsley_transition_dynamics.pdf")
end
