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
    sig = 0.02
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
    xi = 1.1
    psig =  entry_dist(xi,zg)[1]
    tilde_psig = entry_dist(xi,zg)[2]
    ce = 10000
    eta = 0.0
    dt = 2
    tg = 0:dt:300
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
tildeg = ss_result.tildeg
sum(tildeg)


dw = 0.01;
v_path = zeros(J,length(tg))
dv_path = zeros(J,length(tg))
dunderz_path = zeros(length(tg))
for i_t = length(tg):-1:1
    if i_t == length(tg)
        w_in = w_ss + dw;
        v_in = copy(v_ss)
    else
        w_in = copy(w_ss);
        v_in = copy(v_path[:,i_t+1])
    end
    v,underz_index,underz,ng = solve_HJB_VI_transition(param,w_in,v_in)
    v_path[:,i_t] = v
    dv_path[:,i_t] = (v .- v_ss)/dw
    dunderz_path[i_t] = (underz .- underz_ss)/dw;
end


##############################################################################


param = model()
@unpack_model param


@unpack_model param
HJB_result = solve_w(param; calibration=1)
ng = HJB_result.ng
underz_index = HJB_result.underz_index
ce_calibrate = HJB_result.ce_calibrate
underz = HJB_result.underz
param.ce = ce_calibrate
tilde_hatg_za,A_store = solve_stationary_distribution(param,underz_index)
tilde_hatg_za_mat = reshape(tilde_hatg_za,J,Na)
tilde_hatg = sum(tilde_hatg_za_mat,dims=2)
plot(zg,tilde_hatg)
sum(tilde_hatg)

w_ss = HJB_result.w
v_ss = HJB_result.v
underz_ss = HJB_result.underz
dw = 0.01;
v_path = zeros(J,length(tg))
dv_path = zeros(J,length(tg))
dunderz_path = zeros(length(tg))
for i_t = length(tg):-1:1
    if i_t == length(tg)
        w_in = w_ss + dw;
        v_in = copy(v_ss)
    else
        w_in = copy(w_ss);
        v_in = copy(v_path[:,i_t+1])
    end
    v,underz_index,underz,ng = solve_HJB_VI_transition(param,w_in,v_in)
    v_path[:,i_t] = v
    dv_path[:,i_t] = (v .- v_ss)/dw
    dunderz_path[i_t] = (underz .- underz_ss)/dw;
end
A = populate_A_HJB(param)
ng = (alph./w_ss)^(1/(1-alph)).*zg
pig = zg.^(1-alph).*ng.^alph .- w_ss.*ng .- cf
r*dv_path[:,99]  - A*dv_path[:,99] - (dv_path[:,100] - dv_path[:,99])/dt


etag = range(0,0.05,length = 10)
entry_rate_g = zeros(length(etag))
exit_rate_g = zeros(length(etag))
ave_size_g = zeros(length(etag))
ave_age_g = zeros(length(etag))
ave_size_by_age_g = zeros(length(etag),Na)
exit_rate_by_age_g = zeros(length(etag),Na)
for (ieta,eta) in enumerate(etag)
    param.eta = eta
    entry_rate, ave_size, ave_age,ave_size_by_age,Exit_rate_by_age = wrapper_eqm(param)
    entry_rate_g[ieta] = entry_rate
    exit_rate_g[ieta] = entry_rate - eta;
    ave_size_g[ieta] = ave_size
    ave_age_g[ieta] = ave_age
    ave_size_by_age_g[ieta,:] = ave_size_by_age
    exit_rate_by_age_g[ieta,:] = Exit_rate_by_age
end



lw = 4
plt_entry = plt_fun(param,etag,entry_rate_g; lw = lw, ymin=0,ymax=0,xlabel="Population growth",title= "Entry rate")
ymin, ymax = ylims(plt_entry)
plt_exit = plt_fun(param,etag,exit_rate_g,lw=lw,ymin=ymin,ymax=ymax,xlabel="Population growth",title= "Exit rate")
plt_ave_size = plt_fun(param,etag,ave_size_g,lw=lw,xlabel="Population growth",title= "Average firm size")
plt_ave_age = plt_fun(param,etag,ave_age_g,lw=lw,xlabel="Population growth",title= "Average firm age")

plt_all = plot(plt_entry,plt_exit,plt_ave_size,plt_ave_age,layout=(2,2),size=(1200,800))
plot!(margin=5mm)


if fig_save == 1
    savefig(plt_all,"./figure/Karahan_Sahin_Pugsley.pdf")
end


plt_size_by_age = plt_by_age_fun(param,etag,ave_size_by_age_g; lw = lw, ymin=0,ymax=0,xlabel="Population growth",title= "Average firm size conditional on age")
plt_exit_by_age = plt_by_age_fun(param,etag,exit_rate_by_age_g,lw=lw,ymin=0,ymax=0,xlabel="Population growth",title= "Exit rate conditional on age")


plt_all = plot(plt_size_by_age,plt_exit_by_age,layout=(1,2),size=(1200,400))
plot!(margin=5mm)

if fig_save == 1
    savefig(plt_all,"./figure/By_age_Karahan_Sahin_Pugsley.pdf")
end




##############################################################################


param = model()
@unpack_model param
etapath = max.(0.02 .- 0.02/40 .*tg,0)
param.eta = etapath[1];
@unpack_model param
HJB_result = solve_w(param; calibration=1)
ng = HJB_result.ng
underz_index = HJB_result.underz_index
ce_calibrate = HJB_result.ce_calibrate
param.ce = ce_calibrate
tilde_hatg_za,A_store = solve_stationary_distribution(param,underz_index)
g0 = tilde_hatg_za
tilde_hatgpath = solve_transition_distribution(param,underz_index,g0,etapath)

mpath = zeros(T)
entry_rate_path = zeros(T)
for t = 1:T
    tildeg_mat_temp = reshape(tilde_hatgpath[:,t],J,Na)
    tildeg_temp = sum(tildeg_mat_temp,dims=2)
    mpath[t] = L/sum(tildeg_temp.*ng)
    g_temp = tildeg_temp.*mpath[t]
    entry_rate_path[t] = sum(mpath[t].*tilde_psig[underz_index:end])/sum(g_temp)
end


plt_eta = plot(tg,etapath,lw=4,label=:none)
xlabel = "Year"
title = "Population growth rate"
plot!(xlabel=xlabel,title= title)
plot!(titlefontfamily = "Computer Modern",
xguidefontfamily = "Computer Modern",
yguidefontfamily = "Computer Modern",
legendfontfamily = "Computer Modern",
titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)


plt_entry = plot(tg,entry_rate_path,lw=4,label=:none)
xlabel = "Year"
title = "Entry rate"
plot!(xlabel=xlabel,title= title)
plot!(titlefontfamily = "Computer Modern",
xguidefontfamily = "Computer Modern",
yguidefontfamily = "Computer Modern",
legendfontfamily = "Computer Modern",
titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
plt_all = plot(plt_eta,plt_entry,layout=(1,2),size=(1200,400))
plot!(margin=5mm)


#################################################################################


HJB_result = solve_w(param; calibration=1)
v = HJB_result.v
w = HJB_result.w
ng = HJB_result.ng
underz_index = HJB_result.underz_index
ce_calibrate = HJB_result.ce_calibrate
param.ce = ce_calibrate
tilde_hatg_za,A_store = solve_stationary_distribution(param,underz_index)
tilde_hatg = sum(reshape(tilde_hatg_za,J,Na),dims=2)
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
