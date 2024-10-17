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

shock = "eta"
param = model(nu=2)
IRF_path,Jacobian_dict = SSJ_wrapper(param,shock=shock)
param_high_nu = model(nu=10)
IRF_path_high_nu,Jacobian_dict_high_nu = SSJ_wrapper(param_high_nu,shock=shock)


plt = Dict{String,Any}()
plt["eta"] = plot_IRF(param,IRF_path["etapath"]*100,IRF_path["etapath"][1]*100; tpre=10,tmax=0,title = "Population growth rate",ylabel="p.p.")
plt["w_plot"] = plot_IRF(param,IRF_path["w_plot"]*100,0; tpre=10,tmax=0,title = "Wage",label1="ν = "*string(param.nu),y2 = IRF_path_high_nu["w_plot"]*100,label2="ν = "*string(param_high_nu.nu))

plt["Firm_mass"] = plot_IRF(param,IRF_path["Firm_mass_plot"]*100,0; tpre=10,tmax=0,title = "Mass of firms (per capita)",label1="ν = "*string(param.nu),y2 = IRF_path_high_nu["Firm_mass_plot"]*100,label2="ν = "*string(param_high_nu.nu))

plt["Entry_rate"] = plot_IRF(param,IRF_path["Entry_rate_plot"]*100,0; tpre=10,tmax=0,title = "Entry rate",label1="ν = "*string(param.nu),y2 = IRF_path_high_nu["Entry_rate_plot"]*100,label2="ν = "*string(param_high_nu.nu),ylabel="p.p. devition from initial s.s.")

plt["Exit_rate"] = plot_IRF(param,IRF_path["Exit_rate_plot"]*100,0; tpre=10,tmax=0,title = "Exit rate",label1="ν = "*string(param.nu),y2 = IRF_path_high_nu["Exit_rate_plot"]*100,label2="ν = "*string(param_high_nu.nu),ylabel="p.p. devition from initial s.s.")

if shock == "eta"
    plt_all = plot(plt["eta"],plt["Entry_rate"],plt["w_plot"],plt["Firm_mass"],layout=(2,2),size=(1200,800))
elseif shock == "exit"
    plt_all = plot(plt["Exit_rate"],plt["Entry_rate"],plt["w_plot"],plt["Firm_mass"],layout=(2,2),size=(1200,800))
end
plot!(margin=5mm)
if fig_save == 1
    savefig(plt_all,"./figure/SSJ_Karahan_shock_"*shock*".pdf")
end

default_colors = palette(:auto)
shock_index = [1,21,41]
shock_label = ["s = "*string((shock_index[i]-1)*dt) for i in eachindex(shock_index)]

plot(tg, Jacobian_dict["w"]["N"][:,shock_index],lw=2,label=permutedims(shock_label),color=default_colors[1:length(shock_index)]')
plot!(tg, Jacobian_dict_high_nu["w"]["N"][:,shock_index],lw=2,label=:none,linestyle=:dash,color=default_colors[1:length(shock_index)]')
xlims!(0,100)
xlabel!("t")
plot!(titlefontfamily = "Computer Modern",
xguidefontfamily = "Computer Modern",
yguidefontfamily = "Computer Modern",
legendfontfamily = "Computer Modern",
titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
if fig_save == 1
    savefig("./figure/Jacobian_N_w_shock_"*shock*".pdf")
end