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
    nu = Inf
    phi = 10
    s = 0.00
    kappa = 2
    tau_f = 0.0
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
ss_result = solve_w(param,calibration=1,tol=0.01)
HJB_result = ss_result.HJB_result
@unpack exit_or_not =HJB_result
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

################################################################
# Plot firing regulation
################################################################
no_phi_zero = 1
plot(moments.dlZg.*(1-alph),moments.dlng,lw=6,label="\\phi=10")
if no_phi_zero == 0
    plot!(moments.dlZg.*(1-alph),moments.dlZg,lw=6,label="\\phi=0",linestyle=:dash)
    ylims!(-0.1,0.1)
    plot!(legend=:topleft)
else
    plot!(legend=:none)
end
plot!(xlabel="\\Delta log Z",ylabel="\\Delta log n")
plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
if fig_save == 1
    savefig("./figure/slow_to_hire_quick_to_fire_no_phi_"*string(no_phi_zero)*".pdf")
end

using CSV
using DataFrames
using Plots
df = CSV.read("./Topic5/data/OECD.ELS.JAI,DSD_EPL@DF_EPL,+all.csv", DataFrame)
df = filter(row -> row.TIME_PERIOD == 2019,df)
df = filter(row -> row.VERSION == "VERSION4",df)
df = filter(row -> row.MEASURE == "EPL_OV",df)
df = sort(df,:OBS_VALUE,rev=false)
colors = [country == "USA" ? :red : :lightblue for country in  df[!,:REF_AREA]]


bar(df[!,:OBS_VALUE], xlabel="", ylabel="", title="Employment Protection Index",xticks=(1:nrow(df), df[!,:REF_AREA]),xrotation=45,
xtickfont=font(6, "Computer Modern"),
seriescolor=colors,
size=(600,350)
)
bar!(legend=:none)
bar!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=15,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
if fig_save ==1
    savefig("./figure/Employment_Protection_Index.pdf")
end


##########################################################
# Firing cost
##########################################################
param = model()
@unpack_model param
ss_result = solve_w(param,calibration=1,tol=0.01)
ce_calibrate = ss_result.ce_calibrate
tau_f_grid = range(0.0,0.6,step=0.2)
firing_outcome = Dict{String,Any}()
firing_outcome["w"] = zeros(length(tau_f_grid))
firing_outcome["ave_size"] = copy(wgrid)
firing_outcome["entry_rate"] = copy(wgrid)
firing_outcome["LP"] = copy(wgrid)

for (itau,tau) in enumerate(tau_f_grid)
    param = model(ce=ce_calibrate,tau_f=tau)
    ss_result = solve_w(param,calibration=0,tol=0.01)
    firing_outcome["w"][itau] = ss_result.w
    firing_outcome["ave_size"][itau] = ss_result.ave_size
    firing_outcome["entry_rate"][itau] = ss_result.entry_rate
    firing_outcome["LP"][itau] = ss_result.TFP    
end

title_label = Dict("w"=>"Wage","ave_size"=>"Average Size","entry_rate"=>"Entry Rate","LP"=>"Labor Productivity")
plt = Dict{String,Any}()
for y in ["w","LP"]
    plt[y]=plot(tau_f_grid,firing_outcome[y],lw=6,title=title_label[y],legend=:none)
    xlabel!("Firing cost, \\tau")
    plot!(grid=:y)
    plot!(titlefontfamily = "Computer Modern",
        xguidefontfamily = "Computer Modern",
        yguidefontfamily = "Computer Modern",
        legendfontfamily = "Computer Modern",
        titlefontsize=15,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
end
plot(plt["w"],plt["LP"],layout=(1,2),size=(1200,400))
plot!(margin=6mm)
if fig_save == 1
    savefig("./figure/Firing_cost.pdf")
end