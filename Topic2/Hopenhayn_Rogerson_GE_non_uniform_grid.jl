using SparseArrays 
using Parameters
using LinearAlgebra
using Distributions
using Plots
using Kezdi
using ReadStatTables
using StatsPlots
using Plots.Measures

fig_save = 1;
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
    ce = 0.5
    tilde_Delta_z = [ dz[1]; [(dz[i]+dz[i+1])/2 for i in 1:(J-2)]; dz[end]]
    psig_na = entry_dist(xi,zg,tilde_Delta_z)[1]
    tilde_psig = entry_dist(xi,zg,tilde_Delta_z)[2]
    max_iter = 1e3
end
function entry_dist(xi,zg,tilde_Delta_z)
    d = Pareto(xi, 0.1)
    psig = pdf(d,zg)
    tilde_psig = copy(psig)
    J = length(zg)
    for j = 1:J
        tilde_psig[j] = psig[j]*tilde_Delta_z[j]
    end
    tilde_psig = tilde_psig/sum(tilde_psig)
    return psig,tilde_psig
end
function populate_A(param)
    @unpack_model param
    A = spzeros(length(zg),length(zg))
    
    for (i,z) in enumerate(zg)
        dz_plus = dz[min(i,J-1)]
        dz_minus = dz[max(i-1,1)]
        if mu > 0
            A[i,min(i+1,J)] += mu.*z/dz_plus;
            A[i,i] += -mu.*z/dz_plus;
        else
            A[i,i] += mu.*z/dz_minus;
            A[i,max(i-1,1)] += -mu.*z/dz_minus;
        end
        denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;
        A[i,i] += - 1/2*(dz_plus + dz_minus).*(sig*z)^2/denom;
        A[i,max(i-1,1)] += 1/2*dz_plus.*(sig*z)^2/denom;
        A[i,min(i+1,J)] += 1/2*dz_minus.*(sig*z)^2/denom;
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

function solve_w(param; calibration=0 )
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = [];
    v = [];
    ng = [];
    ce_calibrate = [];
    exit_or_not = [];
    while iter < max_iter && abs(err_free_entry) > 1e-6
        if calibration != 0
            w = 0.78;
        else
            w = (w_ub + w_lb)/2
        end
        v,underz_index,ng,exit_or_not = solve_HJB_VI(param,w)
        err_free_entry = sum(v.*tilde_psig) - ce
        if err_free_entry > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," err_free_entry: ",err_free_entry)
        iter += 1
        if calibration != 0
            ce_calibrate = sum(v.*tilde_psig)
            break
        end
    end
    @assert iter < max_iter
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,ce_calibrate = ce_calibrate,
    exit_or_not = exit_or_not)
end

function solve_stationary_distribution(param,HJB_result)
    @unpack_model param
    @unpack exit_or_not,ng = HJB_result
    D = spdiagm(0 => exit_or_not)
    I_D = I-D;
    A = populate_A(param)
    tildeA = A*I_D + D;
    B = -I_D*tilde_psig;  
    hatg_nonuniform = (tildeA')\B;
    m = L/sum(hatg_nonuniform.*ng)
    g_nonuniform = hatg_nonuniform.*m
    g = spdiagm(tilde_Delta_z)\g_nonuniform
    return (g=g,g_nonuniform=g_nonuniform,m=m)
end

function compute_calibration_targets(param,HJB_result,dist_result)
    @unpack_model param
    @unpack underz_index,ng = HJB_result
    @unpack m, g, g_nonuniform = dist_result
    
    underz_index = HJB_result.underz_index
    entry_rate = sum(m.*tilde_psig[underz_index:end])/sum(g_nonuniform)

    entrants_size = sum(ng[underz_index:end].*tilde_psig[underz_index:end])/sum(tilde_psig[underz_index:end])
    ave_size = sum(ng.*g_nonuniform)/sum(g_nonuniform)
    reverse_cumsum_g = reverse(cumsum(reverse(g_nonuniform)))/sum(g_nonuniform)
    emp500_cutoff = findlast(reverse_cumsum_g .> 0.0038)
    share_500 = sum(ng[emp500_cutoff:end].*g_nonuniform[emp500_cutoff:end])/sum(ng.*g_nonuniform)

    println("--------- Average Size -----------")
    println(ave_size)
    println("------ Entrants Size / Incumbents Size ------")
    println(entrants_size/ave_size)
    println("------ Entry Rate ------")
    println(entry_rate)
    println("------ Share of 500+ Firms ------")
    println(share_500)
    println("------ cutoff z ------")
    println(zg[underz_index])
    println("------ cutoff n ------")
    println(ng[underz_index])

end

function produce_bar_plots(param,HJB_result,dist_result,df)
    @unpack_model param
    @unpack underz_index,ng = HJB_result
    @unpack m, g, g_nonuniform = dist_result
    n_cutoff=  [4.5,9.5,19.5,99.5,500,1000,2500,5000,10000]
    cum_firm_share = zeros(length(n_cutoff)+1)
    cum_emp_share = copy(cum_firm_share)

    for (index_n,n_cut) in enumerate(n_cutoff)
        n_index = findlast(ng .< n_cut)
        cum_firm_share[index_n] = sum(g_nonuniform[1:n_index])/sum(g_nonuniform)
        cum_emp_share[index_n] = sum(g_nonuniform[1:n_index].*ng[1:n_index])/sum(g_nonuniform.*ng)
    end
    cum_firm_share[end] = 1
    cum_emp_share[end] = 1
    firm_share = [cum_firm_share[1]; diff(cum_firm_share)]
    emp_share = [cum_emp_share[1]; diff(cum_emp_share)]

    firm_share_data = df.num_firms/sum(df.num_firms)
    emp_share_data = df.num_emp/sum(df.num_emp)

    for firm_emp in ["firm","emp"]
        if firm_emp == "firm"
            y_bar = hcat(firm_share_data,firm_share)
            title = "Firm Share"
        else
            y_bar = hcat(emp_share_data,emp_share)
            title = "Employment Share"
        end
        x_lab = repeat(valuelabels(df.firm_size_cat), outer = 2);
        x_num = repeat(1:10,outer=2)
        ctg = repeat(["Data", "Model"], inner = length(firm_share_data))

        bar_plot = groupedbar(x_num, y_bar, group = ctg, xticks=(1:10,x_lab)
        , ylabel = "Share",
        title = title, 
        lw = 0, framestyle = :box,xrotation= 45)
        plot!(margin=5mm)
        plot!(titlefontfamily = "Computer Modern",
                xguidefontfamily = "Computer Modern",
                yguidefontfamily = "Computer Modern",
                legendfontfamily = "Computer Modern",
                titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
        if fig_save == 1
            savefig(bar_plot, "./figure/Hopenhayn_Rogerson_"*firm_emp*"_size.pdf")
        end
        display(bar_plot)

    end
       
   
end

function produce_density_plots(param,HJB_result,dist_result)
    @unpack_model param
    @unpack underz_index,ng = HJB_result
    @unpack m, g, g_nonuniform = dist_result

    density_plt = plot((ng),g,label="Non-weighted",lw=3, xaxis=:log)
    plot!((ng),g.*ng,label="Employment weighted",lw=3, xaxis=:log,linestyle=:dash)
    xticks!(10 .^(0:5))
    xlabel!("Employment (log scale)")
    ylabel!("Density")
    title!("Firm Size Distribution")
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    display(density_plt)
    if fig_save == 1
        savefig(density_plt, "./figure/Hopenhayn_Rogerson_density.pdf")
    end


end


function plot_power_law(param,HJB_result,dist_result,df_pareto)
    @unpack_model param
    @unpack underz_index,ng = HJB_result
    @unpack m, g, g_nonuniform = dist_result
    G = reverse(cumsum(reverse(g_nonuniform)))/sum(g_nonuniform)
    plt_power_law= plot(log.(ng),log.(G),label="Model",lw=3)
    scatter!(df_pareto.lfirm_size_cat,df_pareto.lcum_firm_share,label="Data",markersize=6)
    xlabel!("log firm Employment")
    ylabel!("log ranking")
    title!("log Ranking and log Firm Size")
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    display(plt_power_law)
    if fig_save == 1
        savefig(plt_power_law, "./figure/Hopenhayn_Rogerson_Power_Law.pdf")
    end
end

param = model()
@unpack_model param

calibration = 1;
HJB_result = solve_w(param; calibration=calibration)
if calibration == 1
    ce_calibrate = HJB_result.ce_calibrate
    param.ce = ce_calibrate
end
dist_result = solve_stationary_distribution(param,HJB_result)
compute_calibration_targets(param,HJB_result,dist_result)
df = @use "../Empirics/Data/temp/bds_temp.dta",clear;
setdf(df)
df = @keep @if year == 2021
produce_bar_plots(param,HJB_result,dist_result,df)
produce_density_plots(param,HJB_result,dist_result)

df_pareto = @use "../Empirics/Data/temp/data_for_julia_plot.dta",clear;
setdf(df_pareto)
df_pareto = @keep @if year == 2021

plot_power_law(param,HJB_result,dist_result,df_pareto)


