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
    sig = 0.3
    zeta = 1.05
    mu = sig^2*(1-zeta)/2
    #zeta = 1 - 2*mu/sig^2
    #mu = -0.000001
    lzg = range(log(0.1),10,length=J)
    #zg = range(0.00001,10,length=J)
    zg = exp.(lzg)
    ag = range(0,50,length=Na)
    da = diff(ag)
    dz = diff(zg)
    alph = 0.64
    cf = 0.1
    r = 0.05
    L = 1
    underv = 0.0
    xi = 1.5
    psig = entry_dist(xi,zg,Na)[1]
    tilde_psig = entry_dist(xi,zg,Na)[2]
    tilde_dn = entry_dist(xi,zg,Na)[3]
    tilde_psig_na = entry_dist(xi,zg,Na)[4]
    ce = 1
    eta = 0.02
    dt = 2
    tg = 0:dt:300
    T = length(tg)
end
function entry_dist(xi,zg,Na)
    d = Pareto(xi, 0.1)
    dz = diff(zg)
    psig = pdf(d,zg)
    tilde_psig = copy(psig)
    J = length(zg)
    tilde_dn = copy(psig)
    for j = 1:J
        if j == 1
            tilde_dn[j] = 1/2*dz[j]
            tilde_psig[j] = psig[j]*tilde_dn[j]
        elseif j == J
            tilde_dn[j] =1/2*dz[j-1]
            tilde_psig[j] = psig[j]*tilde_dn[j] 
        else
            tilde_dn[j] = 1/2*(dz[j-1] + dz[j])
            tilde_psig[j] = psig[j]*tilde_dn[j]
        end
    end
    tilde_psig = tilde_psig/sum(tilde_psig)
    tilde_psig_na = zeros(J*Na,1)
    tilde_psig_na[1:J] = tilde_psig;
    return psig,tilde_psig,tilde_dn,tilde_psig_na
end
function compute_za_index(param,iz,ia)
    @unpack_model param
    return (ia .-1).*J .+ iz
end
function populate_A_HJB(param)
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
function populate_A_KFE(param)
    @unpack_model param
    A_row = ones(Int64,J*Na*10)
    A_col = ones(Int64,J*Na*10)
    A_val = zeros(Float64,J*Na*10)
    A = spzeros(length(zg),length(zg))
    
    k =0 
    for (iz,z) in enumerate(zg)
        for (ia,a) in enumerate(ag)
            dz_plus = dz[min(iz,J-1)]
            dz_minus = dz[max(iz-1,1)]
            
            if mu > 0
                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,min(iz+1,J),ia)
                A_val[k] = mu.*z/dz_plus;

                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,iz,ia)
                A_val[k] = -mu.*z/dz_plus;
            else
                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,iz,ia)
                A_val[k] = mu.*z/dz_plus;

                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,max(iz-1,1),ia)
                A_val[k] = -mu.*z/dz_plus;
            end
            
            denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,ia)
            A_val[k] = - 1/2*(dz_plus + dz_minus).*(sig*z)^2/denom;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,max(iz-1,1),ia)
            A_val[k] = 1/2*dz_plus.*(sig*z)^2/denom;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,min(iz+1,J),ia)
            A_val[k] = 1/2*dz_minus.*(sig*z)^2/denom;
            
            da_plus = da[min(ia,Na-1)]
            da_minus = da[max(ia-1,1)]

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,min(ia+1,Na))
            A_val[k] = 1.0/da_plus;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,ia)
            A_val[k] = -1.0/da_minus;
        end
    end 

    A = sparse(A_row,A_col,A_val,J*Na,J*Na)
    return A
end

function solve_HJB_VI(param,w)
    @unpack_model param
    A = populate_A_HJB(param)
    B = (r.*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    q = -pig + underv.*B*ones(length(zg))
    result = solve!(LCP(B,q),max_iter=1000)
    println(result.converged)
    x = result.sol
    v = x .+ underv
    first_positive = findfirst(x .> 0 )
    if isnothing(first_positive)
        underz_index = J
    else
        underz_index = findfirst(x .> 0 )
    end
    return v,underz_index,ng
end

function solve_w(param; calibration=0 )
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = 0
    v = 0
    ng = 0
    ce_calibrate = 0
    while iter < 1000 && abs(err_free_entry) > 1e-6
        if calibration != 0
            w = 1;
        else
            w = (w_ub + w_lb)/2
        end
        v,underz_index,ng = solve_HJB_VI(param,w)
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
    @assert iter < 1000
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,ce_calibrate = ce_calibrate)
end

function solve_stationary_distribution(param,underz_index)
    @unpack_model param
    A = populate_A_KFE(param)
    A_store = copy(A)
    A = spdiagm(-eta*ones(J*Na)) + A
    B = -(tilde_psig_na);  
    for ia = 1:Na
        underz_set = compute_za_index(param,1:(underz_index-1),ia)
        A[underz_set,:] .= 0;
        A[:,underz_set] .= 0;
        A[underz_set,underz_set] = I(length(underz_set));
        B[underz_set] .= 0;
    end
    g = (A')\B;
    return g,A_store
end

function solve_transition_distribution(param,underz_index,g0,etapath)
    @unpack_model param
    A_store = populate_A_KFE(param)
    tilde_hatgpath = zeros(J*Na,T)
    tilde_hatgpath[:,1] = g0;
    for t = 2:T
        A = spdiagm(-etapath[t]*ones(J*Na)) + A_store
        B = (tilde_psig_na);  
        for ia = 1:Na
            underz_set = compute_za_index(param,1:(underz_index-1),ia)
            A[underz_set,:] .= 0;
            A[:,underz_set] .= 0;
            #A[underz_set,underz_set] = I(length(underz_set));
            B[underz_set] .= 0;
        end
        tilde_hatgpath[:,t] = (I - dt*A')\(tilde_hatgpath[:,t-1] + dt*B);
    end

    return tilde_hatgpath
end

function wrapper_eqm(param)
    @unpack_model param
    HJB_result = solve_w(param; calibration=1)
    ng = HJB_result.ng
    underz_index = HJB_result.underz_index
    ce_calibrate = HJB_result.ce_calibrate
    param.ce = ce_calibrate
    tilde_hatg_za,A_store = solve_stationary_distribution(param,underz_index)
    tilde_hatg_za_mat = reshape(tilde_hatg_za,J,Na)
    tilde_hatg = sum(tilde_hatg_za_mat,dims=2)
    tilde_hatg_age = sum(tilde_hatg_za_mat,dims=1)'
    m = L/sum(tilde_hatg.*ng)
    tilde_g = tilde_hatg.*m
    entry_rate = sum(m.*tilde_psig[underz_index:end])/sum(tilde_g)
    Average_size = sum(tilde_hatg.*ng)/sum(tilde_hatg)
    Average_age = sum(tilde_hatg_age.*ag)/sum(tilde_hatg_age)
    Average_size_by_age = zeros(Na)
    Exit_rate_by_age = zeros(Na)
    for ia = 1:Na
        Average_size_by_age[ia] = sum(tilde_hatg_za_mat[:,ia].*ng)/sum(tilde_hatg_za_mat[:,ia])
        if ia < Na
            Exit_rate_by_age[ia] = (tilde_hatg_age[ia] - tilde_hatg_age[ia+1]*(1+eta))/tilde_hatg_age[ia]
        end
    end

    return entry_rate, Average_size,Average_age,Average_size_by_age,Exit_rate_by_age
end


function plt_fun(param,x,y; lw = 4, ymin=0,ymax=0, xlabel = "",title = "")
    @unpack_model param
    plt = plot(x,y,label=:none,lw=lw)
    plot!(xlabel=xlabel,title= title)
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    if ymin != 0 && ymax != 0
        plot!(ylim=(ymin,ymax))
    end
    display(plt)
    return plt
end


function plt_by_age_fun(param,x,y; lw = 4, ymin=0,ymax=0, xlabel = "",title = "")
    @unpack_model param
    plt_by_age = plot()
    for ia in [25 20 15 10 5 1]
        age = Int(round(ag[ia]))
        plot!(x,y[:,ia],label="age = $ia",lw=3)
    end
    plot!()
    plot!(xlabel=xlabel,title= title)
    plot!(titlefontfamily = "Computer Modern",
    xguidefontfamily = "Computer Modern",
    yguidefontfamily = "Computer Modern",
    legendfontfamily = "Computer Modern",
    titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
    if ymin != 0 && ymax != 0
        plot!(ylim=(ymin,ymax))
    end
    display(plt_by_age)
    return plt_by_age
end




##############################################################################


param = model()
@unpack_model param


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
