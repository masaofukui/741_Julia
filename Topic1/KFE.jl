using SparseArrays 
using Parameters
@with_kw mutable struct model
    J = 1000
    sig = 0.1
    mu = -0.01
    ng = range(1.0,6,length=J)
    dn = ng[2] - ng[1]
    zeta = 1 - 2*mu/sig^2
end
function populate_A(param)
    @unpack_model param
    A = spzeros(length(ng),length(ng))
    for (i,n) in enumerate(ng)
        A[i,i] += -(sig*n)^2/dn^2;
        A[i,min(i+1,J)] += 1/2*(sig*n)^2/dn^2;
        A[i,max(i-1,1)] += 1/2*(sig*n)^2/dn^2;
        if mu > 0
            A[i,i] += -mu*n/dn;
            A[i,min(i+1,J)] += mu*n/dn;
        else
            A[i,i] += mu*n/dn;
            A[i,max(i-1,1)] += -mu*n/dn; 
        end
    end 
    return A
end
function solve_stationary_distribution(param)
    @unpack_model param
    A = populate_A(param)
    B = zeros(length(ng));  
    B[end] = 1;
    A[end,:] = ones(1,length(ng))*dn;
    g = A'\B;
    return g
end
param = model()
g = solve_stationary_distribution(param)


# Plotting 
using Plots
@unpack_model param
colplot_blue = palette(:Blues_4);
colplot_red = palette(:Reds_3);

g = g./sum(g.*dn)
analytical_n = zeta.*ng.^(-zeta-1)
analytical_n = analytical_n./sum(analytical_n.*dn)
plt_dist = plot(ng,g,label="Numerical",lw=6,color=colplot_blue[2])
plot!(ng,analytical_n,lw=6,linestyle = :dash,color=colplot_blue[4],label="Analytical")
plot!(xlabel="n",title= "Density of Firm Size Distribution, g(n)")
plot!(titlefontfamily = "Computer Modern",
            xguidefontfamily = "Computer Modern",
            yguidefontfamily = "Computer Modern",
            titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
display(plt_dist)
savefig(plt_dist, "./figure/density_firm_size_distribution.pdf")

A = populate_A(param)
plt_A = plot(Plots.spy(A))
plot!(cb=:none)
savefig(plt_A,"./figure/spyA.pdf")


G = cumsum(g.*dn);
G_analytical = 1 .- ng.^(-zeta);
ranking = 1 .- G;
ranking_analytical = 1 .- G_analytical;
plt_range = 1:(J-20)
plt_PL = plot(log.(ng[plt_range]), log.(ranking[plt_range]), lw=6, color=colplot_blue[3],label = "Numerical")
plot!(log.(ng[plt_range]), log.(ranking_analytical[plt_range]), lw=6, linestyle=:dash, color=colplot_blue[2],label = "Analytical");
plot!(xlabel="log(n)",ylabel= "log Ranking, log(1-G(n))",title= "Power Law")
plot!(titlefontfamily = "Computer Modern",
            xguidefontfamily = "Computer Modern",
            yguidefontfamily = "Computer Modern",
            titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
savefig(plt_PL,"./figure/Power_Law.pdf")




####################################################
# Transition Dynamics
####################################################
using LinearAlgebra
dt = 0.1;
T = 5000;
A = populate_A(param);
gpath = zeros(J,T);
gpath[:,1] = ones(J)./(J*dn);
for t = 2:T
    gpath[:,t] = (I - dt*A')\gpath[:,t-1]
end

tgrid = [100,500,T]
plt_transition = plot(ng,gpath[:,1],label="t = "*string(0),lw=6,color=colplot_blue[1])
plot!(xlabel="Firm size, n",ylabel= "g(n)",title= "Transition of Firm Size Distribution, g(n)")
plot!(titlefontfamily = "Computer Modern",
            xguidefontfamily = "Computer Modern",
            yguidefontfamily = "Computer Modern",
            titlefontsize=15,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
ylims!(-0.1,3.1)
savefig(plt_transition,"./figure/Transition_firm_size_0.pdf")
for i in eachindex(tgrid)
    plot!(ng,gpath[:,tgrid[i]],label="t = "*string(tgrid[i]*dt),lw=6,color=colplot_blue[i+1])
    savefig(plt_transition,"./figure/Transition_firm_size_"*string(i)*".pdf")
end
plot!(ng,g[:,end],label="Steady state",lw=6,color=colplot_red[2],linestyle=:dash)

savefig(plt_transition,"./figure/Transition_firm_size.pdf")
