using SparseArrays 
using Parameters
using LinearAlgebra
@with_kw mutable struct model
    J = 500
    sig = 0.1
    mu = -0.01
    zg = range(0.001,6,length=J)
    dz = zg[2] - zg[1]
    alph = 0.66
    w = 1
    cf = 0.1
    r = 0.05
end
function populate_A(param)
    @unpack_model param
    A = spzeros(length(zg),length(zg))
    for (i,z) in enumerate(zg)
        if mu > 0
            A[i,min(i+1,J)] += mu.*z/dz;
            A[i,i] += -mu.*z/dz;
        else
            A[i,i] += mu.*z/dz;
            A[i,max(i-1,1)] += -mu.*z/dz;
        end
        A[i,i] += - (sig*z)^2/dz^2;
        A[i,max(i-1,1)] += 1/2*(sig*z)^2/dz^2;
        A[i,min(i+1,J)] += 1/2*(sig*z)^2/dz^2;
    end 
    return A
end

function solve_HJB(param)
    @unpack_model param
    A = populate_A(param)
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    v = (r.*I - A)\pig;
    return v
end
param = model()
v = solve_HJB(param)

@unpack_model param
using Plots
colplot_blue = palette(:Blues_4);
colplot_red = palette(:Reds_3);

plt_v = plot(zg,v,lw=6,label=:none)
plot!(xlabel="z",title= "Firm's value, v(z)")
plot!(titlefontfamily = "Computer Modern",
            xguidefontfamily = "Computer Modern",
            yguidefontfamily = "Computer Modern",
            titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
display(plt_v)
savefig(plt_v, "./figure/HJB_v_noexit.pdf")


using LCPsolve
function solve_HJB_VI(param,underv)
    @unpack_model param
    A = populate_A(param)
    B = r.*I - A;
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    q = -pig + underv.*B*ones(length(zg))
    result = solve!(LCP(B,q))
    @assert result.converged
    x = result.sol
    v = x .+ underv
    underz = zg[findfirst(x .> 0 )]
    return v,underz
end
underv=0.0
v_exit, underz = solve_HJB_VI(param,underv)

plt_v = plot(zg,v_exit,lw=6,label=:none)
vline!([underz], label="Exit threshold",color=colplot_red[2],lw=3,linestyle=:dash)
plot!(xlabel="z",title= "Firm's value, v(z)")
plot!(titlefontfamily = "Computer Modern",
            xguidefontfamily = "Computer Modern",
            yguidefontfamily = "Computer Modern",
            legendfontfamily = "Computer Modern",
            titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
display(plt_v)
savefig(plt_v, "./figure/HJB_v_exit.pdf")
