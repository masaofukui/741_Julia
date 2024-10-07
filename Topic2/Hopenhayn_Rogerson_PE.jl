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
    underv = 0
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    max_iter = 1e3
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


function Howard_Algorithm(param,B)
    @unpack_model param
    iter = 1;
    vold = zeros(length(zg));
    vnew = copy(vold);
    while iter < max_iter
        val_noexit =  (B*vold .- pig);
        val_exit = vold  .- underv
        exit_or_not =val_noexit  .> val_exit;
        Btilde = B.*(1 .-exit_or_not) + I(J) .*(exit_or_not)
        q = pig.*(1 .-exit_or_not) + underv.*(exit_or_not)
        vnew = Btilde\q;
        if norm(vnew - vold) < 1e-6
            break
        end
        vold = copy(vnew)
        iter += 1
    end
    @assert iter < max_iter "Howard Algorithm did not converge"
    return vnew,exit_or_not
end

function solve_HJB_VI(param)
    @unpack_model param
    A = populate_A(param)
    B = (r.*I - A);
    v,exist_or_not = Howard_Algorithm(param,B)
    underz_index = findlast(exist_or_not .>0 )
    if isnothing(underz_index)
        underz_index = J
    end
    underz = zg[underz_index]
    return v,underz
end
v_exit,underz = solve_HJB_VI(param)

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
