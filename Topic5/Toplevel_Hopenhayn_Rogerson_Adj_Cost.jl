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
    Jz = 120
    sig = 0.0
    zeta = 1.05
    mu = -0.01
    lng = range(log(0.1),log(2300000),length=Jn)
    lzg = range(log(0.1),log(2300000),length=Jz)
    ng = exp.(lng)
    zg = exp.(lzg)
    Delta_z = diff(zg)
    Delta_n = diff(ng)
    alph = 0.64
    cf = 0.5
    r = 0.05
    L = 1
    underv = 0.0
    xi = 1.1
    ce = 15
    tilde_Delta_z = [ Delta_z[1]; [(Delta_z[i]+Delta_z[i+1])/2 for i in 1:(Jz-2)]; Delta_z[end]]
    n_start = 1
    tilde_psig_zn = entry_dist(xi,zg,tilde_Delta_z,n_start)
    max_iter = 1e3
    eta = 0.00
    dt = 2
    tg = 0:dt:300
    T = length(tg)
    nu = 2
    phi = 2
    s = 0.08
    g_fun = h -> phi/2 .* h.^2
    h_fun = (dv) -> max.(dv,0) ./phi
end


include("plot_functions.jl")
include("steady_state_functions.jl")
include("subfunctions.jl")

w=1
v_QVI = solve_HJB_QVI(param,w)
vinit = -100000000*ones(Jn,Jz)
v_VI = solve_HJB_VI(param,w,vinit=vinit)


v_diff = maximum(abs.(v_QVI - v_VI))
findall(x -> x == v_diff, abs.(v_QVI - v_VI))

zindx = 59
plot(ng,v_QVI[:,zindx])
plot!(ng,v_VI[:,zindx])

plot!(ng,v_VI[:,100])