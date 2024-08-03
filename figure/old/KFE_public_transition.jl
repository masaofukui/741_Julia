using SparseArrays 
using Plots
using LinearAlgebra
Nn = 100;
sig = 0.1;
mu = -0.01;
ng = range(1.0,6,length=Nn)
dn = ng[2] - ng[1]
zeta = 1 - 2*mu/sig^2
dt = 0.001;
T = 10000;
function populate_A(ng,mu,sig)
    A = zeros(length(ng)+1,length(ng))
    B = zeros(length(ng)+1);
    for (i,n) in enumerate(ng)
        A[i,max(i-1,1)] += 1/2*(sig*ng[max(i-1,1)])^2/dn^2;
        
        A[i,i] += -(sig*n)^2/dn^2;
        A[i,min(i+1,Nn)] += 1/2*(sig*ng[min(i+1,Nn)])^2/dn^2;
        if mu > 0
            A[i,min(i+1,Nn)] += - mu*ng[min(i+1,Nn)]/dn;
            A[i,i] += mu*n/dn;
        else
            A[i,i] += -mu*n/dn;
            if i > 1
                A[i,i-1] += mu*ng[i-1]/dn;
            end
        end
    end    
    #B[1] = 1.0;
    #A[1,:] .= dn;
    A = sparse(A);
    
    #g = A\B;
    return A
end

A = populate_A(ng,mu,sig)
sum(A,dims=1)
TranP = (I+A.*dt)';
vscodedisplay(TranP)
gpath = zeros(Nn,T);
gpath[10,1] = 1.0;
AA = I + A
for t= 2:T
    gpath[:,t] = gpath[:,t-1] + A*gpath[:,t-1].*dt
end
plot(ng,gpath[:,end],label="Numerical",lw=6,color=colplot_blue[2])
g,A = solve_stationary_distribution(ng,mu,sig)

colplot_blue = palette(:Blues_3);

g = g./sum(g.*dn)
analytical_n = zeta.*ng.^(-zeta-1)
analytical_n = analytical_n./sum(analytical_n.*dn)
plot(ng,g,label="Numerical",lw=6,color=colplot_blue[2])
plot!(ng,analytical_n,lw=6,linestyle = :dash,color=colplot_blue[3],label="Analytical")
plot!(xlabel="n",title= "Density of Firm Size Distribution, g(n)")
plot!(titlefontfamily = "Computer Modern",
            xguidefontfamily = "Computer Modern",
            yguidefontfamily = "Computer Modern",
            titlefontsize=20,xguidefontsize=12,legendfontsize=12,yguidefontsize=12)
savefig("./figure/density_firm_size_distribution.pdf")
plot(Plots.spy(A))
plot!(cb=:none)
savefig("./figure/spyA.pdf")
