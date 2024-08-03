using SparseArrays 
using Plots
using UnicodePlots 
Nn = 1000;
sig = 0.1;
mu = -0.01;

ng = range(1.0,10,length=Nn)
dn = ng[2] - ng[1]
z_star = 1;
z_star_index = 1

zeta = 1 - 2*mu/sig^2

function solve_stationary_distribution_z(ng,mu,sig)
    A = zeros(length(ng),length(ng))
    B = zeros(length(ng));

    
    for (i,n) in enumerate(ng)
        
        if i == 1
            A[z_star_index,1] += 1/2*(sig*n)^2/dn^2
        else 
            A[i,max(i-1,1)] += 1/2*(sig*ng[i-1])^2/dn^2 
        end
        
        #A[i,max(i-1,1)] += 1/2*(sig*n)^2/dn^2 
        #A[i,i] = -chi
        A[i,i] += -(sig*n)^2/dn^2
        if i <= Nn
            A[i,min(i+1,Nn)] += 1/2*(sig*ng[min(i+1,Nn)])^2/dn^2
        end
        if mu > 0
            if i <= Nn
                A[i,min(i+1,Nn)] += - mu*ng[min(i+1,Nn)]/dn
            end
            A[i,i] += mu*n/dn
        else
            A[i,i] += -mu*n/dn
            A[i,max(i-1,1)] += mu*ng[max(i-1,1)]/dn
        end
    end
    
    B[1] = 1.0;
    #A[1,:] .= dn;
    #A[1,1] = 1.0;
    A[1,:] .= 0.0;
    A[1,1] = 1.0;
    A = sparse(A)
    p = A\B
    p = p./sum(p.*dn)
    plt_z = plot(ng,p,label=:none,lw=3)
    plot!(ng,zeros(length(ng)),linestyle=:dash)
    xlabel!("n")
    title!("Distribution of n")
    display(plt_z)
    return p 
end


p = solve_stationary_distribution_z(ng,mu,sig)
sum(p.*dn)
analytical_n = zeta.*ng.^(-zeta-1)
sum(analytical_n.*dn)
plot!(ng,analytical_n,lw=3)
#plot(log.(ng),log.(p),label"log(p(n))",lw=3)

#plot(log.(ng),log.(analytical_n),lw=3)
