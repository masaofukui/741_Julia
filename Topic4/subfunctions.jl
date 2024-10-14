function compute_za_index(param,iz,ia)
    @unpack_model param
    return (ia .-1).*J .+ iz
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

function closest_index(vector, target)
    differences = abs.(vector .- target)
    return argmin(differences)
end