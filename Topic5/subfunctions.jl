function compute_nz_index(param,i_n,i_z)
    @unpack_model param
    @assert maximum(i_n) <= Jn && maximum(i_z) <= Jz
    return (i_z .-1).*Jn .+ i_n
end

function entry_dist(xi,ng,zg,tilde_Delta_z,n_start)
    d = Pareto(xi, 0.1)
    psig = pdf(d,zg)
    tilde_psig = copy(psig)
    Jz = length(zg)
    Jn = length(ng)
    for j = 1:Jz
        tilde_psig[j] = psig[j]*tilde_Delta_z[j]
    end
    tilde_psig = tilde_psig/sum(tilde_psig)
    

    ndist = zeros(Jn)
    ndist[n_start] = 1
    tilde_psig_nz = kron(tilde_psig,ndist)
    return tilde_psig_nz
end

function closest_index(vector, target)
    differences = abs.(vector .- target)
    return argmin(differences)
end