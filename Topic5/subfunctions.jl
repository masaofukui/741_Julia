function compute_nz_index(param,i_n,i_z)
    @unpack_model param
    @assert maximum(i_n) <= Jn && maximum(i_z) <= Jz
    @assert minimum(i_n) >= 1 && minimum(i_z) >= 1
    return (i_z .-1).*Jn .+ i_n
end

function entry_dist(xi,ng,zg,tilde_Delta_z,n_start)
    d = Pareto(xi, minimum(zg))
    psig = pdf(d,zg)
    tilde_psig = copy(psig)
    Jz = length(zg)
    Jn = length(ng)
    for j = 1:Jz
        tilde_psig[j] = psig[j]*tilde_Delta_z[j]
    end
    tilde_psig = tilde_psig/sum(tilde_psig)
    
    n_start_grid,distance_to_up = closest_index(ng, n_start)
    ndist = zeros(Jn)
    ndist[n_start_grid] = distance_to_up
    ndist[n_start_grid + 1] = 1 - distance_to_up
    tilde_psig_nz = kron(tilde_psig,ndist)
    return tilde_psig_nz
end

# find nearest index and distance to the upper grid.
function closest_index(x, val)
    ibest = 1
    dxbest = -1e100
    for i in eachindex(x)
        dx = (x[i]-val)
        if dx > dxbest && dx <= 0
            dxbest = dx
            ibest = i
        end
    end
    if ibest +1 <= size(x,1)
        distance_to_up = (x[ibest + 1] - val)./ ( x[ibest + 1] - x[ibest])
        distance_to_up = clamp(distance_to_up,0.0,1.0)
    else
        distance_to_up = 0.0;
        ibest = size(x,1) - 1;
    end

   @assert (distance_to_up <=1 && distance_to_up >= 0) "Something wrong"

    return ibest,distance_to_up
end

