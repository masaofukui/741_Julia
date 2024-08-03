function compute_za_index(param,iz,ia)
    @unpack_model param
    return (ia .-1).*J .+ iz
end

# find nearest index and distance to the upper grid.
function closest_index(x, val)
    ibest = 1
    dxbest = -1e100
    for I in eachindex(x)
        dx = (x[I]-val)
        if dx > dxbest && dx <= 0
            dxbest = dx
            ibest = I
        end
    end
    if ibest +1 <= size(x,1)
        weight_down = (x[ibest + 1] - val)./ ( x[ibest + 1] - x[ibest])
        weight_down = max( min(weight_down,1.0),0.0)
    else
        weight_down = 0.0;
        ibest = size(x,1) - 1;
    end

   @assert (weight_down <=1 && weight_down >= 0) "Something wrong"
   weight_up = 1 - weight_down;

    return ibest,weight_down,weight_up
end
