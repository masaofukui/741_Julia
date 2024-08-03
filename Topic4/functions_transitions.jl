function solve_HJB_VI_transition(param,w,v_ahead)
    @unpack_model param
    A = populate_A_HJB(param)
    B = ((r+1/dt).*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    q = - (pig + v_ahead./dt)+ underv.*B*ones(length(zg))
    result = LCPsolve.solve!(LCP(B,q),max_iter=1000)
    println(result.converged)
    x = result.sol
    v = x .+ underv
   
    underz = []
    first_positive = findfirst(x .> 0 )
    if isnothing(first_positive)
        underz_index = J
    elseif first_positive == 1
        underz_index = 1
    else
        underz_index = findfirst(x .> 0 )
        v_noexit = B*v - pig
        v_noexit_interp = linear_interpolation(zg, v_noexit - (v .-underv))
        underz = find_zero(v_noexit_interp, zg[1])
    end
    return v,underz_index,underz,ng
end


function solve_transition_distribution(param,underz_index,g0,etapath)
    @unpack_model param
    A_store = populate_A_KFE(param)
    tilde_hatgpath = zeros(J*Na,T)
    tilde_hatgpath[:,1] = g0;
    for t = 2:T
        A = spdiagm(-etapath[t]*ones(J*Na)) + A_store
        B = (tilde_psig_na);  
        for ia = 1:Na
            underz_set = compute_za_index(param,1:(underz_index-1),ia)
            A[underz_set,:] .= 0;
            A[:,underz_set] .= 0;
            #A[underz_set,underz_set] = I(length(underz_set));
            B[underz_set] .= 0;
        end
        tilde_hatgpath[:,t] = (I - dt*A')\(tilde_hatgpath[:,t-1] + dt*B);
    end

    return tilde_hatgpath
end