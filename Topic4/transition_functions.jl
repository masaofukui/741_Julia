function solve_HJB_VI_transition(param,v_ahead,w,Z_in)
    @unpack_model param
    A = populate_A_HJB(param)
    B = ((r+1/dt).*I - A);
    ng = (alph./w)^(1/(1-alph)).*(Z_in.*zg)
    pig = (Z_in.*zg).^(1-alph).*ng.^alph .- w.*ng .- cf
    q = - (pig + v_ahead./dt)+ underv.*B*ones(length(zg))
    result = LCPsolve.solve!(LCP(B,q),max_iter=1000)
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

function HJB_backward(param,ss_result,dvar,dx)
    @unpack_model param
    if dvar == "w"
        dw = copy(dx);
        dZ = 0.0;
    elseif dvar == "Z"
        dw = 0.0;
        dZ = copy(dx);
    else
        dw = 0.0;
        dZ = 0.0;
    end
    w_ss = ss_result.w
    v_ss = ss_result.v
    v_path = zeros(J,length(tg))
    underz_path = zeros(length(tg))
    ng_path = zeros(J,length(tg))
    m_path = zeros(length(tg))
    for i_t = length(tg):-1:1
        if i_t == length(tg)
            w_in = w_ss + dw;
            Z_in = Z + dZ;
            v_in = copy(v_ss)
        else
            w_in = copy(w_ss);
            Z_in = copy(Z);
            v_in = copy(v_path[:,i_t+1])
        end
        v,underz_index,underz,ng = solve_HJB_VI_transition(param,v_in,w_in,Z_in)
        v_path[:,i_t] = v
        m_path[i_t] = compute_entry(param,v,underz)
        underz_path[i_t] = underz;
        ng_path[:,i_t] = ng;
    end
    return v_path,underz_path,m_path,ng_path
end

function solve_transition_distribution(param,g0,underz_in,eta_in,m_in)
    @unpack_model param
    A,B = populate_A_KFE(param,underz_in,eta_in)
    g1 = (I - dt*A')\(g0 + m_in*dt*B);

    return g1
end

function Compute_Sequence_Space_Jacobian(param,ss_result,dvar;dx=0.001)
    @unpack_model param

    underz_ss = ss_result.underz
    g_ss = ss_result.tildeg
    ng_ss = ss_result.ng
    A_ss,B_ss = populate_A_KFE(param,underz_ss,eta)


    dx_ghost = 0.0;
    v_ghost_path, underz_ghost_path,m_ghost_path, ng_ghost_path = HJB_backward(param,ss_result,dvar,dx_ghost)
    v_path, underz_path,m_path,ng_path = HJB_backward(param,ss_result,dvar,dx)
    
    
    dn_path = (ng_path .- ng_ghost_path)/dx


    
    g1_ghost = zeros(J,length(tg))
    g1 = copy(g1_ghost)
    dg = copy(g1_ghost)
    N_Jacobian = zeros(length(tg),length(tg))
    for s = 1:length(tg)
        b_index = length(tg) -s + 1;
        
        if dvar == "eta" && s == 1
            eta_in = eta + dx;
        else
            eta_in = copy(eta)
        end
        
        underz_in = underz_ghost_path[b_index]
        m_in = m_ghost_path[b_index]
        g1_ghost[:,s] = solve_transition_distribution(param,g_ss,underz_in,eta,m_in)
    
        underz_in = underz_path[b_index]
        m_in = m_path[b_index]
        g1[:,s] = solve_transition_distribution(param,g_ss,underz_in,eta_in,m_in)
    
        dg[:,s] = (g1[:,s] - g1_ghost[:,s])/dx
    
        N_Jacobian[1,s] = sum(dn_path[:,b_index].*g_ss) + sum(ng_ss.*dg[:,s])
    
        dg_t = dg[:,s];
        for t = 2:length(tg)
            dg_t = (I - dt*A_ss')\dg_t
            if s > 1
                N_Jacobian[t,s] = N_Jacobian[t-1,s-1] + sum(ng_ss.*dg_t)
            else
                N_Jacobian[t,s] = sum(ng_ss.*dg_t)
            end
        end
    end
    return N_Jacobian
end