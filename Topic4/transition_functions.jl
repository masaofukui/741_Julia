function HJB_backward(param,ss_result,dvar,dx)
    @unpack_model param
    A = populate_A_HJB(param)
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
    exit_or_not_path = zeros(J,length(tg))
    ng_path = zeros(J,length(tg))
    m_path = zeros(length(tg))
    for i_t = length(tg):-1:1
        if i_t == length(tg)
            w = w_ss + dw;
            Z = Z + dZ;
            v_ahead = copy(v_ss)
        else
            w = copy(w_ss);
            Z = copy(Z);
            v_ahead = copy(v_path[:,i_t+1])
        end
        B = ((r+1/dt).*I - A);
        ng = (alph./w)^(1/(1-alph)).*(Z.*zg)
        pig = (Z.*zg).^(1-alph).*ng.^alph .- w.*ng .- cf + v_ahead./dt 
        v, exit_or_not = Howard_Algorithm(param,B,pig)
        m = (sum(v.*tilde_psig)./ce)^(nu)

        v_path[:,i_t] = v;
        m_path[i_t] = m;
        exit_or_not_path[:,i_t] = exit_or_not;
        ng_path[:,i_t] = ng;
    end
    return (v_path =v_path,
        exit_or_not_path = exit_or_not_path,
        m_path = m_path,
        ng_path = ng_path)
end

function KFE_forward(param,g0,exit_or_not,eta,m,A_ss)
    @unpack_model param
    D = spdiagm(0 => exit_or_not)
    I_D = I-D;
    A = A_ss - spdiagm(0 => eta*ones(J))
    tildeA = A*I_D + D;
    B = -I_D*tilde_psig;  
    g1 = (I - dt*tildeA')\(g0 + m*dt*B);
    return g1
end

function Compute_Sequence_Space_Jacobian(param,ss_result,dvar;dx=0.001)
    @unpack_model param

    underz_ss = ss_result.underz
    g_ss = ss_result.tildeg
    ng_ss = ss_result.ng
    A_ss = populate_A_KFE(param)


    dx_ghost = 0.0;
    HJB_backward_ghost = HJB_backward(param,ss_result,dvar,dx_ghost)
    exit_or_not_path_ghost  = HJB_backward.exit_or_not_path;
    m_path_ghost  = HJB_backward.m_path;


    
    HJB_backward = HJB_backward(param,ss_result,dvar,dx)
    
    
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
        m = m_ghost_path[b_index]
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