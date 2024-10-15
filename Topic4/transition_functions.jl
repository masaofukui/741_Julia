function HJB_backward(param,ss_result,dvar,dx)
    @unpack_model param
    A = populate_A(param)
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
            Z = 1.0 + dZ;
            v_ahead = copy(v_ss)
        else
            w = copy(w_ss);
            Z = 1.0;
            v_ahead = copy(v_path[:,i_t+1])
        end
        B = ((r+1/dt).*I - A);
        ng = (alph./w)^(1/(1-alph)).*(Z.*zg)
        pig = (Z.*zg).^(1-alph).*ng.^alph .- w.*ng .- cf + v_ahead./dt 
        v, exit_or_not = Howard_Algorithm(param,B,pig,vinit=v_ss)
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

function KFE_forward(param,g0,HJB_backward_result,eta_in,A_ss,b_index)
    @unpack_model param
    @unpack exit_or_not_path,m_path = HJB_backward_result
    exit_or_not = exit_or_not_path[:,b_index]
    m = m_path[b_index]
    D = spdiagm(0 => exit_or_not)
    I_D = I-D;
    A = A_ss - spdiagm(0 => eta_in*ones(J))
    tildeA = A*I_D + D;
    B = I_D*tilde_psig;  
    g1 = (I - dt*tildeA')\(g0 + m*dt*B);
    return g1
end

function Compute_Sequence_Space_Jacobian(param,ss_result,dvar;dx=0.001)
    @unpack_model param

    g_ss = ss_result.tildeg_nonuniform
    ng_ss = ss_result.ng
    A_ss = populate_A(param)


    dx_ghost = 0.0;
    HJB_backward_result_ghost = HJB_backward(param,ss_result,dvar,dx_ghost)    
    HJB_backward_result = HJB_backward(param,ss_result,dvar,dx)
    
    
    dn_path = (HJB_backward_result.ng_path .- HJB_backward_result_ghost.ng_path)/dx


    
    g1_ghost = zeros(J,length(tg))
    g1 = copy(g1_ghost)
    dg = copy(g1_ghost)
    N_Jacobian = zeros(length(tg),length(tg))
    for s = 1:length(tg)
        b_index = length(tg) -s + 1;
        
        if dvar == "eta" && s == 1
            eta_shock = eta + dx;
        else
            eta_shock = copy(eta)
        end
        
        eta_in = copy(eta);
        g1_ghost[:,s] = KFE_forward(param,g_ss,HJB_backward_result_ghost,eta_in,A_ss,b_index)
    
        eta_in = copy(eta_shock);
        g1[:,s] = KFE_forward(param,g_ss,HJB_backward_result,eta_in,A_ss,b_index)
    
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