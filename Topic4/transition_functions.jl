function HJB_backward(param,ss_result,dvar,dx;compute_underz_smooth_pasting=0)
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

        # if we want to be rigorous, compute exit threshold using the first-order approximation of smooth pasting, v'(\underline{z}) = 0
        if compute_underz_smooth_pasting == 1
            exit_or_not = compute_underz(v,ss_result)
        end

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
    ss_stats = compute_calibration_targets(param,ss_result)

    g_ss = ss_result.tildeg_nonuniform
    ng_ss = ss_result.ng
    underz_index_ss = ss_result.underz_index
    m_ss = ss_result.m
    exit_or_not_ss = ss_result.exit_or_not
    A_ss = populate_A(param)
    D = spdiagm(0 => exit_or_not_ss)
    I_D = I-D;
    A = A_ss - spdiagm(0 => eta*ones(J))
    tildeA_ss = A*I_D + D;


    dx_ghost = 0.0;
    HJB_backward_result_ghost = HJB_backward(param,ss_result,dvar,dx_ghost)    
    HJB_backward_result = HJB_backward(param,ss_result,dvar,dx)
    
    
    dn_path = (HJB_backward_result.ng_path .- HJB_backward_result_ghost.ng_path)/dx
    dm_path = (HJB_backward_result.m_path .- HJB_backward_result_ghost.m_path)/dx
    dexit_or_not_path = (HJB_backward_result.exit_or_not_path .- HJB_backward_result_ghost.exit_or_not_path)/dx


    
    g1_ghost = zeros(J,length(tg))
    g1 = copy(g1_ghost)
    dg = copy(g1_ghost)
    Jacobian = Dict{String,Any}()
    for v in ["N","Entry","Exit","Exit_rate","Firm_mass","Entry_rate","Firm_size"]
        Jacobian[v] = zeros(length(tg),length(tg))
    end
    Jacobian["ss0"] = Dict{String,Any}()
    Jacobian["ss1"] = Dict{String,Any}()
    Jacobian["dss"] = Dict{String,Any}()
    for s = 1:length(tg)
        b_index = length(tg) -s + 1;
        
        g0_in = copy(g_ss);
        if dvar == "eta" && s == 1
            eta_shock = eta + dx;
        elseif dvar == "initial_eta" && s==1
            eta_shock = copy(eta)
            # 0 is the old steady state with eta + dx, 1 is the new steady state with eta.
            for i in [0,1]
                param_eta_init = deepcopy(param)
                param_eta_init.eta = eta + dx*(1-i);
                ss0_result = solve_w(param_eta_init)
                ss0_stats = compute_calibration_targets(param_eta_init,ss0_result)
                Jacobian["ss"*string(i)]["g0"] = ss0_result.tildeg_nonuniform
                for v in ["w","Entry","Firm_mass","Entry_rate","Firm_size"] 
                    Jacobian["ss"*string(i)][v] = ss0_stats[v]
                end
            end
            g0_in = Jacobian["ss0"]["g0"]
            for v in ["w","Entry","Firm_mass","Entry_rate","Firm_size"] 
                Jacobian["dss"][v] = (Jacobian["ss0"][v] - Jacobian["ss1"][v])./dx
            end
        else
            eta_shock = copy(eta)
        end
        
        eta_in = copy(eta);
        g1_ghost[:,s] = KFE_forward(param,g_ss,HJB_backward_result_ghost,eta_in,A_ss,b_index)
    
        eta_in = copy(eta_shock);
        g1[:,s] = KFE_forward(param,g0_in,HJB_backward_result,eta_in,A_ss,b_index)
    
        dg[:,s] = (g1[:,s] - g1_ghost[:,s])/dx
    
        Jacobian["N"][1,s] = sum(dn_path[:,b_index].*g_ss) + sum(ng_ss.*dg[:,s])
        Jacobian["Firm_mass"][1,s] = sum(dg[:,s])
        Jacobian["Entry"][1,s] = sum(dm_path[b_index].*tilde_psig[underz_index_ss:end]) + sum(m_ss.*dexit_or_not_path[:,b_index].*tilde_psig)
        Jacobian["Exit"][1,s] = Jacobian["Entry"][1,s] - Jacobian["Firm_mass"][1,s]
    
        dg_t = dg[:,s];
        for t = 1:length(tg)
            if t > 1
                dg_t = (I - dt*tildeA_ss')\dg_t
                if s > 1
                    Jacobian["N"][t,s] =  Jacobian["N"][t-1,s-1] + sum(ng_ss.*dg_t)
                    Jacobian["Firm_mass"][t,s] = Jacobian["Firm_mass"][t-1,s-1] + sum(dg_t)
                    Jacobian["Entry"][t,s] = copy(Jacobian["Entry"][t-1,s-1])
                else
                    Jacobian["N"][t,s] = sum(ng_ss.*dg_t)
                    Jacobian["Firm_mass"][t,s] = sum(dg_t)
                end
                Jacobian["Exit"][t,s] = Jacobian["Entry"][t,s] - (Jacobian["Firm_mass"][t,s] - Jacobian["Firm_mass"][t-1,s])
            end
            for v in ["Entry","Exit"]
                Jacobian[v*"_rate"][t,s] = (Jacobian[v][t,s]*ss_stats["Firm_mass"]-Jacobian["Firm_mass"][t,s]*ss_stats[v]) /(ss_stats["Firm_mass"]^2)
            end
            Jacobian["Firm_size"][t,s] = - Jacobian["Firm_mass"][t,s]/(ss_stats["Firm_mass"]^2)
        end
    end
    return Jacobian
end

function compute_underz(v,ss_result)
    v_ss = ss_result.v
    exit_or_not = ss_result.exit_or_not
    underz_index = ss_result.underz_index
    # formula:
    # v''(\underline{z}) d\underline{z} + dv'(\underline{z}) = 0

    dv_zz_ss = compute_2nd_derivative(param,underz_index,v_ss);
    dv_z_ss = compute_1st_derivative(param,underz_index,v_ss);
    dv_z = compute_1st_derivative(param,underz_index,v);

    dunderz = - (dv_z-dv_z_ss)/dv_zz_ss;
    exit_or_not_out = Float64.(exit_or_not)
    if dunderz > 0
        exit_or_not_out[underz_index+1] = dunderz/tilde_Delta_z[underz_index+1];
    else
        exit_or_not_out[underz_index] = 1.0 - dunderz/tilde_Delta_z[underz_index];
    end
    return exit_or_not_out
end

function compute_2nd_derivative(param,i,v)
    @unpack_model param
    dz_plus = dz[min(i,J-1)]
    dz_minus = dz[max(i-1,1)]
    denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;
    dv_zz = (dz_minus.*v[min(i+1,J)] - (dz_plus + dz_minus)*v[i] + dz_plus*v[max(i-1,1)])/denom;
    return dv_zz
end

function compute_1st_derivative(param,i,v)
    @unpack_model param
    dz_plus = dz[min(i,J-1)]
    dz_minus = dz[max(i-1,1)]
    dv_z = (v[min(i+1,J)] - v[max(i-1,1)])/(dz_plus+dz_minus);
    return dv_z
end


function SSJ_wrapper(param; shock = "eta")
    @unpack_model param
    ss_result = solve_w(param; calibration=0 )
    ss_stats = compute_calibration_targets(param,ss_result)

    Jacobian_dict = Dict{String,Any}()
    for dvar in ["w","eta","initial_eta"]
        Jacobian_dict[dvar] = Compute_Sequence_Space_Jacobian(param,ss_result,dvar,dx=0.0001)
    end
    etapath = zeros(T)
    if shock == "eta"
        etapath[1] = 0.02
        etapath = max.(0.02 .- 0.02/40 .*tg,0)
        initial_eta = copy(etapath[1])
    elseif shock == "exit"
        etapath[1] = 0.02
        initial_eta = 0;
    end



    IRF_path = Dict{String,Any}()
    for v in ["w","Entry_rate","Firm_size","Entry","Firm_mass"]
        if v == "w"
            IRF_path[v] = - Jacobian_dict["w"]["N"]\(Jacobian_dict["eta"]["N"]*etapath + Jacobian_dict["initial_eta"]["N"][:,1]*initial_eta);
        else
            IRF_path[v] = Jacobian_dict["w"][v]*IRF_path["w"] + Jacobian_dict["eta"][v]*etapath + Jacobian_dict["initial_eta"][v][:,1]*initial_eta
        end

        if v == "w" || v == "Firm_size" || v == "Firm_mass"
            IRF_path[v*"_plot"] = (IRF_path[v] .- Jacobian_dict["initial_eta"]["dss"][v].*initial_eta)./(Jacobian_dict["initial_eta"]["ss0"][v] + Jacobian_dict["initial_eta"]["dss"][v].*initial_eta)
        elseif v == "Entry_rate"
            IRF_path[v*"_plot"] =  (IRF_path[v] .- Jacobian_dict["initial_eta"]["dss"][v].*initial_eta)
        end
    end
    return IRF_path,Jacobian_dict
end