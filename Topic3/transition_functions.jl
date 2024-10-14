
function solve_transition_distribution(param,shockpath; shock = "eta")
    @unpack_model param
    ce_calibrate = param.ce

    etapath = zeros(length(shockpath))
    Lpath = L*ones(length(shockpath))
    chipath = ones(length(shockpath))
    if shock == "eta"
        etapath = copy(shockpath)
    elseif shock == "L"
        Lpath = copy(shockpath)
    elseif shock == "chi"
        chipath = copy(shockpath)
    else
        println("shock not recognized")
    end
    param = model(eta = etapath[1],ce = ce_calibrate,L = Lpath[1])
    HJB_result = solve_w(param; calibration=0)

    @unpack_model param
    @unpack exit_or_not,ng,underz_index,w = HJB_result
    ss_dist = solve_stationary_distribution(param,HJB_result)
    @unpack tildeg_nonuniform,m,tildeg_nonuniform_normalized = ss_dist
    g0 = copy(reshape(tildeg_nonuniform_normalized,Na*J,1))


    A_store = populate_A_KFE(param)
    D = spdiagm(0 => kron(ones(Na),exit_or_not))
    I_D = I-D;
    tildegpath_nonuniform_normalized = zeros(J*Na,T)
    mpath = zeros(T)
    tildegpath_nonuniform = zeros(Na*J,T)
    entry_rate_path = zeros(T)
    Average_size_path = zeros(T)
    Average_age_path = zeros(T)
    firm_mass_ss = sum(tildeg_nonuniform)
    firm_mass = zeros(T)
    wpath = w*ones(T)
    for t = 1:T
        if t == 1
            gold = g0;
        else
            gold = tildegpath_nonuniform_normalized[:,t-1];
        end
        A = A_store - spdiagm(0 => (etapath[t]+chipath[t])*ones(J*Na))
        tildeA = A*I_D + D;
        B = I_D*tilde_psig_na;  
        tildegpath_nonuniform_normalized[:,t] = (I - dt*tildeA')\(gold + dt*B);
        dist_temp = tildegpath_nonuniform_normalized[:,t];
        dist_temp_mat = reshape(dist_temp,J,Na)
        dist_temp_z = vec(sum(dist_temp_mat,dims=2))
        dist_temp_a = vec(sum(dist_temp_mat,dims=1))

        # Mass of potential entrants
        mpath[t] = Lpath[t]/sum(dist_temp_z.*ng)
        # Non-normalized distribution
        tildegpath_nonuniform[:,t] = dist_temp.*mpath[t]
        # Entry rates
        entry_rate_path[t] = sum(mpath[t].*tilde_psig[underz_index:end])/sum(tildegpath_nonuniform[:,t])
        # Average size
        Average_size_path[t] = sum(dist_temp_z.*ng)/sum(dist_temp_z)
        # Average age
        Average_age_path[t] = sum(dist_temp_a.*ag)/sum(dist_temp_a)
        # Mass of all firms
        firm_mass[t] = sum(tildegpath_nonuniform[:,t])
    end

    # compute exit rates, which is equal to entry rates minus the growth rate of the mass of firms and the population growth rates
    dfirm_mass = (firm_mass[1:end] - [firm_mass_ss; firm_mass[1:(end-1)]])./firm_mass[1:(end)]./dt
    #dfirm_mass = [dfirm_mass;0]
    exit_rate_path = entry_rate_path .- dfirm_mass .- etapath

    return (entry_rate_path=entry_rate_path,
            mpath=mpath,
            Average_size_path=Average_size_path,
            Average_age_path=Average_age_path,
            exit_rate_path=exit_rate_path,
            firm_mass=firm_mass,
            wpath=wpath)
end


