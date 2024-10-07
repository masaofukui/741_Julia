
function solve_transition_distribution(param,etapath)
    ce_calibrate = param.ce
    param = model(eta = etapath[1],ce = ce_calibrate)
    HJB_result = solve_w(param; calibration=0)

    @unpack_model param
    @unpack exit_or_not,ng,underz_index = HJB_result
    ss_dist = solve_stationary_distribution(param,HJB_result)
    @unpack tildeg_nonuniform,m,tildeg_nonuniform_normalized = ss_dist
    g0 = copy(reshape(tildeg_nonuniform_normalized,Na*J,1))


    A_store = populate_A_KFE(param)
    D = spdiagm(0 => kron(ones(Na),exit_or_not))
    I_D = I-D;
    tildegpath_nonuniform_normalized = zeros(J*Na,T)
    tildegpath_nonuniform_normalized[:,1] = g0;
    for t = 2:T
        A = A_store - spdiagm(0 => etapath[t]*ones(J*Na))
        tildeA = A*I_D + D;
        B = I_D*tilde_psig_na;  
        tildegpath_nonuniform_normalized[:,t] = (I - dt*tildeA')\(tildegpath_nonuniform_normalized[:,t-1] + dt*B);
    end
    mpath = zeros(T)
    tildegpath_nonuniform = zeros(Na*J,T)
    entry_rate_path = zeros(T)
    Average_size_path = zeros(T)
    Average_age_path = zeros(T)
    firm_mass = zeros(T)
    for t = 1:T
        dist_temp = tildegpath_nonuniform_normalized[:,t];
        dist_temp_mat = reshape(dist_temp,J,Na)
        dist_temp_z = vec(sum(dist_temp_mat,dims=2))
        dist_temp_a = vec(sum(dist_temp_mat,dims=1))
        mpath[t] = L/sum(dist_temp_z.*ng)
        tildegpath_nonuniform[:,t] = dist_temp.*mpath[t]
        entry_rate_path[t] = sum(mpath[t].*tilde_psig[underz_index:end])/sum(tildegpath_nonuniform[:,t])

        Average_size_path[t] = sum(dist_temp_z.*ng)/sum(dist_temp_z)
        Average_age_path[t] = sum(dist_temp_a.*ag)/sum(dist_temp_a)

        firm_mass[t] = sum(tildegpath_nonuniform[:,t])
    end

    dfirm_mass = (firm_mass[2:end] - firm_mass[1:(end-1)])./firm_mass[1:(end-1)]./dt
    dfirm_mass = [dfirm_mass;0]
    exit_rate_path = entry_rate_path .- dfirm_mass .- etapath

    return (entry_rate_path=entry_rate_path,
            mpath=mpath,
            Average_size_path=Average_size_path,
            Average_age_path=Average_age_path,
            exit_rate_path=exit_rate_path)
end


