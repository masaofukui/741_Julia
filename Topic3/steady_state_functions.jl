function populate_A(param)
    @unpack_model param
    A = spzeros(length(zg),length(zg))
    
    for (i,z) in enumerate(zg)
        dz_plus = dz[min(i,J-1)]
        dz_minus = dz[max(i-1,1)]
        if mu > 0
            A[i,min(i+1,J)] += mu.*z/dz_plus;
            A[i,i] += -mu.*z/dz_plus;
        else
            A[i,i] += mu.*z/dz_minus;
            A[i,max(i-1,1)] += -mu.*z/dz_minus;
        end
        denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;
        A[i,i] += - 1/2*(dz_plus + dz_minus).*(sig*z)^2/denom;
        A[i,max(i-1,1)] += 1/2*dz_plus.*(sig*z)^2/denom;
        A[i,min(i+1,J)] += 1/2*dz_minus.*(sig*z)^2/denom;
    end 
    return A
end
function populate_A_KFE(param)
    @unpack_model param
    A_row = ones(Int64,J*Na*10)
    A_col = ones(Int64,J*Na*10)
    A_val = zeros(Float64,J*Na*10)
    A = spzeros(length(zg),length(zg))
    
    k =0 
    for (iz,z) in enumerate(zg)
        for (ia,a) in enumerate(ag)
            dz_plus = dz[min(iz,J-1)]
            dz_minus = dz[max(iz-1,1)]
            
            if mu > 0
                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,min(iz+1,J),ia)
                A_val[k] = mu.*z/dz_plus;

                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,iz,ia)
                A_val[k] = -mu.*z/dz_plus;
            else
                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,iz,ia)
                A_val[k] = mu.*z/dz_minus;

                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,max(iz-1,1),ia)
                A_val[k] = -mu.*z/dz_minus;
            end
            
            denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,ia)
            A_val[k] = - 1/2*(dz_plus + dz_minus).*(sig*z)^2/denom;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,max(iz-1,1),ia)
            A_val[k] = 1/2*dz_plus.*(sig*z)^2/denom;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,min(iz+1,J),ia)
            A_val[k] = 1/2*dz_minus.*(sig*z)^2/denom;
            
            da_plus = da[min(ia,Na-1)]

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,min(ia+1,Na))
            A_val[k] = 1.0/da_plus;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,ia)
            A_val[k] = -1.0/da_plus;
        end
    end 

    A = sparse(A_row,A_col,A_val,J*Na,J*Na)
    return A
end

function Howard_Algorithm(param,B,pig)
    @unpack_model param
    iter = 1;
    vold = zeros(length(zg));
    vnew = copy(vold);
    exit_or_not = []
    while iter < max_iter 
        val_noexit =  (B*vold .- pig);
        val_exit = vold  .- underv
        exit_or_not =val_noexit  .> val_exit;
        Btilde = B.*(1 .-exit_or_not) + I(J).*(exit_or_not)
        q = pig.*(1 .-exit_or_not) + underv.*(exit_or_not)
        vnew = Btilde\q;
        if norm(vnew - vold) < 1e-6
            break
        end
        vold = copy(vnew)
    end
    return vnew,exit_or_not
end

function solve_HJB_VI(param,w)
    @unpack_model param
    A = populate_A(param)
    B = (r.*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    v,exit_or_not = Howard_Algorithm(param,B,pig)
    underz_index = findlast(exit_or_not .> 0 )
    if isnothing(underz_index)
        underz_index = 1
    end
    return v,underz_index,ng,exit_or_not
end

function solve_w(param; calibration=0 )
    @unpack_model param
    w_ub = 0.78*2;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = [];
    v = [];
    ng = [];
    ce_calibrate = [];
    exit_or_not = [];
    while iter < max_iter && abs(err_free_entry) > 1e-6
        if calibration != 0
            w = 0.78;
        else
            w = (w_ub + w_lb)/2
        end
        v,underz_index,ng,exit_or_not = solve_HJB_VI(param,w)
        err_free_entry = sum(v.*tilde_psig) - ce
        if err_free_entry > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," err_free_entry: ",err_free_entry)
        iter += 1
        if calibration != 0
            ce_calibrate = sum(v.*tilde_psig)
            break
        end
    end
    @assert iter < max_iter
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,ce_calibrate = ce_calibrate,
    exit_or_not = exit_or_not)
end

function solve_stationary_distribution(param,HJB_result)
    @unpack_model param
    @unpack exit_or_not,ng = HJB_result
    D = spdiagm(0 => kron(ones(Na),exit_or_not))
    I_D = I-D;
    A = populate_A_KFE(param)
    A = A - spdiagm(0 => eta*ones(J*Na))
    tildeA = A*I_D + D;
    B = -I_D*tilde_psig_na;  
    tildeg_nonuniform_normalized = (tildeA')\B;
    tildeg_nonuniform_normalized = reshape(tildeg_nonuniform_normalized,J,Na)
    tildeg_nonuniform_normalized_z = sum(tildeg_nonuniform_normalized,dims=2)
    m = L/sum(tildeg_nonuniform_normalized_z.*ng)
    tildeg_nonuniform = tildeg_nonuniform_normalized.*m

    return (tildeg_nonuniform=tildeg_nonuniform,
            m=m,
            tildeg_nonuniform_normalized=tildeg_nonuniform_normalized)
end

function wrapper_eqm(param;calibration=1)
    @unpack_model param
    HJB_result = solve_w(param; calibration=calibration)
    @unpack ng, underz_index = HJB_result
    ss_dist = solve_stationary_distribution(param,HJB_result)
    @unpack tildeg_nonuniform,m = ss_dist
    tildeg_nonuniform_ng = vec(sum(tildeg_nonuniform,dims=2))
    tildeg_nonuniform_ag = vec(sum(tildeg_nonuniform,dims=1))
    tildeg_ag = spdiagm(0 => tilde_Delta_a)\tildeg_nonuniform_ag
    
    entry_rate = sum(m.*tilde_psig[underz_index:end])/sum(tildeg_nonuniform)
    Average_size = sum(tildeg_nonuniform_ng.*ng)/sum(tildeg_nonuniform_ng)
    Average_age = sum(tildeg_nonuniform_ag.*ag)/sum(tildeg_nonuniform_ag)
    Average_size_by_age = zeros(Na)
    Exit_rate_by_age = zeros(Na)
    for ia = 1:Na
        Average_size_by_age[ia] = sum(tildeg_nonuniform[:,ia].*ng)/sum(tildeg_nonuniform[:,ia])
        if ia < Na
            Exit_rate_by_age[ia] = -eta -
            (tildeg_ag[ia+1] - tildeg_ag[ia])/da[ia]/tildeg_ag[ia+1]
        end
    end

    return (entry_rate = entry_rate, Average_size = Average_size, Average_age = Average_age, Average_size_by_age = Average_size_by_age, Exit_rate_by_age = Exit_rate_by_age)
end


function ss_comparative_statics(param,variable_name,variable_grid)

    cf_baseline = param.cf
    r_baseline = param.r
    eta_baseline = param.eta
    ce_calibrate = param.ce
    @unpack_model param
    
    entry_rate_g = zeros(length(variable_grid))
    exit_rate_g = zeros(length(variable_grid))
    ave_size_g = zeros(length(variable_grid))
    ave_age_g = zeros(length(variable_grid))
    ave_size_by_age_g = zeros(length(variable_grid),Na)
    exit_rate_by_age_g = zeros(length(variable_grid),Na)
    for (i,v_i) in enumerate(variable_grid)
        if variable_name == "eta"
            param = model(ce = ce_calibrate,eta = v_i)
        elseif variable_name == "ce"
            param = model(ce = v_i*ce_calibrate)
        elseif variable_name == "cf"
            param = model(ce = ce_calibrate, cf = cf_baseline*v_i)
        elseif variable_name == "r"
            param = model(ce = ce_calibrate, r = v_i)
        end
        eqm_result = wrapper_eqm(param,calibration=0)
        @unpack entry_rate, Average_size, Average_age, Average_size_by_age, Exit_rate_by_age = eqm_result
        entry_rate_g[i] = entry_rate
        exit_rate_g[i] = entry_rate - eta;
        ave_size_g[i] = Average_size
        ave_age_g[i] = Average_age
        ave_size_by_age_g[i,:] = Average_size_by_age
        exit_rate_by_age_g[i,:] = Exit_rate_by_age
    end

    if variable_name == "eta"
        xlabel_str = "Population growth, η"
        xgrid = copy(variable_grid)
        x_baseline = copy(eta_baseline)
    elseif variable_name == "ce"
        xlabel_str = "Entry cost, c_e"
        xgrid = variable_grid.*ce_calibrate
        x_baseline = copy(ce_calibrate)
    elseif variable_name == "cf"
        xlabel_str = "Fixed cost, c_f"
        xgrid = variable_grid.*cf_baseline
        x_baseline = copy(cf_baseline)
    elseif variable_name == "r"
        xlabel_str = "Interest rate, r"
        xgrid = copy(variable_grid)
        x_baseline = copy(r_baseline)
    end

    lw = 4
    plt_entry = plt_fun(param,etag,entry_rate_g; lw = lw, ymin=0,ymax=0,xlabel=xlabel_str,title= "Entry rate",x_baseline=x_baseline)
    ymin, ymax = ylims(plt_entry)
    plt_exit = plt_fun(param,etag,exit_rate_g,lw=lw,ymin=ymin,ymax=ymax,xlabel=xlabel_str,title= "Exit rate",x_baseline=x_baseline)
    plt_ave_size = plt_fun(param,etag,ave_size_g,lw=lw,xlabel=xlabel_str,title= "Average firm size",x_baseline=x_baseline)
    plt_ave_age = plt_fun(param,etag,ave_age_g,lw=lw,xlabel=xlabel_str,title= "Average firm age",x_baseline=x_baseline)

    plt_all = plot(plt_entry,plt_exit,plt_ave_size,plt_ave_age,layout=(2,2),size=(1200,800))
    plot!(margin=5mm)
    display(plt_all)
    if fig_save == 1
        savefig(plt_all,"./figure/Karahan_Sahin_Pugsley_"*variable_name*".pdf")
    end

    plt_size_by_age = plt_by_age_fun(param,etag,ave_size_by_age_g; lw = lw, ymin=0,ymax=0,xlabel=xlabel_str,title= "Average firm size conditional on age")
    plt_exit_by_age = plt_by_age_fun(param,etag,exit_rate_by_age_g,lw=lw,ymin=0,ymax=0,xlabel=xlabel_str,title= "Exit rate conditional on age")


    plt_all = plot(plt_size_by_age,plt_exit_by_age,layout=(1,2),size=(1200,400))
    plot!(margin=5mm)
    display(plt_all)
    if fig_save == 1
        savefig(plt_all,"./figure/By_age_Karahan_Sahin_Pugsley_"*variable_name*".pdf")
    end

    return (entry_rate_g = entry_rate_g, exit_rate_g = exit_rate_g, ave_size_g = ave_size_g,ave_size_by_age_g = ave_size_by_age_g,ave_age_g = ave_age_g,exit_rate_by_age_g = exit_rate_by_age_g)
end