
function construct_jump_matrix_M(param,HJB_result)
    @unpack_model param
    @unpack i_n_jump,exit_or_not,dn = HJB_result
    
    M_fire_row = 1:(Jn*Jz);
    M_fire_col = ones(Int64,Jn*Jz)
    M_fire_val = zeros(Int64,Jn*Jz)
    for i_n in eachindex(ng)
        for i_z in eachindex(zg)
            indx = compute_nz_index(param,i_n,i_z)
            if exit_or_not[i_n,i_z] == 0
                M_fire_col[indx] = compute_nz_index(param,i_n_jump[i_n,i_z],i_z)
                M_fire_val[indx] = 1
            else
                M_fire_col[indx] = indx
                M_fire_val[indx] = 0
            end
        end
    end
    # M_fire[i,j] denotes when firm in state i jumps to fire to state j
    M = sparse(M_fire_row,M_fire_col,M_fire_val,Jn*Jz,Jn*Jz)
    #=
    exit_or_not = reshape(exit_or_not,Jn*Jz)
    # M_noexit takes one when firm survives
    M_noexit = spdiagm(0 => 1 .- exit_or_not)

    M = M_noexit + M_fire
    =#
    # D is a diagonal matrix with 1 if firm jumps or exits
    D = spdiagm(0=> diag(M) .== 0 )

    return M,D
end

function solve_stationary_distribution(param,HJB_result;m=NaN)
    @unpack_model param
    @unpack dn,exit_or_not = HJB_result
    @assert sum(exit_or_not) > 0 "No exit"
    M,D = construct_jump_matrix_M(param,HJB_result)
    Az = populate_Az(param)
    An = populate_An(param,dn);
    A = An + Az
    tilde_psig_nz = (I-D)*tilde_psig_nz;

    tildeg_nonuniform = []
    tildeg_nonuniform_normalized = []

    if nu == Inf
        tildeg_nonuniform_normalized = (D + (A*M)')\(-tilde_psig_nz);
        ng_repeat = repeat(ng,Jz)
        m = sum(ng_repeat.*tildeg_nonuniform_normalized)/L
        tildeg_nonuniform = tildeg_nonuniform_normalized.*m
    else
        @assert !isnan(m)
        tildeg_nonuniform = (D + (A*M)')\(-m.*tilde_psig_nz);
    end

    return (tildeg_nonuniform=tildeg_nonuniform,
            tildeg_nonuniform_normalized=tildeg_nonuniform_normalized,
            m = m)
end



function compute_moments(param,ss_result)
    @unpack_model param
    HJB_result = ss_result.HJB_result
    dist_result = ss_result.dist_result
    @unpack dn,exit_or_not = HJB_result
    @unpack tildeg_nonuniform,m = dist_result
    M,D = construct_jump_matrix_M(param,HJB_result)
    Az = populate_Az(param)
    An = populate_An(param,dn);
    A = An + Az
    T_within_year = 10
    dt_within_year = 1/T_within_year;
    Delta_one_year =  (inv(I-Matrix(A*M)'.*dt_within_year))^T_within_year;
    Delta_one_year = Delta_one_year'

    dlZg = range(-(1-alph)*1.5*sig,(1-alph)*1.5*sig,length=5)
    dlng_mass =  zeros(length(dlZg))
    dlZg_mass = zeros(length(dlZg))
    dln_Second = 0;
    dln_First = 0;
    dln_Mass = 0;

    for (i_n,n) in enumerate(ng)
        for (i_z,z) in enumerate(zg)
            for (i_n_next,n_next) in enumerate(ng)
                for (i_z_next,z_next) in enumerate(zg)
                    current_index = compute_nz_index(param,i_n,i_z)
                    next_index = compute_nz_index(param,i_n_next,i_z_next)
                    dlZ = (1-alph).*(log(z_next) - log(z));

                    if dlZ < dlZg[1] || dlZ > dlZg[end]
                        continue
                    end

                    dlZ_indx, distance_to_up = closest_index(dlZg,dlZ)
                    dln = log(n_next) - log(n);
                    mass = tildeg_nonuniform[current_index]*Delta_one_year[current_index,next_index]
                    dlZg_mass[dlZ_indx] += mass.*distance_to_up
                    dlng_mass[dlZ_indx] += mass*dln.*distance_to_up
                    dlZg_mass[dlZ_indx+1] += mass.*(1-distance_to_up)
                    dlng_mass[dlZ_indx+1] += mass*dln.*(1-distance_to_up)

                    dln_Mass += mass
                    dln_First += mass*dln
                    dln_Second += mass*dln^2
                end
            end
        end
    end

    dlng = dlng_mass./dlZg_mass
    dln_Mean = dln_First/dln_Mass
    dln_Var = dln_Second/dln_Mass - dln_Mean^2
    dln_std = sqrt(dln_Var)

    tilde_psig_nz_mat = reshape(tilde_psig_nz,Jn,Jz)
    
    entry_rate = sum(m.*tilde_psig_nz_mat.*( 1 .- exit_or_not))/sum(tildeg_nonuniform)

    tildeg_nonuniform_mat = reshape(tildeg_nonuniform,Jn,Jz)
    tildeg_nonuniform_n = sum(tildeg_nonuniform_mat,dims=2)
    ave_size = sum(ng.*tildeg_nonuniform_n)/sum(tildeg_nonuniform_n)


    println("--------- Average Size -----------")
    println(ave_size)
    println("------ Entry Rate ------")
    println(entry_rate)
    println("--------- Std(dlog(n)) -----------")
    println(dln_std)


    return (dlng=dlng,
            dln_Mean=dln_Mean,
            dln_std=dln_std,
            dlZg=dlZg,
            entry_rate=entry_rate,
            ave_size=ave_size)
end

function compute_calibration_targets(param,ss_result)
    @unpack_model param
    @unpack underz_index,ng,tildeg_nonuniform,m,w = ss_result
    
    entry_rate = sum(m.*tilde_psig[underz_index:end])/sum(tildeg_nonuniform)
    Entry = sum(m.*tilde_psig[underz_index:end]);

    entrants_size = sum(ng[underz_index:end].*tilde_psig[underz_index:end])/sum(tilde_psig[underz_index:end])
    ave_size = sum(ng.*tildeg_nonuniform)/sum(tildeg_nonuniform)
    reverse_cumsum_g = reverse(cumsum(reverse(tildeg_nonuniform)))/sum(tildeg_nonuniform)
    emp500_cutoff = findlast(reverse_cumsum_g .> 0.0038)
    share_500 = sum(ng[emp500_cutoff:end].*tildeg_nonuniform[emp500_cutoff:end])/sum(ng.*tildeg_nonuniform)
    Firm_mass = sum(tildeg_nonuniform)
    Exit_rate = entry_rate 
    Exit = Exit_rate*Firm_mass


    println("--------- Average Size -----------")
    println(ave_size)
    println("------ Entrants Size / Incumbents Size ------")
    println(entrants_size/ave_size)
    println("------ Entry Rate ------")
    println(entry_rate)
    println("------ Share of 500+ Firms ------")
    println(share_500)
    println("------ cutoff z ------")
    println(zg[underz_index])
    println("------ cutoff n ------")
    println(ng[underz_index])

    ss_stats = Dict{String,Any}()
    ss_stats["Entry_rate"] = entry_rate
    ss_stats["Entry"] = Entry
    ss_stats["entrants_size"] = entrants_size
    ss_stats["Firm_size"] = ave_size
    ss_stats["share_500"] = share_500
    ss_stats["w"] = w
    ss_stats["Firm_mass"] = Firm_mass
    ss_stats["Exit"] = Exit
    ss_stats["Exit_rate"] = Exit_rate


    return ss_stats
end