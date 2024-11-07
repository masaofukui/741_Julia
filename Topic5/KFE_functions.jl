
function solve_stationary_distribution(param,HJB_result,m)
    @unpack_model param
    @unpack exit_or_not = HJB_result
    D = spdiagm(0 => exit_or_not)
    I_D = I-D;
    A = populate_A(param)
    A = A - spdiagm(0 => eta*ones(J))
    tildeA = A*I_D + D;
    B = -I_D*tilde_psig;  
    tildeg_nonuniform_normalized = (tildeA')\B;
    tildeg_nonuniform = tildeg_nonuniform_normalized.*m

    return (tildeg_nonuniform=tildeg_nonuniform,
            tildeg_nonuniform_normalized=tildeg_nonuniform_normalized)
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
    Exit_rate = entry_rate - eta
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