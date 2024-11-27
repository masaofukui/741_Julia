function populate_Az(param)
    @unpack_model param
    A_row = zeros(Int32,Jz*Jn*5)
    A_col = zeros(Int32,Jz*Jn*5)
    A_val = zeros(Float64,Jz*Jn*5)

    k =0
    for (i_n,n) in enumerate(ng)
        for (i_z,z) in enumerate(zg)
            dz_plus = Delta_z[min(i_z,Jz-1)]
            dz_minus = Delta_z[max(i_z-1,1)]
            
            if mu > 0
                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,i_n,i_z)
                A_val[k] = -mu.*z/dz_plus;

                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,i_n,clamp(i_z+1,1,Jz))
                A_val[k] = mu.*z/dz_plus;
            else
                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,i_n,i_z)
                A_val[k] = mu.*z/dz_minus;

                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,i_n,clamp(i_z-1,1,Jz))
                A_val[k] = -mu.*z/dz_minus;
            end
            denom = 1/2*(dz_plus + dz_minus)*dz_plus*dz_minus;

            k += 1
            A_row[k] = compute_nz_index(param,i_n,i_z)
            A_col[k] = compute_nz_index(param,i_n,i_z)
            A_val[k] = - 1/2*(dz_plus + dz_minus).*(sig*z)^2/denom;

            k += 1
            A_row[k] = compute_nz_index(param,i_n,i_z)
            A_col[k] = compute_nz_index(param,i_n,clamp(i_z-1,1,Jz))
            A_val[k] = 1/2*dz_plus.*(sig*z)^2/denom;
            k += 1
            A_row[k] = compute_nz_index(param,i_n,i_z)
            A_col[k] = compute_nz_index(param,i_n,clamp(i_z+1,1,Jz))
            A_val[k] = 1/2*dz_minus.*(sig*z)^2/denom;
        end
    end
    Az = sparse(A_row[1:k],A_col[1:k],A_val[1:k])
    return Az
end

function populate_An(param,dn)
    @unpack_model param
    A_row = zeros(Int32,Jz*Jn*5)
    A_col = zeros(Int32,Jz*Jn*5)
    A_val = zeros(Float64,Jz*Jn*5)

    k =0
    for (i_n,n) in enumerate(ng)
        for (i_z,z) in enumerate(zg)
            Deltan_plus = Delta_n[min(i_n,Jn-1)]
            Deltan_minus = Delta_n[max(i_n-1,1)]

            if dn[i_n,i_z] > 0
                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,i_n,i_z)
                A_val[k] = -dn[i_n,i_z]/Deltan_plus;

                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,clamp(i_n+1,1,Jn),i_z)
                A_val[k] = dn[i_n,i_z]/Deltan_plus;
            else
                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,i_n,i_z)
                A_val[k] = dn[i_n,i_z]/Deltan_minus;

                k += 1
                A_row[k] = compute_nz_index(param,i_n,i_z)
                A_col[k] = compute_nz_index(param,clamp(i_n-1,1,Jn),i_z)
                A_val[k] = -dn[i_n,i_z]/Deltan_minus;
            end
            
        end
    end
    An = sparse(A_row[1:k],A_col[1:k],A_val[1:k])
    return An
end

function optimal_firing(param,Sold,U; compute_policy=0)
    Snew = copy(Sold)
    @unpack_model param
    for i_n in eachindex(ng)
        for i_z in eachindex(zg)
            Snew[i_n,i_z] = maximum(Sold[1:i_n,i_z])
        end
    end
    Sexit = compute_Sexit(param,U)
    Sexit_mat = reshape(Sexit,Jn,Jz)
    i_n_jump = zeros(Int64,Jn,Jz)
    exit_or_not = zeros(Int64,Jn,Jz)
    if compute_policy == 1
        for i_n in eachindex(ng)
            for i_z in eachindex(zg)
                exit_or_not[i_n,i_z] = (Sexit_mat[i_n,i_z] >= Snew[i_n,i_z])
                if exit_or_not[i_n,i_z] == 0
                    i_n_jump[i_n,i_z] = findfirst(Sold[1:i_n,i_z] .== Snew[i_n,i_z])
                end
            end
        end
    end
    return (v = Snew, i_n_jump = i_n_jump, exit_or_not = exit_or_not)
end


function Howard_Algorithm(param,Az,pig,S_fire,theta,U;Sinit=0)
    @unpack_model param
    iter = 0;
    if Sinit == 0
        Sold = reshape(pig,Jn,Jz)./r;
    else
        Sold = copy(Sinit)
    end
    Snew = copy(Sold);
    A = copy(Az);
    Sexit = compute_Sexit(param,U)
    S_adjust = max.(Sexit,S_fire)
    adj_or_not = []
    dn = zeros(Jn,Jz)
    v = zeros(Jn,Jz)
    Phi = zeros(Jn,Jz)
    while iter < max_iter
        iter += 1

        diag_S = zeros(Jn,Jz)
        for i = 1:Jz
            dS_n_plus = (Sold[2:Jn,i] - Sold[1:(Jn-1),i])./ Delta_n
            dS_n_plus = [dS_n_plus;0]
            value_hiring = qfun(theta).*dS_n_plus + qfun(theta).*Sold[:,i]./ng.*gamma
            v_plus = v_fun.(value_hiring,ng)
            dn_plus = qfun.(theta).*v_plus - s.*ng
            HF = -Phi_fun.(v_plus,ng) + dS_n_plus.*dn_plus

            dS_n_minus = (Sold[2:Jn,i] - Sold[1:(Jn-1),i])./ Delta_n
            dS_n_minus = [0; dS_n_minus]
            value_hiring = qfun(theta).*dS_n_minus + qfun(theta).*Sold[:,i]./ng.*gamma
            v_minus = v_fun.(value_hiring,ng)
            dn_minus = qfun.(theta).*v_minus - s.*ng
            HB = -Phi_fun.(v_plus,ng) + dS_n_minus.*dn_minus

            
            dn[:,i] = dn_plus.*(dn_plus .> 0).*(dn_minus .> 0) + dn_plus.*(HF .> HB).*(dn_plus .> 0).*(dn_minus .< 0 ) + dn_minus.*(dn_minus .< 0 ).*(dn_plus .< 0 ) + dn_minus.*(HF .< HB).*(dn_plus .> 0).*(dn_minus .< 0 );
            v[:,i] = (dn[:,i] + s.*ng)./qfun.(theta); 
            Phi[:,i] = Phi_fun(v[:,i],ng)
            diag_S[:,i] = v[:,i].*qfun.(theta)./ng.*gamma

        end

        diag_S = reshape(diag_S,Jn*Jz)
        An = populate_An(param,dn);
        A = Az + An;
        B = r*I + spdiagm(0=> diag_S) - A;
        #B = r*I  - A;
        Phi_vec = reshape(Phi,Jn*Jz)
        pig_g = pig .- Phi_vec
        Sold = reshape(Sold,Jz*Jn)
        
        RHS_noadjust =  (B*Sold .- pig_g);
        RHS_adjust = Sold  .- S_adjust
        adj_or_not =RHS_noadjust  .> RHS_adjust;
        D = spdiagm(0 => adj_or_not)
        Btilde = (I-D)*B + D
        q = pig_g.*(1 .-adj_or_not) + S_adjust.*(adj_or_not)
        Snew = Btilde\q;
        Sdiff = maximum(abs.(Snew - Sold))
        println("iter: ",iter," Sdiff: ",Sdiff)
        if Sdiff < 1e-4
            break
        end
        Sold = copy(Snew)
        Sold = reshape(Sold,Jn,Jz)
        
    end
    Snew = reshape(Snew,Jn,Jz)
    @assert iter < max_iter "Howard Algorithm did not converge"
    return (S = Snew, dn = dn)
end



function solve_HJB_QVI(param,theta,U;max_iter_outer=100)
    @unpack_model param
    Az = populate_Az(param)
    S_fire_old = -1e10*ones(Jn*Jz);
    pig = [ zg[i_z].^(1-alph).*ng[i_n].^alph .- cf - r.*ng[i_n].*U for i_n = 1:Jn, i_z = 1:Jz]
    pig = reshape(pig,Jz*Jn)
    v_fire_new = []
    S_fire_new_mat = []
    iter = 0;
    dn = [];
    while iter < max_iter_outer
        if iter == 0
            Sinit = 0 
        else
            Sinit = S_fire_new_mat
        end
        result_hward = Howard_Algorithm(param,Az,pig,S_fire_old,theta,U, Sinit=Sinit)
        S = result_hward.S;
        dn = result_hward.dn;
        fire_result = optimal_firing(param,S,U)
        S_fire_new_mat = fire_result.v;
        S_fire_new = reshape(S_fire_new_mat,Jn*Jz)
        Sdiff = maximum(abs.(S_fire_new - S_fire_old))
        if mod(iter,5) == 0
            println("outer loop iter: ",iter," Sdiff: ",Sdiff)
        end
        if Sdiff < 1e-4
            break
        end
        S_fire_old = copy(S_fire_new)
        iter +=1
    end
    @assert iter < max_iter_outer "Outer loop did not converge"
    S_fire_new = reshape(S_fire_new,Jn,Jz)
    fire_result = optimal_firing(param,S_fire_new,U;compute_policy=1)
    i_n_jump = fire_result.i_n_jump
    exit_or_not = fire_result.exit_or_not
    return (v = v_fire_new, dn = dn,i_n_jump=i_n_jump,exit_or_not=exit_or_not,pig=pig)
end

function compute_Sexit(param,U)
    @unpack_model param
    ng_repeat = repeat(ng,Jz)
    Sexit = underJ .+ng_repeat.*U
    return Sexit
end