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

function optimal_firing(param,vold; compute_policy=0)
    vnew = copy(vold)
    @unpack_model param
    for i_n in eachindex(ng)
        for i_z in eachindex(zg)
            vnew[i_n,i_z] = maximum(vold[1:i_n,i_z])
        end
    end
    i_n_jump = zeros(Int64,Jn,Jz)
    exit_or_not = zeros(Int64,Jn,Jz)
    if compute_policy == 1
        for i_n in eachindex(ng)
            for i_z in eachindex(zg)
                exit_or_not[i_n,i_z] = (underv >= vnew[i_n,i_z])
                if exit_or_not[i_n,i_z] == 0
                    i_n_jump[i_n,i_z] = findfirst(vold[1:i_n,i_z] .== vnew[i_n,i_z])
                end
            end
        end
    end
    return (v = vnew, i_n_jump = i_n_jump, exit_or_not = exit_or_not)
end


function Howard_Algorithm(param,Az,pig,v_fire;vinit=0)
    @unpack_model param
    iter = 0;
    if vinit == 0
        vold = reshape(pig,Jn,Jz)./r;
    else
        vold = copy(vinit)
    end
    vnew = copy(vold);
    A = copy(Az);
    v_adjust = max.(underv,v_fire)
    adj_or_not = []
    dn = zeros(Jn,Jz)
    h = zeros(Jn,Jz)
    while iter < max_iter
        iter += 1

        
        for i = 1:Jz
            dv_n_plus = (vold[2:Jn,i] - vold[1:(Jn-1),i])./ Delta_n
            dv_n_plus = [dv_n_plus;0]
            dn_plus = h_fun.(dv_n_plus) - s.*ng

            dv_n_minus = (vold[2:Jn,i] - vold[1:(Jn-1),i])./ Delta_n
            dv_n_minus = [0; dv_n_minus]
            dn_minus = h_fun.(dv_n_minus) - s.*ng

            dn[:,i] = dn_plus.*(dn_plus .> 0) + dn_minus.*(dn_minus .< 0);
            h[:,i] = h_fun.(dv_n_plus).*(dn_plus .> 0) + h_fun.(dv_n_minus).*(dn_minus .< 0) + s.*ng.*(dn_plus .< 0).*(dn_minus .> 0);
        end

        An = populate_An(param,dn);
        A = Az + An;
        B = r*I - A;
        h_vec = reshape(h,Jn*Jz)
        pig_g = pig .- g_fun(h_vec)
        vold = reshape(vold,Jz*Jn)
        
        RHS_noadjust =  (B*vold .- pig_g);
        RHS_adjust = vold  .- v_adjust
        adj_or_not =RHS_noadjust  .> RHS_adjust;
        D = spdiagm(0 => adj_or_not)
        Btilde = (I-D)*B + D
        q = pig_g.*(1 .-adj_or_not) + v_adjust.*(adj_or_not)
        vnew = Btilde\q;
        vdiff = maximum(abs.(vnew - vold))
        #println("iter: ",iter," vdiff: ",vdiff)
        if vdiff < 1e-4
            break
        end
        vold = copy(vnew)
        vold = reshape(vold,Jn,Jz)
        
    end
    vnew = reshape(vnew,Jn,Jz)
    @assert iter < max_iter "Howard Algorithm did not converge"
    return (v = vnew, dn = dn)
end



function solve_HJB_QVI(param,w;max_iter_outer=100)
    @unpack_model param
    Az = populate_Az(param)
    v_fire_old = -1e10*ones(Jn*Jz);
    pig = [ zg[i_z].^(1-alph).*ng[i_n].^alph .- w.*ng[i_n] .- cf for i_n = 1:Jn, i_z = 1:Jz]
    pig = reshape(pig,Jz*Jn)
    v_fire_new = []
    v_fire_new_mat = []
    iter = 0;
    dn = [];
    while iter < max_iter_outer
        if iter == 0
            vinit = 0 
        else
            vinit = v_fire_new_mat
        end
        result_hward = Howard_Algorithm(param,Az,pig,v_fire_old,vinit=vinit)
        v = result_hward.v;
        dn = result_hward.dn;
        fire_result = optimal_firing(param,v)
        v_fire_new_mat = fire_result.v;
        v_fire_new = reshape(v_fire_new_mat,Jn*Jz)
        vdiff = maximum(abs.(v_fire_new - v_fire_old))
        if mod(iter,5) == 0
            println("outer loop iter: ",iter," vdiff: ",vdiff)
        end
        if vdiff < 1e-4
            break
        end
        v_fire_old = copy(v_fire_new)
        iter +=1
    end
    @assert iter < max_iter_outer "Outer loop did not converge"
    v_fire_new = reshape(v_fire_new,Jn,Jz)
    fire_result = optimal_firing(param,v_fire_new;compute_policy=1)
    i_n_jump = fire_result.i_n_jump
    exit_or_not = fire_result.exit_or_not
    return (v = v_fire_new, dn = dn,i_n_jump=i_n_jump,exit_or_not=exit_or_not)
end
