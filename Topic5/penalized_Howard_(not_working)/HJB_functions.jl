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

function optimal_firing(param,vold)
    vnew = copy(vold)
    @unpack_model param
    Firing_matrix = zeros(Jn*Jz,Jn*Jz)
    for i_n in eachindex(ng)
        for i_z in eachindex(zg)
            vnew[i_n,i_z] = maximum(vold[1:i_n,i_z])
            indx = argmax(vold[1:i_n,i_z])
            Firing_matrix[compute_nz_index(param,i_n,i_z),compute_nz_index(param,indx,i_z)] = 1
        end
    end
    return vnew,Firing_matrix
end


function Howard_Algorithm(param,Az,pig,v_fire;vinit=0,update_v_fire=0)
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
    while iter < max_iter
        iter += 1
        if update_v_fire == 1
            v_fire = optimal_firing(param,vold)[1]
            v_fire = reshape(v_fire,Jn*Jz)
            v_adjust = max.(underv,v_fire)
        end
        dn = zeros(Jn,Jz)
        h = zeros(Jn,Jz)
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
        println("iter: ",iter," vdiff: ",vdiff)
        if vdiff < 1e-4
            break
        end
        vold = copy(vnew)
        vold = reshape(vold,Jn,Jz)
        
    end
    vnew = reshape(vnew,Jn,Jz)
    @assert iter < max_iter "Howard Algorithm did not converge"
    return vnew
end



function Howard_Algorithm_penalized(param,Az,pig,v_fire;vinit=0,update_v_fire=0,epsilon=1e-1)
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
    while iter < max_iter
        iter += 1
        if update_v_fire == 1
            v_fire,Firing_matrix = optimal_firing(param,vold)
            v_fire = reshape(v_fire,Jn*Jz)
            v_adjust = max.(underv,v_fire)
            Firing_matrix[v_fire .< underv,:] .=0;
            Firing_matrix = sparse(Firing_matrix)
        end
        dn = zeros(Jn,Jz)
        h = zeros(Jn,Jz)
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

        adj_or_not = v_adjust - vold .> 0 
        D = spdiagm(0 => adj_or_not)
        Btilde = B + 1/epsilon .* (D - Firing_matrix)
        q = pig_g
        vnew = Btilde\q;
        vdiff = maximum(abs.(vnew - vold))
        println("iter: ",iter," vdiff: ",vdiff)
        if vdiff < 1e-4
            break
        end
        vold = copy(vnew)
        vold = reshape(vold,Jn,Jz)
        
    end
    vnew = reshape(vnew,Jn,Jz)
    @assert iter < max_iter "Howard Algorithm did not converge"
    return vnew
end
function solve_HJB_VI(param,w;vinit=0, penalized=1)
    @unpack_model param
    Az = populate_Az(param)
    v_adjust = -1e10*ones(Jn*Jz);
    pig = [ zg[i_z].^(1-alph).*ng[i_n].^alph .- w.*ng[i_n] .- cf for i_n = 1:Jn, i_z = 1:Jz]
    pig = reshape(pig,Jz*Jn)
    if penalized == 1
        v = Howard_Algorithm_penalized(param,Az,pig,v_adjust,update_v_fire=1,vinit=vinit)
    else
        v = Howard_Algorithm(param,Az,pig,v_adjust,update_v_fire=1,vinit=vinit)
    end

    return v
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
    while iter < max_iter_outer
        if iter == 0
            v = Howard_Algorithm(param,Az,pig,v_fire_old)
        else
            v = Howard_Algorithm(param,Az,pig,v_fire_old,vinit=v_fire_new_mat)
        end
        v_fire_new_mat = optimal_firing(param,v)[1];
        v_fire_new = reshape(v_fire_new_mat,Jn*Jz)
        vdiff = maximum(abs.(v_fire_new - v_fire_old))
        println("outer loop iter: ",iter," vdiff: ",vdiff)
        if vdiff < 1e-4
            break
        end
        v_fire_old = copy(v_fire_new)
        iter +=1
    end
    v_fire_new = reshape(v_fire_new,Jn,Jz)
    return v_fire_new
end


function solve_w(param; calibration=0,tol=1e-8 )
    @unpack_model param
    w_ub = 20;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    excess_labor = 100
    iter = 0
    underz_index = [];
    v = [];
    ng = [];
    ce_calibrate = [];
    exit_or_not = [];
    tildeg_nonuniform = [];
    m = []
    while iter < max_iter && abs(excess_labor) > tol
        if calibration != 0
            w = 0.78;
        else
            w = (w_ub + w_lb)/2
        end
        HJB_result = solve_HJB_VI(param,w)
        @unpack v,underz_index,ng,exit_or_not = HJB_result
        m = (sum(v.*tilde_psig)./ce)^(nu)
        dist_result = solve_stationary_distribution(param,HJB_result,m)
        @unpack tildeg_nonuniform = dist_result
        excess_labor = sum(ng.*tildeg_nonuniform) - L
        
        if excess_labor > 0
            w_lb = w
        else
            w_ub = w
        end
        println("iter: ",iter," w: ",w," excess labor demand: ",excess_labor)
        iter += 1
        if calibration != 0
            ce_calibrate = sum(v.*tilde_psig)
            break
        end
    end
    @assert iter < max_iter
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,ce_calibrate = ce_calibrate,
    exit_or_not = exit_or_not,
    tildeg_nonuniform=tildeg_nonuniform,
    m=m)
end
