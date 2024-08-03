
function entry_dist(xi,zg,Na)
    d = Pareto(xi, 0.1)
    dz = diff(zg)
    psig = pdf(d,zg)
    tilde_psig = copy(psig)
    J = length(zg)
    tilde_dn = copy(psig)
    for j = 1:J
        if j == 1
            tilde_dn[j] = 1/2*dz[j]
            tilde_psig[j] = psig[j]*tilde_dn[j]
        elseif j == J
            tilde_dn[j] =1/2*dz[j-1]
            tilde_psig[j] = psig[j]*tilde_dn[j] 
        else
            tilde_dn[j] = 1/2*(dz[j-1] + dz[j])
            tilde_psig[j] = psig[j]*tilde_dn[j]
        end
    end
    tilde_psig = tilde_psig/sum(tilde_psig)
    tilde_psig_na = zeros(J*Na,1)
    tilde_psig_na[1:J] = tilde_psig;
    return psig,tilde_psig,tilde_dn,tilde_psig_na
end
function populate_A_HJB(param)
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
                A_val[k] = mu.*z/dz_plus;

                k += 1
                A_row[k] = compute_za_index(param,iz,ia)
                A_col[k] = compute_za_index(param,max(iz-1,1),ia)
                A_val[k] = -mu.*z/dz_plus;
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
            da_minus = da[max(ia-1,1)]

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,min(ia+1,Na))
            A_val[k] = 1.0/da_plus;

            k += 1
            A_row[k] = compute_za_index(param,iz,ia)
            A_col[k] = compute_za_index(param,iz,ia)
            A_val[k] = -1.0/da_minus;
        end
    end 

    A = sparse(A_row,A_col,A_val,J*Na,J*Na)
    return A
end

function solve_HJB_VI(param,w)
    @unpack_model param
    A = populate_A_HJB(param)
    B = (r.*I - A);
    ng = (alph./w)^(1/(1-alph)).*zg
    pig = zg.^(1-alph).*ng.^alph .- w.*ng .- cf
    q = -pig + underv.*B*ones(length(zg))
    result = LCPsolve.solve!(LCP(B,q),max_iter=1000)
    println(result.converged)
    x = result.sol
    v = x .+ underv
   
    underz = []
    first_positive = findfirst(x .> 0 )
    if isnothing(first_positive)
        underz_index = J
    elseif first_positive == 1
        underz_index = 1
    else
        underz_index = findfirst(x .> 0 )
        v_noexit = B*v - pig
        v_noexit_interp = linear_interpolation(zg, v_noexit - (v .-underv))
        underz = find_zero(v_noexit_interp, zg[1])
    end
    return v,underz_index,underz,ng
end

function solve_w(param; calibration=0 )
    @unpack_model param
    w_ub = 10;
    w_lb = 0;
    w = (w_ub + w_lb)/2
    err_free_entry = 100
    iter = 0
    underz_index = 0
    v = 0
    ng = 0
    ce_calibrate = 0
    underz = 0
    while iter < 1000 && abs(err_free_entry) > 1e-6
        if calibration != 0
            w = 1;
        else
            w = (w_ub + w_lb)/2
        end
        v,underz_index,underz,ng = solve_HJB_VI(param,w)
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
    @assert iter < 1000
    
    return (w = w, v = v, underz_index  = underz_index,ng=ng,underz=underz,ce_calibrate = ce_calibrate)
end

function compute_entry(param,v,underz)
    @unpack_model param
    underz_grid_down,weight_down,weight_up = closest_index(zg, underz)
    tilde_psig[1:(underz_grid_down-1)] = 0
    tilde_psig[underz_grid_down] = weight_down*tilde_psig[underz_grid_down]
    M*(sum(v.*tilde_psig)./ce)^(nu)


end

function solve_stationary_distribution(param,underz)
    @unpack_model param
    A = populate_A_KFE(param)
    A_store = copy(A)
    A = spdiagm(-eta*ones(J*Na)) + A
    B = -(tilde_psig_na); 
    underz_grid_down,weight_down,weight_up = closest_index(zg, underz) 
    for ia = 1:Na
        underz_set = compute_za_index(param,1:(underz_grid_down-1),ia)
        underz_ia_index = compute_za_index(param,1:(underz_grid_down-1),ia)
        A[underz_set,:] .= 0;
        A[:,underz_set] .= 0;
        A[underz_set,underz_set] = I(length(underz_set));
        A[:,underz_ia_index] = weight_down.*A[:,underz_ia_index]
        B[underz_set] .= 0;
        B[underz_ia_index] = weight_down*B[underz_ia_index];
    end
    g = (A')\B;
    return g,A_store
end