# EC 741 Julia Codes (Boston University)

This repository has Julia codes for the second part of EC741, a part of the second-year macro sequence at Boston University, as taught in fall 2024.

The course covers heterogeneous firm models with a particular focus on the determinants of aggregate labor demand. All the codes are written in continuous time.
* Topic 1: Firm size distribution (Gabaix, 1999).
  * [KFE.jl](./Topic1/KFE.jl): Code to solve Kolmogorov Forward Equation to obtain the firm size distribution in the steady state and in the transition.

* Topic 2: Canonical model of firm dynamics (Hopenhayn and Rogerson, 1993)
  * [Hopenhayn_Rogerson_PE.jl](./Topic2/Hopenhayn_Rogerson_PE.jl): Code to solve HJB-VI using Howard's algorithm to obtain the value and policy functions of optimal stopping time problem (i.e., when to exit).
  * [Hopenhayn_Rogerson_GE.jl](./Topic2/Hopenhayn_Rogerson_GE.jl): Code to solve the general equilibrium of Hopenhayn-Rogerson model with a uniform grid.
  * [Hopenhayn_Rogerson_GE_non_uniform_grid.jl](./Topic2/Hopenhayn_Rogerson_GE_non_uniform_grid.jl): Code to solve the general equilibrium of Hopenhayn-Rogerson model with a non-uniform grid.

* Topic 3: Declining Business Dynamism and Transition Dynamics with Free-entry (Karahan, Şahin, and Pugsley, 2024)
  * [Toplevel_Hopenhayn_Rogerson_business_dynamism.jl](Topic3/Toplevel_Hopenhayn_Rogerson_business_dynamism.jl): Top-level code to study comparative statics across steady states and transition dynamics in response to change in population growth rates and other parameters.
* Topic 4: Transition Dynamics without Free-entry (Auclert, Bardóczy, Rognlie, and Straub, 2021)
  * [Toplevel_Hopenhayn_Rogerson_sequence_space_Jacobian.jl](Topic4/Toplevel_Hopenhayn_Rogerson_sequence_space_Jacobian.jl): Top-level code to implement sequence-space Jacobian algorithms in Hopenhayn-Rogerson model.
* Topic 5: Firm Dynamics with Labor Adjustment Costs (Hopenhayn and Rogerson, 1993)
  * [Toplevel_Hopenhayn_Rogerson_Adj_Cost.jl](Topic5/Toplevel_Hopenhayn_Rogerson_Adj_Cost.jl): Top-levle code to solve Hopenhyan-Rogerson model with hiring costs with an application if linear firing taxes. 
* Topic 6: Firm Dynamics in a Frictional Labor Market (McCrary, 2024)


