module MadNLPCoreGPUExt

import MadNLPCore
import MadCoreGPU: AbstractGPUVectorOrSubVector, AbstractGPUVector, SDSSolver
import MadNLPCore: get_obj_val_R, get_theta_R, get_inf_pr_R, get_inf_du_R, get_inf_compl_R,
    get_alpha_max_R, get_alpha_z_R, get_varphi_R, get_F, get_varphi_d_R, populate_RR_nn!,
    get_varphi, _get_varphi, get_varphi_d, get_rel_search_norm

include("line_search.jl")
include("restoration.jl")
include("options.jl")

end # module
