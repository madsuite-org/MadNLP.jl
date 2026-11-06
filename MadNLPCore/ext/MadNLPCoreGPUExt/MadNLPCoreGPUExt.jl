module MadNLPCoreGPUExt

import MadNLPCore
import MadCoreGPU: AbstractGPUVectorOrSubVector
import MadNLPCore: get_obj_val_R, get_theta_R, get_inf_pr_R, get_inf_du_R, get_inf_compl_R,
    get_alpha_max_R, get_alpha_z_R, get_varphi_R, get_F, get_varphi_d_R, populate_RR_nn!

include("restoration.jl")

end # module
