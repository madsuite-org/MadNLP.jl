# Workspace of the robust restoration phase

mutable struct RobustRestorer{T, VT}
    obj_val_R::T
    f_R::VT
    x_ref::VT

    theta_ref::T
    D_R::VT
    obj_val_R_trial::T

    pp::VT
    nn::VT
    zp::VT
    zn::VT

    dpp::VT
    dnn::VT
    dzp::VT
    dzn::VT

    pp_trial::VT
    nn_trial::VT

    inf_pr_R::T
    inf_du_R::T
    inf_compl_R::T

    mu_R::T
    tau_R::T
    zeta::T

    filter::Vector{Tuple{T,T}}
end
