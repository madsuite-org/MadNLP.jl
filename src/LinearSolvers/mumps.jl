setindex(tup,a,n) = (tup[1:n-1]...,a,tup[n+1:end]...)
tzeros(n) = tuple((0 for i=1:n)...)

@kwdef mutable struct MumpsOptions <: AbstractOptions
    mumps_mem_percent::Int = 35
    mumps_permuting_scaling::Int = 0
    mumps_pivot_order::Int = 0
    mumps_pivtol::Float64 = 1e-6
    mumps_pivtolmax::Float64 = 1e-1
    mumps_scaling::Int = 1
end

@kwdef mutable struct Struc{T}
    sym::Cint = 0
    par::Cint = 0
    job::Cint = 0

    comm_fortran::Cint = 0

    icntl::NTuple{60,Cint} = tzeros(60)
    keep::NTuple{500,Cint} = tzeros(500)
    cntl::NTuple{15,T} = tzeros(15)
    dkeep::NTuple{230,T} = tzeros(230)
    keep8::NTuple{150,Int64} = tzeros(150)
    n::Cint = 0
    nblk::Cint = 0

    nz_alloc::Cint = 0

    nz::Cint = 0
    nnz::Int64 = 0
    irn::Ptr{Cint} = C_NULL
    jcn::Ptr{Cint} = C_NULL
    a::Ptr{T} = C_NULL

    nz_loc::Cint = 0
    nnz_loc::Int64 = 0
    irn_loc::Ptr{Cint} = C_NULL
    jcn_loc::Ptr{Cint} = C_NULL
    a_loc::Ptr{T} = C_NULL ###

    nelt::Cint = 0
    eltptr::Ptr{Cint} = C_NULL
    eltvar::Ptr{Cint} = C_NULL
    a_elt::Ptr{T} = C_NULL

    blkptr::Ptr{Cint} = C_NULL
    blkvar::Ptr{Cint} = C_NULL

    perm_in::Ptr{Cint} = C_NULL

    sym_perm::Ptr{Cint} = C_NULL
    uns_perm::Ptr{Cint} = C_NULL

    colsca::Ptr{T} = C_NULL
    rowsca::Ptr{T} = C_NULL
    colsca_from_mumps::Cint = 0
    rowsca_from_mumps::Cint = 0

    colsca_loc::Ptr{T} = C_NULL
    rowsca_loc::Ptr{T} = C_NULL

    rowind::Ptr{Cint} = C_NULL
    colind::Ptr{Cint} = C_NULL
    pivots::Ptr{T} = C_NULL

    rhs::Ptr{T} = C_NULL
    redrhs::Ptr{T} = C_NULL
    rhs_sparse::Ptr{T} = C_NULL
    sol_loc::Ptr{T} = C_NULL
    rhs_loc::Ptr{T} = C_NULL
    rhsintr::Ptr{T} = C_NULL

    irhs_sparse::Ptr{Cint} = C_NULL
    irhs_ptr::Ptr{Cint} = C_NULL
    isol_loc::Ptr{Cint} = C_NULL
    irhs_loc::Ptr{Cint} = C_NULL

    glob2loc_rhs::Ptr{Cint} = C_NULL
    glob2loc_sol::Ptr{Cint} = C_NULL

    nrhs::Cint = 0
    lrhs::Cint = 0
    lredrhs::Cint = 0
    nz_rhs::Cint = 0
    lsol_loc::Cint = 0
    nloc_rhs::Cint = 0
    lrhs_loc::Cint = 0
    nsol_loc::Cint = 0

    schur_mloc::Cint = 0
    schur_nloc::Cint = 0
    schur_lld::Cint = 0

    mblock::Cint = 0
    nblock::Cint = 0
    nprow::Cint = 0
    npcol::Cint = 0

    ld_rhsintr::Cint = 0

    info::NTuple{80,Cint} = tzeros(80)
    infog::NTuple{80,Cint} = tzeros(80)
    rinfo::NTuple{40,T} = tzeros(40)
    rinfog::NTuple{40,T} = tzeros(40)

    pivnul_list::Ptr{Cint} = C_NULL
    mapping::Ptr{Cint} = C_NULL
    singular_values::Ptr{T} = C_NULL
    nb_singular_values::Cint = 0

    size_schur::Cint = 0
    listvar_schur::Ptr{Cint} = C_NULL
    schur::Ptr{T} = C_NULL ##

    wk_user::Ptr{T} = C_NULL

    version_number::NTuple{32,Cchar} = tzeros(32)

    ooc_tmpdir::NTuple{1024,Cchar} = tzeros(1024)
    ooc_prefix::NTuple{256,Cchar} = tzeros(256)

    write_problem::NTuple{1024,Cchar} = tzeros(1024)
    lwk_user::Cint = 0

    save_dir::NTuple{1024,Cchar} = tzeros(1024)
    save_prefix::NTuple{256,Cchar} = tzeros(256)

    metis_options::NTuple{40,Cint} = tzeros(40)
    instance_number::Cint = 0
end

mutable struct MumpsSolver{T} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,Int32}
    I::Vector{Int32}
    J::Vector{Int32}
    sym_perm::Vector{Int32}
    pivnul_list::Vector{Int32}
    mumps_struc::Struc
    is_singular::Bool
    opt::MumpsOptions
    logger::MadNLPLogger
end

# this is necessary, when multi-threaded calls are made with Mumps, not to clash with MPI
mumps_lock = Threads.SpinLock()

function locked_mumps_c(mumps_struc::Struc{Float32})
    lock(mumps_lock)
    try
        @ccall MUMPS_seq_jll.libsmumps.smumps_c(mumps_struc::Ref{Struc{Float32}})::Cvoid
    finally
        unlock(mumps_lock)
    end
end

function locked_mumps_c(mumps_struc::Struc{Float64})
    lock(mumps_lock)
    try
        @ccall MUMPS_seq_jll.libdmumps.dmumps_c(mumps_struc::Ref{Struc{Float64}})::Cvoid
    finally
        unlock(mumps_lock)
    end
end
# ---------------------------------------------------------------------------------------

function MumpsSolver(csc::SparseMatrixCSC{T,Int32};
    opt=MumpsOptions(), logger=MadNLPLogger(), pos=nothing,
) where T

    I,J = findIJ(csc)
    sym_perm = zeros(Int32,csc.n)
    pivnul_list = zeros(Int32,csc.n)

    mumps_struc = Struc{T}()

    mumps_struc.sym =  2
    mumps_struc.par =  1
    mumps_struc.job = -1
    mumps_struc.comm_fortran = -987654 # MPI.COMM_WORLD.val

    locked_mumps_c(mumps_struc)
    mumps_struc.n = csc.n;
    mumps_struc.nz= nnz(csc);
    mumps_struc.a = pointer(csc.nzval)
    mumps_struc.irn = pointer(I)
    mumps_struc.jcn = pointer(J)
    mumps_struc.sym_perm = pointer(sym_perm)
    mumps_struc.pivnul_list = pointer(pivnul_list)

    # symbolic factorization
    mumps_struc.job = 1;

    mumps_struc.icntl = setindex(mumps_struc.icntl,0,2)
    mumps_struc.icntl = setindex(mumps_struc.icntl,0,3)
    mumps_struc.icntl = setindex(mumps_struc.icntl,0,4)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_permuting_scaling,6)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_pivot_order,7)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_scaling,8)
    mumps_struc.icntl = setindex(mumps_struc.icntl,0,10)
    mumps_struc.icntl = setindex(mumps_struc.icntl,1,13)
    mumps_struc.icntl = setindex(mumps_struc.icntl,opt.mumps_mem_percent,14)


    mumps_struc.cntl = setindex(mumps_struc.cntl,opt.mumps_pivtol,1)

    locked_mumps_c(mumps_struc);
    mumps_struc.info[1] < 0 && throw(SymbolicException())

    M = MumpsSolver{T}(csc,I,J,sym_perm,pivnul_list,mumps_struc,false,opt,logger)
    finalizer(finalize,M)

    return M
end

function factorize!(M::MumpsSolver)
    M.is_singular = false
    M.mumps_struc.job = 2;
    cnt = 0
    while true
        locked_mumps_c(M.mumps_struc)
        if M.mumps_struc.info[1] in (-8,-9)
            cnt >= 10 && throw(FactorizationException())
            M.mumps_struc.icntl = setindex(M.mumps_struc.icntl,M.mumps_struc.icntl[14]*2.,14)
            cnt += 1
        elseif M.mumps_struc.info[1] == -10
            M.is_singular = true
            break
        elseif M.mumps_struc.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve_linear_system!(M::MumpsSolver{T},rhs::Vector{T}) where T
    M.is_singular && return rhs
    M.mumps_struc.rhs = pointer(rhs)
    M.mumps_struc.job = 3
    locked_mumps_c(M.mumps_struc)
    M.mumps_struc.info[1] < 0 && throw(SolveException())
    return rhs
end

is_inertia(::MumpsSolver) = true
function inertia(M::MumpsSolver)
    return (M.csc.n-M.is_singular-M.mumps_struc.infog[12], M.is_singular, M.mumps_struc.infog[12])
end


function improve!(M::MumpsSolver)
    if M.mumps_struc.cntl[1] == M.opt.mumps_pivtolmax
        @debug(M.logger,"improve quality failed.")
        return false
    end
    M.mumps_struc.cntl = setindex(M.mumps_struc.cntl,min(M.opt.mumps_pivtolmax,M.mumps_struc.cntl[1]^.5),1)
    @debug(M.logger,"improved quality: pivtol = $(M.mumps_struc.cntl[1])")
    return true
end

function finalize(M::MumpsSolver)
    M.mumps_struc.job = -2
    locked_mumps_c(M.mumps_struc);
end

introduce(M::MumpsSolver)="MUMPS v$(join(Char(c) for c in M.mumps_struc.version_number if c != 0))"
input_type(::Type{MumpsSolver}) = :csc
default_options(::Type{MumpsSolver}) = MumpsOptions()
is_supported(::Type{MumpsSolver},::Type{Float32}) = true
is_supported(::Type{MumpsSolver},::Type{Float64}) = true
