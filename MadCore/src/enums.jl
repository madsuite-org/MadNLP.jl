# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

# Options
@enum(LogLevels::Int,
      TRACE  = 1,
      DEBUG  = 2,
      INFO   = 3,
      NOTICE = 4,
      WARN   = 5,
      ERROR  = 6)

@enum(BFGSInitStrategy::Int,
      SCALAR1  = 1,
      SCALAR2  = 2,
      SCALAR3  = 3,
      SCALAR4  = 4,
      CONSTANT = 5,
)

@enum(Status::Int,
      SOLVE_SUCCEEDED = 1,
      SOLVED_TO_ACCEPTABLE_LEVEL = 2,
      SEARCH_DIRECTION_BECOMES_TOO_SMALL = 3,
      DIVERGING_ITERATES = 4,
      INFEASIBLE_PROBLEM_DETECTED = 5,
      MAXIMUM_ITERATIONS_EXCEEDED = 6,
      MAXIMUM_WALLTIME_EXCEEDED = 7,
      INITIAL = 11,
      REGULAR = 12,
      RESTORE = 13,
      ROBUST  = 14,
      LINESEARCH_SUCCEEDED = 15,
      RESTORATION_FAILED = -1,
      INVALID_NUMBER_DETECTED = -2,
      ERROR_IN_STEP_COMPUTATION = -3,
      NOT_ENOUGH_DEGREES_OF_FREEDOM = -4,
      USER_REQUESTED_STOP = -5,
      INTERNAL_ERROR = -6,
      INVALID_NUMBER_OBJECTIVE = -7,
      INVALID_NUMBER_GRADIENT = -8,
      INVALID_NUMBER_CONSTRAINTS = -9,
      INVALID_NUMBER_JACOBIAN = -10,
      INVALID_NUMBER_HESSIAN_LAGRANGIAN = -11,
)

function get_status_output(status, opt)
    if status == SOLVE_SUCCEEDED
        return @sprintf "Optimal Solution Found (tol = %5.1e)." opt.tol
    elseif status == SOLVED_TO_ACCEPTABLE_LEVEL
        return @sprintf "Solved To Acceptable Level (tol = %5.1e)." opt.acceptable_tol
    elseif status == SEARCH_DIRECTION_BECOMES_TOO_SMALL
        return "Search Direction is becoming Too Small."
    elseif status == DIVERGING_ITERATES
        return "Iterates diverging; problem might be unbounded."
    elseif status == MAXIMUM_ITERATIONS_EXCEEDED
        return "Maximum Number of Iterations Exceeded."
    elseif status == MAXIMUM_WALLTIME_EXCEEDED
        return "Maximum wall-clock Time Exceeded."
    elseif status == RESTORATION_FAILED
        return "Restoration Failed"
    elseif status == INFEASIBLE_PROBLEM_DETECTED
        return "Converged to a point of local infeasibility. Problem may be infeasible."
    elseif status == INVALID_NUMBER_DETECTED
        return "Invalid number in NLP function or derivative detected."
    elseif status == ERROR_IN_STEP_COMPUTATION
        return "Error in step computation."
    elseif status == NOT_ENOUGH_DEGREES_OF_FREEDOM
        return "Problem has too few degrees of freedom."
    elseif status == USER_REQUESTED_STOP
        return "Stopping optimization at current point as requested by user."
    elseif status == INTERNAL_ERROR
        return "Internal Error."
    elseif status == INVALID_NUMBER_OBJECTIVE
        return "Invalid number in NLP objective function detected."
    elseif status == INVALID_NUMBER_GRADIENT
        return "Invalid number in NLP objective gradient detected."
    elseif status == INVALID_NUMBER_CONSTRAINTS
        return "Invalid number in NLP constraint function detected."
    elseif status == INVALID_NUMBER_JACOBIAN
        return "Invalid number in NLP constraint Jacobian detected."
    elseif status == INVALID_NUMBER_HESSIAN_LAGRANGIAN
        return "Invalid number in NLP Hessian Lagrangian detected."
    else
        error("status code is not valid")
    end
end
