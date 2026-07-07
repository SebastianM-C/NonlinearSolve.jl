"""
    SimpleNewtonRaphson(autodiff)
    SimpleNewtonRaphson(; autodiff = nothing)

A low-overhead implementation of Newton-Raphson. This method is non-allocating on scalar
and static array problems.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to  `nothing` (i.e.
    automatic backend selection). Valid choices include jacobian backends from
    `DifferentiationInterface.jl`.

!!! tip "Fixed-iteration (GPU kernel) mode"

    `solve(prob, SimpleNewtonRaphson(); maxiters = N, termination_condition =
    NonlinearSolveBase.NoTermination())` runs exactly `N` Newton steps with no
    per-iteration convergence branch (the check dispatches to a constant `false` and is
    dead-code-eliminated) and returns the final iterate with `ReturnCode.Success`.
    Combined with the fused residual+derivative evaluation (one `f` call per iteration
    on the AD path), this makes warm-started scalar solves inside GPU kernels
    warp-lockstep and evaluation-minimal.
"""
@kwdef @concrete struct SimpleNewtonRaphson <: AbstractSimpleNonlinearSolveAlgorithm
    autodiff = nothing
end

const SimpleGaussNewton = SimpleNewtonRaphson

function configure_autodiff(prob, alg::SimpleNewtonRaphson)
    autodiff = something(alg.autodiff, AutoForwardDiff())
    autodiff = SciMLBase.has_jac(prob.f) ? autodiff :
        NonlinearSolveBase.select_jacobian_autodiff(prob, autodiff)
    @set! alg.autodiff = autodiff
    return alg
end

function SciMLBase.__solve(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        alg::SimpleNewtonRaphson, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias::Union{Nothing, SciMLBase.NonlinearAliasSpecifier} = nothing,
        alias_u0 = false,
        termination_condition = nothing, kwargs...
    )
    # Extract alias_u0: if alias struct provided, use it; otherwise use alias_u0 kwarg
    _alias_u0 = alias === nothing ? alias_u0 : Utils.get_alias_u0(alias, alias_u0)
    autodiff = alg.autodiff
    x = NLBUtils.maybe_unaliased(prob.u0, _alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    @bb xo = similar(x)
    fx_cache = Utils.should_cache_fx(prob, prob.f) ?
        NLBUtils.safe_similar(fx) : fx
    jac_cache = Utils.prepare_jacobian(prob, autodiff, fx_cache, x)
    J = Utils.compute_jacobian!!(nothing, prob, autodiff, fx_cache, x, jac_cache)

    for _ in 1:maxiters
        @bb copyto!(xo, x)
        δx = NLBUtils.restructure(x, J \ NLBUtils.safe_vec(fx))
        @bb x .-= δx

        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        # Fused residual+jacobian: one `f` evaluation per iteration on the DI
        # paths instead of separate value and derivative calls.
        fx, J = Utils.compute_fx_jac!!(J, fx, prob, autodiff, fx_cache, x, jac_cache)
    end

    # Under `NoTermination` running out the iteration budget is the expected
    # outcome (fixed-iteration mode), not a failure.
    retcode = tc_cache.mode isa NonlinearSolveBase.NoTermination ?
        ReturnCode.Success : ReturnCode.MaxIters
    return SciMLBase.build_solution(prob, alg, x, fx; retcode)
end
