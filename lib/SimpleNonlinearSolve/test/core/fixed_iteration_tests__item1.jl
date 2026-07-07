using SimpleNonlinearSolve, StaticArrays
using NonlinearSolveBase, SciMLBase
using SciMLBase: ReturnCode

quadratic_f(u, p) = u .* u .- p
quadratic_f!(du, u, p) = (du .= u .* u .- p)
quadratic_df(u::Number, p) = 2u

@testset "NoTermination: exactly maxiters steps, Success retcode" begin
    p = 2.0
    for N in (1, 2, 3, 5)
        sol = solve(
            NonlinearProblem{false}(quadratic_f, 1.0, p), SimpleNewtonRaphson();
            maxiters = N, termination_condition = NonlinearSolveBase.NoTermination()
        )
        # Hand-rolled Newton for the same number of steps must match exactly.
        x = 1.0
        for _ in 1:N
            x = x - (x^2 - p) / (2x)
        end
        @test sol.u == x
        @test sol.retcode == ReturnCode.Success
    end

    # Default termination still reports MaxIters when genuinely not converged.
    sol = solve(
        NonlinearProblem{false}(quadratic_f, 1.0e10, p), SimpleNewtonRaphson();
        maxiters = 2
    )
    @test sol.retcode == ReturnCode.MaxIters
end

@testset "NoTermination with analytic derivative" begin
    p = 2.0
    fn = NonlinearFunction{false}(quadratic_f; jac = quadratic_df)
    sol = solve(
        NonlinearProblem(fn, 1.0, p), SimpleNewtonRaphson();
        maxiters = 6, termination_condition = NonlinearSolveBase.NoTermination()
    )
    @test sol.u ≈ sqrt(p) atol = 1e-12
    @test sol.retcode == ReturnCode.Success
end

@testset "fused value+jacobian: one f call per iteration (AD path)" begin
    p = 2.0
    calls = Ref(0)
    counted_f = (u, p) -> (calls[] += 1; u * u - p)

    N = 5
    sol = solve(
        NonlinearProblem{false}(counted_f, 1.0, p), SimpleNewtonRaphson();
        maxiters = N, termination_condition = NonlinearSolveBase.NoTermination()
    )
    @test sol.u ≈ sqrt(p)
    # init: one value eval + one derivative eval; loop: one fused eval per iteration.
    @test calls[] == 2 + N
end

@testset "fused path preserves convergence (default termination)" begin
    # scalar
    sol = solve(NonlinearProblem{false}(quadratic_f, 1.0, 2.0), SimpleNewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ sqrt(2.0)

    # SArray
    sol = solve(
        NonlinearProblem{false}(quadratic_f, SA[1.0, 1.0], 2.0), SimpleNewtonRaphson()
    )
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ SA[sqrt(2.0), sqrt(2.0)]

    # mutable vector, out-of-place
    sol = solve(NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0), SimpleNewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [sqrt(2.0), sqrt(2.0)]

    # in-place
    sol = solve(NonlinearProblem{true}(quadratic_f!, [1.0, 1.0], 2.0), SimpleNewtonRaphson())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [sqrt(2.0), sqrt(2.0)]
end
