using Test
using FiniteDifferences
using SSP
using Random

Nx = Ny = 10
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)
data = [sinpi(x) * sinpi(y) for (x, y) in Iterators.product(grid...)]

target_grid = (
    range(0.0, 0.1, length=Nx),
    range(0.0, 0.1, length=Ny),
)
target_points = vec(collect(Iterators.product(target_grid...)))

for alg in (
    SSP.Project.SSP1_linear(),
    SSP.Project.SSP1(),
    SSP.Project.SSP2(),
)

    # test that adjoints match finite differences
    Random.seed!(0)
    perturb = randn(size(data))
    test = let perturb=perturb, data=copy(data), alg=alg, grid=grid, target_points=target_points
        function (h)
            prob = SSP.Project.ProjectionProblem(;
                rho_filtered=data + h * perturb,
                grid,
                target_points,
                beta = Inf,
                eta = 0.5
            )
            sol = SSP.solve(prob, alg)
            return sum(abs2, sol.value)
        end
    end
    dtest_di_fd = central_fdm(5, 1)(test, 0.0)

    prob = SSP.Project.ProjectionProblem(;
        rho_filtered=data,
        grid,
        target_points,
        beta = Inf,
        eta = 0.5
    )
    solver = SSP.init(prob, alg)
    sol = SSP.solve!(solver)
    adj_sol = (; value=2*sol.value)
    adj_prob = SSP.adjoint_solve!(solver, adj_sol, nothing)
    @test dtest_di_fd ≈ sum(adj_prob.rho_filtered .* perturb)
end