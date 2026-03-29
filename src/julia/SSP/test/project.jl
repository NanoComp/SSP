using Test
using FiniteDifferences
using SSP


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
    perturb_index = (3, 7)
    test = let perturb_index=perturb_index, data=copy(data), alg=alg, grid=grid, target_points=target_points
        function (x)
            data[perturb_index...] = x
            prob = SSP.Project.ProjectionProblem(;
                rho_filtered=data,
                grid,
                target_points,
                beta = Inf,
                eta = 0.5
            )
            sol = SSP.solve(prob, alg)
            return sum(abs2, sol.value)
        end
    end
    xi = 1.0
    dtest_di_fd = central_fdm(5, 1)(test, xi)

    prob = SSP.Project.ProjectionProblem(;
        rho_filtered=data,
        grid,
        target_points,
        beta = Inf,
        eta = 0.5
    )
    solver = SSP.init(prob, alg)
    solver.rho_filtered[perturb_index...] = xi
    sol = SSP.solve!(solver)
    adj_prob = SSP.adjoint_solve!(solver, 2*sol.value, nothing)
    @test dtest_di_fd ≈ adj_prob.rho_filtered[perturb_index...]
end