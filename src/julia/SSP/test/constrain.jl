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

radius = 0.1 # need to pick a large radius so that features violate constraint
target_grid = (
    range(0.0, 0.1, length=Nx),
    range(0.0, 0.1, length=Ny),
)
target_points = vec(collect(Iterators.product(target_grid...)))

for material in (
    SSP.Constrain.solid,
    SSP.Constrain.void,
)

    # test for adj_prob.rho_filtered
    # test that adjoints match finite differences
    Random.seed!(0)
    perturb = randn(size(data))
    beta = Inf
    eta = 0.5
    rho_projected = SSP.Project.tanh_projection.(data, beta, eta)
    alg = SSP.Constrain.GeometricConstraints()
    test = let perturb=perturb, data=copy(data), alg=alg, grid=grid, target_points=target_points, material=material, radius=radius, rho_projected=rho_projected
        function (h)
            prob = SSP.Constrain.LengthConstraintProblem(;
                rho_filtered=data + h * perturb,
                grid,
                rho_projected,
                target_points,
                material,
                target_length=radius,
            )
            sol = SSP.solve(prob, alg)
            return sol.value
        end
    end
    dtest_di_fd = central_fdm(5, 1)(test, 0.0)

    prob = SSP.Constrain.LengthConstraintProblem(;
        rho_filtered=data,
        grid,
        rho_projected,
        target_points,
        material,
        target_length=radius,
    )
    solver = SSP.init(prob, alg)
    sol = SSP.solve!(solver)
    adj_sol = (; value=1.0)
    adj_prob = SSP.adjoint_solve!(solver, adj_sol, nothing)
    @test dtest_di_fd ≈ sum(adj_prob.rho_filtered .* perturb) atol=1e-8 rtol=1e-8


    # test for adj_prob.rho_projected
    # test that adjoints match finite differences
    Random.seed!(0)
    perturb = randn(size(data))
    beta = Inf
    eta = 0.5
    rho_projected = SSP.Project.tanh_projection.(data, beta, eta)
    alg = SSP.Constrain.GeometricConstraints()
    test = let perturb=perturb, data=copy(data), alg=alg, grid=grid, target_points=target_points, material=material, radius=radius, rho_projected=rho_projected
        function (h)
            prob = SSP.Constrain.LengthConstraintProblem(;
                rho_filtered=data,
                grid,
                rho_projected=rho_projected + h * perturb,
                target_points,
                material,
                target_length=radius,
            )
            sol = SSP.solve(prob, alg)
            return sol.value
        end
    end
    dtest_di_fd = central_fdm(5, 1)(test, 0.0)

    prob = SSP.Constrain.LengthConstraintProblem(;
        rho_filtered=data,
        grid,
        rho_projected,
        target_points,
        material,
        target_length=radius,
    )
    solver = SSP.init(prob, alg)
    sol = SSP.solve!(solver)
    adj_sol = (; value=1.0)
    adj_prob = SSP.adjoint_solve!(solver, adj_sol, nothing)
    @test dtest_di_fd ≈ sum(adj_prob.rho_projected .* perturb) atol=1e-8 rtol=1e-8

end
