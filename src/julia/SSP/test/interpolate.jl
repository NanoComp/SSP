using Test
using FiniteDifferences
using Interpolations
using FastInterpolations: ZeroCurvBC, ZeroSlopeBC
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

ref_interp = cubic_spline_interpolation(grid, data; bc=Line(OnCell()))
ref_values = [ref_interp(x...) for x in target_points]
ref_gradient = [Interpolations.gradient(ref_interp, x...) for x in target_points]
ref_hessian = [Interpolations.hessian(ref_interp, x...) for x in target_points]

prob = SSP.Interpolate.InterpolationProblem(;
    data,
    grid,
    target_points,
)
alg = SSP.Interpolate.CubicInterp(; bc=ZeroCurvBC(), deriv=SSP.Interpolate.ValueWithGradientAndHessian())
sol = SSP.solve(prob, alg)

# I can't figure out how to match boundary conditions across the two packages so
# the interpolants don't agree exactly and I have to use a large atol for comparison
@test ref_values ≈ sol.value atol=1e-3
@test stack(ref_gradient; dims=1) ≈ sol.gradient atol=1e-3
@test stack(ref_hessian; dims=1) ≈ sol.hessian atol=1e-1


# test that adjoints match finite differences
perturb_index = (3, 7)
# interpolation value
test = let perturb_index=perturb_index, data=copy(data), target_points=target_points, grid=grid
    function (x)
        data[perturb_index...] = x
        prob = SSP.Interpolate.InterpolationProblem(;
            data,
            grid,
            target_points,
        )
        alg = SSP.Interpolate.CubicInterp(; deriv=SSP.Interpolate.Value())
        sol = SSP.solve(prob, alg)
        return sum(abs2, sol.value)
    end
end
xi = 1.0
dtest_di_fd = central_fdm(5, 1)(test, xi)

prob = SSP.Interpolate.InterpolationProblem(;
    data,
    grid,
    target_points,
)
alg = SSP.Interpolate.CubicInterp(; deriv=SSP.Interpolate.Value())
solver = SSP.init(prob, alg)
solver.data[perturb_index...] = xi
sol = SSP.solve!(solver)
adj_prob = SSP.adjoint_solve!(solver, (; value=2*sol.value), nothing)
@test dtest_di_fd ≈ adj_prob.data[perturb_index...]

# interpolation gradienttest = let perturb_index=perturb_index, data=copy(data), target_points=target_points
test = let perturb_index=perturb_index, data=copy(data), target_points=target_points, grid=grid
    function (x)
        data[perturb_index...] = x
        prob = SSP.Interpolate.InterpolationProblem(;
            data,
            grid,
            target_points,
        )
        alg = SSP.Interpolate.CubicInterp(; deriv=SSP.Interpolate.ValueWithGradient())
        sol = SSP.solve(prob, alg)
        return sum(abs2, sol.gradient)
    end
end
xi = 1.0
dtest_di_fd = central_fdm(5, 1)(test, xi)

prob = SSP.Interpolate.InterpolationProblem(;
    data,
    grid,
    target_points,
)
alg = SSP.Interpolate.CubicInterp(; deriv=SSP.Interpolate.ValueWithGradient())
solver = SSP.init(prob, alg)
solver.data[perturb_index...] = xi
sol = SSP.solve!(solver)
adj_prob = SSP.adjoint_solve!(solver, (; value=0*sol.value, gradient=2*sol.gradient), nothing)
@test dtest_di_fd ≈ adj_prob.data[perturb_index...]

# interpolation hessian
test = let perturb_index=perturb_index, data=copy(data), target_points=target_points, grid=grid
    function (x)
        data[perturb_index...] = x
        prob = SSP.Interpolate.InterpolationProblem(;
            data,
            grid,
            target_points,
        )
        alg = SSP.Interpolate.CubicInterp(; deriv=SSP.Interpolate.ValueWithGradientAndHessian())
        sol = SSP.solve(prob, alg)
        return sum(abs2, sol.hessian)
    end
end
xi = 1.0
dtest_di_fd = central_fdm(5, 1)(test, xi)

prob = SSP.Interpolate.InterpolationProblem(;
    data,
    grid,
    target_points,
)
alg = SSP.Interpolate.CubicInterp(; deriv=SSP.Interpolate.ValueWithGradientAndHessian())
solver = SSP.init(prob, alg)
solver.data[perturb_index...] = xi
sol = SSP.solve!(solver)
adj_prob = SSP.adjoint_solve!(solver, (; value=0*sol.value, gradient=0*sol.gradient, hessian=2*sol.hessian), nothing)
@test dtest_di_fd ≈ adj_prob.data[perturb_index...]
