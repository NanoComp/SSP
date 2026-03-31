using Test
using FiniteDifferences
using DSP
using SSP
using Random

Nx = Ny = 10
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)
data = [sinpi(x) * sinpi(y) for (x, y) in Iterators.product(grid...)]
radius = 0.5
kernel = SSP.Kernel.conickernel(grid, radius)

refconv = conv(data, kernel)

prob = SSP.Convolve.DiscreteConvolutionProblem(;
    data,
    kernel,
)
alg = SSP.Convolve.FFTConvolution()
sol = SSP.solve(prob, alg)

@test refconv[map((s, idx) -> s .+ idx, size(kernel) .÷ 2, axes(data))...] ≈ sol.value

# test that adjoints match finite differences
Random.seed!(0)
perturb = randn(size(data))
test = let perturb=perturb, data=data
    function (h)
        prob = SSP.Convolve.DiscreteConvolutionProblem(;
            data = data + h * perturb,
            kernel,
        )
        alg = SSP.Convolve.FFTConvolution()
        sol = SSP.solve(prob, alg)
        return sum(abs2, sol.value)
    end
end
dtest_di_fd = central_fdm(5, 1)(test, 0.0)

prob = SSP.Convolve.DiscreteConvolutionProblem(;
    data,
    kernel,
)
alg = SSP.Convolve.FFTConvolution()
solver = SSP.init(prob, alg)
sol = SSP.solve!(solver)
adj_sol = (; value=2*sol.value)
adj_prob = SSP.adjoint_solve!(solver, adj_sol, nothing)
@test dtest_di_fd ≈ sum(adj_prob.data .* perturb)