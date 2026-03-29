using Test
using FiniteDifferences
using DSP
using SSP

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
perturb_index = (3, 7)
test = let perturb_index=perturb_index, data=copy(data)
    function (x)
        data[perturb_index...] = x
        prob = SSP.Convolve.DiscreteConvolutionProblem(;
            data,
            kernel,
        )
        alg = SSP.Convolve.FFTConvolution()
        sol = SSP.solve(prob, alg)
        return sum(abs2, sol.value)
    end
end
xi = 1.0
dtest_di_fd = central_fdm(5, 1)(test, xi)

prob = SSP.Convolve.DiscreteConvolutionProblem(;
    data,
    kernel,
)
alg = SSP.Convolve.FFTConvolution()
solver = SSP.init(prob, alg)
solver.data[perturb_index...] = xi
sol = SSP.solve!(solver)
adj_prob = SSP.adjoint_solve!(solver, 2*sol.value, nothing)
@test dtest_di_fd ≈ adj_prob.data[perturb_index...]