using Test
using FiniteDifferences
using ImageFiltering: padarray, Pad, Fill
using SSP
using Random

Nx = Ny = 10
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)
data = [sinpi(x) * sinpi(y) for (x, y) in Iterators.product(grid...)]
padsize = 2

for (refborder, sspborder) in [
    (Pad(:replicate, padsize, padsize), SSP.Pad.BoundaryPadding((padsize, padsize), (padsize, padsize))),
    (Fill(1.3, (padsize, padsize)), SSP.Pad.FillPadding(1.3, (padsize, padsize), (padsize, padsize))),
    # (Inner(-padsize, -padsize), SSP.Pad.Inner((padsize, padsize), (padsize, padsize))),
]
    # test that output agrees with ImageFiltering
    pad_ref = padarray(data, refborder)

    prob = SSP.Pad.PaddingProblem(;
        data,
        boundary = sspborder,
    )
    alg = SSP.Pad.DefaultPaddingAlgorithm()
    sol = SSP.solve(prob, alg)

    @test collect(pad_ref) == sol.value

    # test that adjoints match finite differences
    Random.seed!(0)
    perturb = randn(size(data))
    test = let perturb=perturb, data=copy(data)
        function (h)
            prob = SSP.Pad.PaddingProblem(;
                data = data + h * perturb,
                boundary = sspborder,
            )
            alg = SSP.Pad.DefaultPaddingAlgorithm()
            sol = SSP.solve(prob, alg)
            return sum(abs2, sol.value)
        end
    end
    dtest_di_fd = central_fdm(5, 1)(test, 0.0)

    prob = SSP.Pad.PaddingProblem(;
        data,
        boundary = sspborder,
    )
    alg = SSP.Pad.DefaultPaddingAlgorithm()
    solver = SSP.init(prob, alg)
    sol = SSP.solve!(solver)
    adj_sol = (; value=2*sol.value)
    adj_prob = SSP.adjoint_solve!(solver, adj_sol, nothing)
    @test dtest_di_fd ≈ sum(adj_prob.data .* perturb)
end