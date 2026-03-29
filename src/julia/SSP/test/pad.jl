using Test
using FiniteDifferences
using ImageFiltering: padarray, Pad, Fill
using SSP


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
    perturb_index = (3, 7)
    test = let perturb_index=perturb_index, data=copy(data)
        function (x)
            data[perturb_index...] = x
            prob = SSP.Pad.PaddingProblem(;
                data,
                boundary = sspborder,
            )
            alg = SSP.Pad.DefaultPaddingAlgorithm()
            sol = SSP.solve(prob, alg)
            return sum(abs2, sol.value)
        end
    end
    xi = 1.0
    dtest_di_fd = central_fdm(5, 1)(test, xi)

    prob = SSP.Pad.PaddingProblem(;
        data,
        boundary = sspborder,
    )
    alg = SSP.Pad.DefaultPaddingAlgorithm()
    solver = SSP.init(prob, alg)
    solver.data[perturb_index...] = xi
    sol = SSP.solve!(solver)
    adj_prob = SSP.adjoint_solve!(solver, 2*sol.value, nothing)
    @test dtest_di_fd ≈ adj_prob.data[perturb_index...]
end