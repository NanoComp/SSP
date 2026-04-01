using Test
using ImageFiltering
using FiniteDifferences
using SSP
using Zygote
using Random

Nx = Ny = 10
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)
data = [sinpi(x) * sinpi(y) for (x, y) in Iterators.product(grid...)]
radius = 0.5
kernel = SSP.Kernel.conickernel(grid, radius)

reffilt = imfilter(data, centered(kernel), "replicate")
myfilt = SSP.conic_filter(data, radius, grid)

@test reffilt ≈ myfilt

# test that adjoints match finite differences for filtering
Random.seed!(0)
perturb = randn(size(data))
test = let radius=radius, grid=grid
    function (data)
        rho_filtered = SSP.conic_filter(data, radius, grid)
        return sum(abs2, rho_filtered)
    end
end
test_i = let perturb=perturb, data=copy(data), test=test
    h -> test(data + h * perturb)
end
dtest_di_fd = central_fdm(5, 1)(test_i, 0.0)
ddata, = Zygote.gradient(test, data)
@test dtest_di_fd ≈ sum(ddata .* perturb)

# test that adjoints match finite differences for projection
ssp_algs = (
    SSP.ssp1_linear,
    SSP.ssp1,
    SSP.ssp2,
)
beta = Inf
eta = 0.5
for ssp in ssp_algs
    test_ssp = let radius=radius, grid=grid, ssp=ssp, beta=beta, eta=eta
        function (data)
            rho_projected = ssp(data, beta, eta, grid)
            return sum(abs2, rho_projected)
        end
    end
    test_ssp_i = let perturb=perturb, data=copy(myfilt), test_ssp=test_ssp
        h -> test_ssp(data + h * perturb)
    end

    dtest_ssp_di_fd = central_fdm(5, 1)(test_ssp_i, 0.0)
    ddata_ssp, = Zygote.gradient(test_ssp, myfilt)
    @test dtest_ssp_di_fd ≈ sum(ddata_ssp .* perturb)
end

constraint_algs = (
    SSP.constraint_solid,
    SSP.constraint_void,
)
for constraint in constraint_algs
    test_constraint = let radius=radius, grid=grid, beta=beta, eta=eta
        function (data)
            rho_filtered = SSP.conic_filter(data, radius, grid)
            rho_projected = SSP.ssp2(data, beta, eta, grid)
            return constraint(rho_filtered, rho_projected, grid, radius)
        end
    end
    test_constraint_i = let perturb=perturb, data=data, test_constraint=test_constraint
        h -> test_constraint(data + h * perturb)
    end

    dtest_constraint_di_fd = central_fdm(5, 1)(test_constraint_i, 0.0)
    ddata_constraint, = Zygote.gradient(test_constraint, data)
    @test dtest_constraint_di_fd ≈ sum(ddata_constraint .* perturb) rtol=1e-5
end