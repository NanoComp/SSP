using Test
using ImageFiltering
using FiniteDifferences
using SSP
using Zygote


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
perturb_index = (3, 7)
test = let radius=radius, grid=grid
    function (data)
        rho_filtered = SSP.conic_filter(data, radius, grid)
        return sum(abs2, rho_filtered)
    end
end
test_i = let perturb_index=perturb_index, data=copy(data), test=test
    function (x)
        data[perturb_index...] = x
        return test(data)
    end
end
xi = 1.0
dtest_di_fd = central_fdm(5, 1)(test_i, xi)
ddata, = Zygote.gradient(test, let data=copy(data); data[perturb_index...] = xi; data; end)
@test dtest_di_fd ≈ ddata[perturb_index...]

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
    test_ssp_i = let perturb_index=perturb_index, data=copy(myfilt), test_ssp=test_ssp
        function (x)
            data[perturb_index...] = x
            return test_ssp(data)
        end
    end

    dtest_ssp_di_fd = central_fdm(5, 1)(test_ssp_i, xi)
    ddata_ssp, = Zygote.gradient(test_ssp, let data=copy(myfilt); data[perturb_index...] = xi; data; end)
    @test dtest_ssp_di_fd ≈ ddata_ssp[perturb_index...]
end