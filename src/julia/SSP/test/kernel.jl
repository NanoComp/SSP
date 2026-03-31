using Test
using SSP

Nx = Ny = 10
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)
radius = 0.5
kernel = SSP.Kernel.conickernel(grid, radius)

@test sum(kernel) ≈ 1