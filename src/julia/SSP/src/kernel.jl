module Kernel

public conickernel

"""
    conickernel(grids, radius)

Returns an array with values of a conic filtering kernel with a support of `2radius` with the same resolution as `grids`.
`grids` must be a tuple of ranges with the same number of ranges as dimensions of the kernel.
Additionally, the kernel is normalized to sum to unity.
"""
function conickernel(grids, radius)
    diam = 2radius
    convolving_grids = map(grids) do grid
        n = 1 + trunc(Int, diam/step(grid))
        n = iseven(n) ? n+1 : n
        (step(grid)*(n-1)/2) * range(-1, 1, length=n)
    end
    kernel = [ max(0.0, radius - sqrt(sum(abs2, x))) for x in Iterators.product(convolving_grids...)]
    kernel ./= sum(kernel) # normalize
    return kernel
end

end