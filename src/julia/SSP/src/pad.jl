module Pad

import SSP: init!, solve!, adjoint_solve!

public PaddingProblem, DefaultPaddingAlgorithm
public FillPadding, BoundaryPadding, Inner

"""
    AbstractSizedPadding

Supertype for flavors of array padding of a fixed size.
Concrete types must implement [`size_lo`](@ref) and [`size_hi`](@ref).
"""
abstract type AbstractSizedPadding end

"""
    size_lo(::AbstractSizedPadding)::Tuple{Vararg{Int}}

Return the size of the padding of the beginning of an array as a tuple of integers.
"""
function size_lo end

"""
    size_hi(::AbstractSizedPadding)::Tuple{Vararg{Int}}

Return the size of the padding of the end of an array as a tuple of integers.
"""
function size_hi end

"""
    FillPadding(value, lo, hi)

Pads an array with a given `value` with a padding size of `lo` at the beginning of the array and `hi` at the end of the array.
`lo` and `hi` must be of the same type as the `size` of the array.
"""
struct FillPadding{V,T} <: AbstractSizedPadding
    value::V
    lo::T
    hi::T
end
size_lo(bc::FillPadding) = bc.lo
size_hi(bc::FillPadding) = bc.hi

"""
    BoundaryPadding(lo, hi)

Pads an array by extending the values at the boundary to a padding size of `lo` at the beginning of the array and `hi` at the end of the array.
Equivalent to setting the value of the padding pixels equal to the value of the nearest pixel in the image.
`lo` and `hi` must be of the same type as the `size` of the array.
"""
struct BoundaryPadding{T} <: AbstractSizedPadding
    lo::T
    hi::T
end
size_lo(bc::BoundaryPadding) = bc.lo
size_hi(bc::BoundaryPadding) = bc.hi

"""
    Inner(lo, hi)

Removes an array's padding by truncating `lo` entries at the beginning of the array and `hi` entries at the end of the array.
`lo` and `hi` must be of the same type as the `size` of the array.
"""
struct Inner{T} <: AbstractSizedPadding
    lo::T
    hi::T
end
size_lo(bc::Inner) = .-bc.lo
size_hi(bc::Inner) = .-bc.hi

"""
    PaddingProblem(; data, boundary, grid=nothing)

Defines a problem of padding an array of `data` using a style of `boundary` padding.
Boundary padding styles include [`BoundaryPadding`](@ref), [`FillPadding`](@ref), and [`Inner`](@ref).
Optionally, if the data are defined on a `grid`, a tuple of ranges, then the `grid` is also extended to the size of the padded array.

The solution of a `PaddingProblem` contains a `value` field with the padded data, a `tape` field for the adjoint solve, and possibly a `grid` field with the padded grid.
"""
Base.@kwdef struct PaddingProblem{D,B,G}
    data::D
    boundary::B
    grid::G=nothing
end

function Base.copy(prob::PaddingProblem)
    newprob = PaddingProblem(;
        data = copy(prob.data),
        boundary = prob.boundary,
        grid = prob.grid,
    )
    return newprob
end

mutable struct PaddingSolver{D,B,G,A,C}
    data::D
    boundary::B
    grid::G
    alg::A
    cacheval::C
end

"""
    DefaultPaddingAlgorithm()

Default algorithm for padding arrays.
"""
struct DefaultPaddingAlgorithm end

function init!(prob::PaddingProblem, alg::DefaultPaddingAlgorithm)
    (; data, boundary, grid) = prob
    cacheval = init_cacheval(alg, boundary, data, grid)
    return PaddingSolver(data, boundary, grid, alg, cacheval)
end

function init_cacheval(::DefaultPaddingAlgorithm, boundary, data, grid)
    default_init_cacheval(boundary, data, grid)
end
function default_init_cacheval(bc::AbstractSizedPadding, data, grid)
    output = similar(data, size(data) .+ size_lo(bc) .+ size_hi(bc))
    adj_data = similar(data)
    return (; output, adj_data)
end
function solve!(solver::PaddingSolver)
    pad_solve!(solver, solver.alg)
end

function pad_solve!(solver, ::DefaultPaddingAlgorithm)
    default_pad_solve!(solver, solver.boundary, solver.data, solver.grid, solver.cacheval)
end

function default_pad_solve!(solver, bc::AbstractSizedPadding, data, grid, cacheval)
    (; output) = cacheval
    _pad!(bc, output, data)
    padded_grid = _padgrid(bc, grid)
    return (; value=output, padded_grid..., tape=nothing)
end

function _padgrid(::AbstractSizedPadding, ::Nothing)
    (;)
end
function _padgrid(bc::AbstractSizedPadding, grid)
    l = size_lo(bc)
    h = size_hi(bc)
    padded_grid = map(grid, l, h) do grid, l, h
        range(first(grid)-l*step(grid); step=step(grid), length=length(grid)+l+h)
        # range(first(grid)-l*step(grid), last(grid)+h*step(grid); length=length(grid)+l+h)
    end
    (; grid=padded_grid)
end


function _pad!(bc::BoundaryPadding, y, x)
    pad_lo = size_lo(bc)
    M = size(x)
    N = size(y)
    # edge padding by accessing nearest element in the image x
    for i in CartesianIndices(map(n -> 1:n, N))
        y[i] = x[map((j, m, p) -> max(1, min(m, j-p)), Tuple(i), M, pad_lo)...]
    end
    return y
end

function _pad!(bc::FillPadding, y, x)
    pad_lo = size_lo(bc)
    M = size(x)
    N = size(y)
    for i in CartesianIndices(map(n -> 1:n, N))
        val = if all(map((j, m, p) -> 0 < j-p < m+1, Tuple(i), M, pad_lo))
            x[map((j, p) -> j-p, Tuple(i), pad_lo)...]
        else
            bc.value
        end
        y[i] = val
    end
    return y
end

function _pad!(bc::Inner, y, x)
    pad_lo = size_lo(bc)
    pad_hi = size_hi(bc)
    copy!(y, view(x, map((n, l, h) -> firstindex(x, n) - l:lastindex(x, n) + h, ntuple(identity, ndims(x)), pad_lo, pad_hi)...))
    return y
end


function adjoint_solve!(solver::PaddingSolver, adj_output, tape)
    adjoint_pad_solve!(solver, solver.alg, adj_output, tape)
end

function adjoint_pad_solve!(solver, ::DefaultPaddingAlgorithm, adj_output, tape)
    return adjoint_default_pad_solve!(solver, solver.boundary, solver.data, solver.cacheval, adj_output, tape)
end

function adjoint_default_pad_solve!(solver, bc::AbstractSizedPadding, data, cacheval, adj_output, tape)
    (; adj_data, output) = cacheval
    @assert size(output) == size(adj_output)
    _adjoint_pad!(bc, adj_data, adj_output)
    return (; data=adj_data, boundary=nothing)
end

function _adjoint_pad!(bc::BoundaryPadding, x, y)
    pad_lo = size_lo(bc)
    M = size(x)
    N = size(y)
    # edge padding by accessing nearest element in the image x
    fill!(x, zero(eltype(x)))
    for i in CartesianIndices(map(n -> 1:n, N))
        x[map((j, m, p) -> max(1, min(m, j-p)), Tuple(i), M, pad_lo)...] += y[i]
    end
    return x
end

function _adjoint_pad!(bc::FillPadding, x, y)
    pad_lo = size_lo(bc)
    M = size(x)
    for i in CartesianIndices(map(n -> 1:n, M))
        x[i] = y[map((j, p) -> j+p, Tuple(i), pad_lo)...]
    end
    return x
end

function _adjoint_pad!(bc::Inner, x, y)
    pad_lo = size_lo(bc)
    pad_hi = size_lo(bc)
    fill!(x, zero(eltype(x)))
    copy!(view(x, map((n, l, h) -> firstindex(x, n) - l:lastindex(x, n) + h, ntuple(identity, ndims(x)), pad_lo, pad_hi)...), y)
    return x
end

end
