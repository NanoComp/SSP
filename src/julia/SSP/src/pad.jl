module Pad

import SSP: init, solve!, adjoint_solve!

abstract type AbstractSizedPadding end

function size_lo end
function size_hi end

struct FillPadding{V,T} <: AbstractSizedPadding
    value::V
    lo::T
    hi::T
end
size_lo(bc::FillPadding) = bc.lo
size_hi(bc::FillPadding) = bc.hi

struct BoundaryPadding{T} <: AbstractSizedPadding
    lo::T
    hi::T
end
size_lo(bc::BoundaryPadding) = bc.lo
size_hi(bc::BoundaryPadding) = bc.hi

struct Inner{T} <: AbstractSizedPadding
    lo::T
    hi::T
end
size_lo(bc::Inner) = .-bc.lo
size_hi(bc::Inner) = .-bc.hi


Base.@kwdef struct PaddingProblem{D,B}
    data::D
    boundary::B
end

mutable struct PaddingSolver{D,B,A,C}
    data::D
    boundary::B
    alg::A
    cacheval::C
end

struct DefaultPaddingAlgorithm end

function init(prob::PaddingProblem, alg::DefaultPaddingAlgorithm)
    (; data, boundary) = prob
    cacheval = init_cacheval(alg, boundary, data)
    return PaddingSolver(data, boundary, alg, cacheval)
end

function init_cacheval(::DefaultPaddingAlgorithm, boundary, data)
    default_init_cacheval(boundary, data)
end
function default_init_cacheval(bc::AbstractSizedPadding, data)
    output = similar(data, size(data) .+ size_lo(bc) .+ size_hi(bc))
    adj_data = similar(data)
    return (; output, adj_data)
end
function solve!(solver::PaddingSolver)
    pad_solve!(solver, solver.alg)
end

function pad_solve!(solver, ::DefaultPaddingAlgorithm)
    default_pad_solve!(solver, solver.boundary, solver.data, solver.cacheval)
end

function default_pad_solve!(solver, bc::AbstractSizedPadding, data, cacheval)
    (; output) = cacheval
    _pad!(bc, output, data)
    return (; value=output, tape=nothing)
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
