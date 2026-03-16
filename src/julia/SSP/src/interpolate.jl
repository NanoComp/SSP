module Interpolate

using FastInterpolations: cubic_interp!, cubic_adjoint, CubicFit, DerivOp
import SSP: init, solve!, adjoint_solve!

Base.@kwdef struct InterpolationProblem{D,G,T}
    data::D
    grid::G
    target_points::T
end

mutable struct InterpolationSolver{D,G,T,A,C}
    data::D
    const grid::G
    const target_points::T
    alg::A
    cacheval::C
end

struct Value end
struct ValueWithGradient end
struct ValueWithGradientAndHessian end

Base.@kwdef struct CubicInterp{B,D}
    bc::B=CubicFit()
    deriv::D=Value()
end

function init(prob::InterpolationProblem, alg::CubicInterp)
    (; data, grid, target_points) = prob

    # TODO refactor these workspaces as views of a single array
    cacheval_value = if alg.deriv isa Value || alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        interp_value = similar(data, length(target_points))
        adj_data = similar(data)
        adj_data_tmp = similar(data) # this is only used by higher derivatives
        adj_op = cubic_adjoint(grid, target_points)
        (; interp_value, adj_data, adj_data_tmp, adj_op)
    else
        (;)
    end

    ndim = Val{ndims(data)}()
    dims = ntuple(n -> Val{n}(), ndim) # construct a type-stable tuple / range because this gets unrolled

    cacheval_gradient = if alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        interp_gradient = similar(data, length(target_points), length(grid))
        initgradadjop = function (::Val{dim}) where {dim}
            # construct deriv tuple in a type-stable way by using a function barrier
            deriv = ntuple(n -> DerivOp{n == dim ? 1 : 0}(), ndim)
            cubic_adjoint(grid, target_points; deriv)
        end
        adj_op_g = map(initgradadjop, dims)
        (; interp_gradient, adj_op_g)
    else
        (;)
    end

    cacheval_hessian = if alg.deriv isa ValueWithGradientAndHessian
        interp_hessian = similar(data, length(target_points), length(grid), length(grid))
        inithessadjop = function (::Val{dim1}, ::Val{dim2}) where {dim1, dim2}
            deriv = ntuple(n -> DerivOp{(n == dim1 ? 1 : 0) + (n == dim2 ? 1 : 0)}(), ndim)
            cubic_adjoint(grid, target_points; deriv)
        end
        adj_op_h = map(dim1 -> map(dim2 -> inithessadjop(dim1, dim2), dims), dims)
        (; interp_hessian, adj_op_h)
    else
        (;)
    end

    cacheval = (; cacheval_value..., cacheval_gradient..., cacheval_hessian...)
    return InterpolationSolver(data, grid, target_points, alg, cacheval)
end

function solve!(solver::InterpolationSolver)
    interp_solve!(solver, solver.alg)
end

function interp_solve!(solver, alg::CubicInterp)
    (; data, grid, target_points, cacheval) = solver
    
    sol_value = if alg.deriv isa Value || alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        let (; interp_value) = cacheval
            cubic_interp!(interp_value, grid, data, target_points)
            (; value=interp_value)
        end
    else
        (;)
    end

    ndim = Val{ndims(data)}()
    dims = ntuple(n -> Val{n}(), ndim) # construct a type-stable tuple / range because this gets unrolled

    sol_gradient = if alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        let (; interp_gradient) = cacheval
            dograd = function (::Val{dim}) where {dim}
                # construct deriv tuple in a type-stable way by using a function barrier
                deriv = ntuple(n -> DerivOp{n == dim ? 1 : 0}(), ndim)
                out = view(interp_gradient, :, dim)
                cubic_interp!(out, grid, data, target_points; deriv)
            end
            foreach(dograd, dims)
            (; gradient=interp_gradient)
        end
    else
        (;)
    end

    sol_hessian = if alg.deriv isa ValueWithGradientAndHessian
        let (; interp_hessian) = cacheval
            dohess = function (::Val{dim1}, ::Val{dim2}) where {dim1, dim2}
                deriv = ntuple(n -> DerivOp{(n == dim1 ? 1 : 0) + (n == dim2 ? 1 : 0)}(), ndim)
                out = view(interp_hessian, :, dim1, dim2)
                cubic_interp!(out, grid, data, target_points; deriv)
            end
            foreach(dim1 -> foreach(dim2 -> dohess(dim1, dim2), dims), dims)
            (; hessian=interp_hessian)
        end
    else
        (;)
    end

    # no tape because intermediate results are not needed for the adjoint
    sol = (; tape=nothing, sol_value..., sol_gradient..., sol_hessian...)
    return sol
end

function adjoint_solve!(solver::InterpolationSolver, adj_sol, tape)
    adjoint_interp_solve!(solver, solver.alg, adj_sol, tape)
end

function adjoint_interp_solve!(solver, alg::CubicInterp, adj_sol, tape)
    (; data, grid, target_points, cacheval) = solver

    adj_prob_value = if alg.deriv isa Value || alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        # create a local scope for cache variables so that they don't get boxed
        let (; adj_data, adj_data_tmp, adj_op) = cacheval
            adj_op(adj_data, adj_sol.value)
            (; data=adj_data)
        end
    else
        (;)
    end

    ndim = Val{ndims(data)}()
    dims = ntuple(identity, ndim) # we do not need type-stable indexing into dimensions for views

    adj_prob_gradient = if alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        let (; adj_data, adj_data_tmp, adj_op_g) = cacheval
            dograd = function (dim, adj_op_grad)
                adj_grad = view(adj_sol.gradient, :, dim)
                adj_op_grad(adj_data_tmp, adj_grad)
                adj_data .+= adj_data_tmp
            end
            foreach(dograd, dims, adj_op_g)
            (;) # data adjoint is already accumulated into adj_prob_value.data
        end
    else
        (;)
    end

    adj_prob_hessian = if alg.deriv isa ValueWithGradientAndHessian
        let (; adj_data, adj_data_tmp, adj_op_h) = cacheval
            dohess = function (dim1, dim2, adj_op_hess)
                adj_hess = view(adj_sol.hessian, :, dim1, dim2)
                adj_op_hess(adj_data_tmp, adj_hess)
                adj_data .+= adj_data_tmp
            end
            foreach((dim1, adj_op_h1) -> foreach((dim2, adj_op_h12) -> dohess(dim1, dim2, adj_op_h12), dims, adj_op_h1), dims, adj_op_h)
            (;) # data adjoint is already accumulated into adj_prob_value.data
        end
    else
        (;)
    end

    adj_prob = (; data=nothing, grid=nothing, target_points=nothing, adj_prob_value..., adj_prob_gradient..., adj_prob_hessian...)
    return adj_prob
end

end