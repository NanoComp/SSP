module SSPChainRulesCoreExt

import ChainRulesCore: rrule, NoTangent, unthunk
using SSP: conic_filter, ssp1_linear, ssp1, ssp2, Project, conic_filter_withsolver, ssp_withsolver, conic_filter_rrule, ssp_rrule
using SSP: constraint_solid, constraint_void, lengthconstraint_withsolver, lengthconstraint_rrule, Constrain

function rrule(::typeof(conic_filter), data, radius, grid)
    rho_filtered, solvers = conic_filter_withsolver(data, radius, grid)
    function conic_filter_pullback(adj_rho_filtered)
        adj_data = conic_filter_rrule(unthunk(adj_rho_filtered), solvers...)
        return NoTangent(), adj_data, NoTangent(), NoTangent()
    end
    return rho_filtered, conic_filter_pullback
end

function _ssp_rrule(alg, rho_filtered, beta, eta, grid)
    rho_projected, solver = ssp_withsolver(alg, rho_filtered, beta, eta, grid)
    function ssp_pullback(adj_rho_projected)
        adj_rho_filtered = ssp_rrule(unthunk(adj_rho_projected), solver)
        return NoTangent(), adj_rho_filtered, NoTangent(), NoTangent(), NoTangent()
    end
    return rho_projected, ssp_pullback
end
function rrule(::typeof(ssp1_linear), rho_filtered, beta, eta, grid; kws...)
    _ssp_rrule(Project.SSP1_linear(; kws...), rho_filtered, beta, eta, grid)
end
function rrule(::typeof(ssp1), rho_filtered, beta, eta, grid; kws...)
    _ssp_rrule(Project.SSP1(; kws...), rho_filtered, beta, eta, grid)
end
function rrule(::typeof(ssp2), rho_filtered, beta, eta, grid; kws...)
    _ssp_rrule(Project.SSP2(; kws...), rho_filtered, beta, eta, grid)
end

function _lengthconstraint_rrule(material, rho_filtered, rho_projected, grid, target_length)
    constraint, solver = lengthconstraint_withsolver(material, rho_filtered, rho_projected, grid, target_length)
    function lengthconstraint_pullback(adj_constraint)
        adj_prob = lengthconstraint_rrule(unthunk(adj_constraint), solver)
        return NoTangent(), adj_prob.rho_filtered, adj_prob.rho_projected, NoTangent(), NoTangent()
    end
    return constraint, lengthconstraint_pullback
end
function rrule(::typeof(constraint_solid), rho_filtered, rho_projected, grid, target_length)
    _lengthconstraint_rrule(Constrain.solid, rho_filtered, rho_projected, grid, target_length)
end
function rrule(::typeof(constraint_void), rho_filtered, rho_projected, grid, target_length)
    _lengthconstraint_rrule(Constrain.void, rho_filtered, rho_projected, grid, target_length)
end

end