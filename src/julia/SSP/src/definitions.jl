# Define a CommonSolve-like interface for adjoint problems

"""
    init!(prob, alg)

Return a `solver` object that can find a solution to the given problem `prob` using algorithm `alg` by calling `solve!(solver)`.
This in-place variant keeps references to the data in `prob`.
"""
function init! end

"""
    init(prob, alg)

Return a `solver` object that can find a solution to the given problem `prob` using algorithm `alg` by calling `solve!(solver)`.
This out-of-place variant makes copies of the data in `prob`.
"""
function init(prob, alg)
    newprob = copy(prob)
    solver = init!(newprob, alg)
    return solver
end

"""
    solve!(solver)

Return the solution using a `solver` to a problem defined using `init`.
The solution should contain a tape for `adjoint_solve!`.
"""
function solve! end

"""
    solve(prob, alg)

Return the solution to a problem `prob` using algorithm `alg`.
The solution should contain a tape for `adjoint_solve`.
"""
function solve(prob, alg)
    solver = init(prob, alg)
    sol = solve!(solver)
    return sol
end

"""
    adjoint_solve!(solver, adj_sol, tape)

Apply the pullback of `solve!` to the adjoint of the solution `adj_sol`, using the data stored in `tape` from the forward solve.
"""
function adjoint_solve! end

"""
    adjoint_solve(prob, alg, adj_sol, tape)

Apply the pullback of `solve` to the adjoint of the solution `adj_sol`, using the data stored in `tape` from the forward solve.
"""
function adjoint_solve(prob, alg, adj_sol, tape)
    solver = init(prob, alg)
    adj_prob = adjoint_solve!(solver, adj_sol, tape)
    return adj_prob
end