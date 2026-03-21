# barrier_certificate.jl
#
# Exercise: Sum-of-squares (SOS) barrier certificate synthesis (for a continuous-time, polynomial ODE)
#
# Goal: Fill in the TODOs so the script runs and synthesizes a polynomial barrier B(x)
#       that separates an initial set from an unsafe set while satisfying a
#       barrier condition on a given domain.

using JuMP
using SumOfSquares
using DynamicPolynomials
using MultivariatePolynomials
import MathOptInterface as MOI
include("visualize.jl")

using SCS
const SOLVER = SCS.Optimizer

# -------------------------
# 1) State vector
# -------------------------
@polyvar x1 x2
x = [x1, x2]

# -------------------------
# 2) Dynamics: ẋ = f(x)
# -------------------------
f = [
    -x1 + x1 * x2,
    -x2
]

# -------------------------
# 3) Specification sets
# -------------------------
# Initial set: states where the system may start
# Circle of radius 0.5 centered around the origin
r_init = 0.5
initial_states = r_init^2 - (x1^2 + x2^2)

# Unsafe set: states that must never be reached
# Circle of radius 0.25 centered around (1, 0)
r_unsafe = 0.25
unsafe_states = r_unsafe^2 - ((x1 - 1)^2 + x2^2)

# Domain: region where the barrier condition must hold
# Circle of radius 2 centered around the origin
r_domain = 2.0
domain = r_domain^2 - (x1^2 + x2^2)

# -------------------------
# 4) Degrees and scalars
# -------------------------
# degB: maximum degree of the barrier polynomial B(x).
# Increasing this allows more expressive barriers but makes the
# resulting semidefinite program larger and slower to solve.
degB = 4

# degS: maximum degree of the SOS multiplier polynomials.
# These multipliers are used to enforce nonnegativity over semialgebraic sets.
# Higher degrees give more flexibility but also increase solver cost.
degS = 2

gamma = 1e-2
lambda = 1.0

# -------------------------
# 5) Build SOS model
# -------------------------
model = Model(SOLVER)
set_silent(model)

monsB = monomials(x, 0:degB)
@variable(model, b[1:length(monsB)])
B = sum(b[i] * monsB[i] for i in eachindex(monsB))

gradB = differentiate.(Ref(B), x)
Bdot = sum(gradB[i] * f[i] for i in eachindex(x))

monsS = monomials(x, 0:degS)

# -------------------------
# 6) SOS constraints
# -------------------------
# We encode the barrier certificate conditions directly as SOS constraints.

# (A) Initial set condition: B(x) <= -gamma on initial_states
# TODO: Complete the SOS encoding for this condition
@variable(model, s1, SOSPoly(monsS))
# expr_A = 
# @constraint(model, expr_A in SOSCone())

# (B) Unsafe set condition: B(x) >= gamma on unsafe_states
# TODO: Complete the SOS encoding for this condition
@variable(model, s2, SOSPoly(monsS))
# expr_B = ...
# @constraint(model, expr_B in SOSCone())

# (C) Flow condition: Bdot(x) + lambda*B(x) <= 0 on domain
# TODO: Complete the SOS encoding for this condition
@variable(model, s3, SOSPoly(monsS))
# expr_C = ...
# @constraint(model, expr_C in SOSCone())

# -------------------------
# 7) Normalization
# -------------------------
# Fix the constant coefficient of B(x) to remove scaling ambiguity.
# If B(x) is a valid barrier, then c*B(x) for any c>0 is also valid.
# This constraint picks one representative by enforcing B(0,0) = -gamma
# (since the first monomial in monsB is the constant 1).
@constraint(model, b[1] == -gamma)

# -------------------------
# 8) Solve
# -------------------------
optimize!(model)

term = MOI.get(model, MOI.TerminationStatus())
if term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    B_sol = value(B)
    println("\nFound barrier candidate B(x):\n")
    println(B_sol)
    visualize_barrier(monsB, value.(b))
else
    println("\nNo solution.")
end
