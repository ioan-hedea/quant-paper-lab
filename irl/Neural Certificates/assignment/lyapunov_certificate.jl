# lyapunov_certificate.jl
#
# Exercise: Sum-of-squares (SOS) Lyapunov certificate synthesis
#          (for a continuous-time, polynomial ODE)
#
# Goal: Fill in the TODOs so the script synthesizes a polynomial Lyapunov
#       function V(x) that certifies (local) stability of the origin on a
#       given domain.

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
# 3) Specification set
# -------------------------
# Domain: region where the barrier condition must hold
# Circle of radius 2 centered around the origin
r_domain = 2.0
domain = r_domain^2 - (x1^2 + x2^2)

# -------------------------
# 4) Degrees and scalars
# -------------------------
# degV: maximum degree of the Lyapunov polynomial V(x).
# degS: maximum degree of the SOS multiplier polynomials.
degV = 4
degS = 2

# eps: small margin used in positivity/decrease conditions (can be tuned, 
# but this is not necessary in this exercise)
eps = 1e-2

# -------------------------
# 5) Build SOS model
# -------------------------
model = Model(SOLVER)
set_silent(model)

# Parameterize V(x) in a monomial basis
monsV = monomials(x, 0:degV)
@variable(model, v[1:length(monsV)])
V = sum(v[i] * monsV[i] for i in eachindex(monsV))

# Lie derivative: Vdot(x) = ∇V(x) · f(x)
gradV = differentiate.(Ref(V), x)
Vdot = sum(gradV[i] * f[i] for i in eachindex(x))

monsS = monomials(x, 0:degS)

# -------------------------
# 6) SOS constraints
# -------------------------
# TODO: Complete the SOS encodings for the Lyapunov conditions as we did in the lecture.
# (A) Positivity on the domain:
#     V(x) >= eps*(x1^2 + x2^2) on domain
@variable(model, s_pos, SOSPoly(monsS))
# expr_pos = ...
# @constraint(model, expr_pos in SOSCone())

# (B) Decrease along trajectories on the domain:
#     -Vdot(x) >= eps*(x1^2 + x2^2) on domain
@variable(model, s_dec, SOSPoly(monsS))
# expr_dec = ...
# @constraint(model, expr_dec in SOSCone())

# -------------------------
# 7) Normalization
# -------------------------
# Fix V(0,0) = 0 by setting the constant coefficient to 0.
# This removes a trivial additive shift.
@constraint(model, v[1] == 0.0)

# -------------------------
# 8) Solve
# -------------------------
optimize!(model)

term = MOI.get(model, MOI.TerminationStatus())
if term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    V_sol = value(V)
    println("\nFound Lyapunov candidate V(x):\n")
    println(V_sol)
    # Visualize (requires visualize_lyapunov in visualize.jl)
    visualize_lyapunov(monsV, value.(v))
else
    println("\nNo solution.")
end