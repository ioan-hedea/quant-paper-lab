# visualize_barrier.jl
# No need to edit anything here, all parts of the exercise are in
# barrier_certificate.jl and lyapunov_certificate.jl

using Plots
using MultivariatePolynomials

function visualize_barrier(monsB, coeffs;
    dynamics = (x1, x2) -> (-x1 + x1*x2, -x2),
    g_initial = (x1, x2) -> 0.5^2 - (x1^2 + x2^2),
    g_unsafe  = (x1, x2) -> 0.25^2 - ((x1 - 1.0)^2 + x2^2),
    g_domain  = (x1, x2) -> 2.0^2 - (x1^2 + x2^2),
    R = 2.0,
    n = 201,
    nq = 15,
    show_vector_field = true,
    normalize_arrows = true,
    arrow_scale = 0.10,
    cmap = :viridis,
    arrow_color = :black,
    title_text = "Barrier certificate",
    legend_pos = :topright,
    show_plot = true
)
    # Robust polynomial reconstruction
    vars = collect(variables(monsB))
    @assert length(vars) == 2 "visualize_barrier assumes exactly 2 variables."
    v1, v2 = vars[1], vars[2]

    Bpoly = sum(coeffs[i] * monsB[i] for i in eachindex(coeffs))

    Bfun = (a, b) -> begin
        val = Bpoly(v1 => a, v2 => b)
        return val isa Number ? float(val) : float(constantterm(val))
    end

    xmin, xmax = -R, R
    ymin, ymax = -R, R
    xs = range(xmin, xmax, length=n)
    ys = range(ymin, ymax, length=n)

    ZB = [Bfun(x, y) for y in ys, x in xs]
    ZI = [g_initial(x, y) for y in ys, x in xs]
    ZU = [g_unsafe(x, y)  for y in ys, x in xs]
    ZD = [g_domain(x, y)  for y in ys, x in xs]

    plt = heatmap(xs, ys, ZB;
        xlabel="x1",
        ylabel="x2",
        title=title_text,
        aspect_ratio=:equal,
        color=cmap,
        colorbar=true,
        label=false,
        legend=:bottomleft,
        # size=(600, 400)
    )

    if show_vector_field
        xq = range(xmin, xmax, length=nq)
        yq = range(ymin, ymax, length=nq)
        X = [x for y in yq, x in xq]
        Y = [y for y in yq, x in xq]

        U = similar(X)
        W = similar(X)
        for j in eachindex(X)
            dx1, dx2 = dynamics(X[j], Y[j])
            U[j] = dx1
            W[j] = dx2
        end

        if normalize_arrows
            Un = similar(U)
            Wn = similar(W)
            for j in eachindex(U)
                s = sqrt(U[j]^2 + W[j]^2)
                if s < 1e-12
                    Un[j] = 0.0
                    Wn[j] = 0.0
                else
                    Un[j] = U[j] / s
                    Wn[j] = W[j] / s
                end
            end
            quiver!(plt, vec(X), vec(Y);
                quiver=(arrow_scale .* vec(Un), arrow_scale .* vec(Wn)),
                linewidth=1.2,
                color=arrow_color,
                label="Vector field"
            )
        else
            quiver!(plt, vec(X), vec(Y);
                quiver=(arrow_scale .* vec(U), arrow_scale .* vec(W)),
                linewidth=1.2,
                color=arrow_color,
                label="Vector field"
            )
        end
    end

    contour!(plt, xs, ys, ZI; levels=[0.0], linewidth=2.5, color=:cyan, label="Initial boundary")
    contour!(plt, xs, ys, ZU; levels=[0.0], linewidth=2.5, color=:red,  label="Unsafe boundary")
    contour!(plt, xs, ys, ZD; levels=[0.0], linewidth=2.5, color=:white, linestyle=:dash, label="Domain boundary")
    contour!(plt, xs, ys, ZB;
        levels=[0.0],
        linewidth=2.5,
        color=:white,
        label="B(x) = 0 (barrier)"
    )

    xlims!(plt, xmin, xmax)
    ylims!(plt, ymin, ymax)

    if show_plot
        display(plt)
    end

    return plt
end


"""
    visualize_lyapunov(monsV, coeffs;
        dynamics = (x1, x2) -> (-x1 + x1*x2, -x2),
        g_domain = (x1, x2) -> 2.0^2 - (x1^2 + x2^2),
        R = 2.0,
        show_vector_field = true,
        normalize_arrows = true,
        arrow_scale = 0.1,
        nq = 15,
        n = 201,
        cmap = :viridis,
        arrow_color = :red,
        show_plot = true
    )

Visualize a 2D Lyapunov candidate V(x) given monomials `monsV` and numeric coefficients `coeffs`.
Overlays:
- heatmap of V
- optional direction field
- contour for domain boundary g_domain=0
"""
function visualize_lyapunov(monsV, coeffs;
    dynamics = (x1, x2) -> (-x1 + x1*x2, -x2),
    g_domain = (x1, x2) -> 2.0^2 - (x1^2 + x2^2),
    R = 2.0,
    show_vector_field = true,
    normalize_arrows = true,
    arrow_scale = 0.1,
    nq = 15,
    n = 201,
    cmap = :viridis,
    arrow_color = :red,
    show_plot = true
)
    vars = collect(variables(monsV))
    @assert length(vars) == 2 "visualize_lyapunov assumes a 2D state."

    exps = [exponents(m) for m in monsV]

    # numeric evaluator for V
    Vfun = (a, b) -> begin
        s = 0.0
        for i in eachindex(coeffs)
            e1 = exps[i][1]
            e2 = exps[i][2]
            s += coeffs[i] * (a^e1) * (b^e2)
        end
        s
    end

    xmin, xmax = -R, R
    ymin, ymax = -R, R

    xs = range(xmin, xmax, length=n)
    ys = range(ymin, ymax, length=n)

    ZV = [Vfun(x, y) for y in ys, x in xs]
    ZD = [g_domain(x, y) for y in ys, x in xs]

    vmin = min(minimum(ZV),0)
    vmax = maximum(ZV)

    plt = heatmap(xs, ys, ZV;
        xlabel="x1",
        ylabel="x2",
        title="Lyapunov candidate V(x)",
        aspect_ratio=:equal,
        color=cmap,
        colorbar=true,
        clims=(vmin,vmax),
    )

    contour!(plt, xs, ys, ZD; levels=[0.0], linewidth=5.0, linecolor=:white, linestyle=:dash, fill=false, label=false)   
    theta = range(0, 2pi, length=400)
    plot!(plt,
        R .* cos.(theta),
        R .* sin.(theta);
        linecolor=:white,
        linestyle=:dash,
        linewidth=2.0,
        label=false
    )

    if show_vector_field
        xq = range(xmin, xmax, length=nq)
        yq = range(ymin, ymax, length=nq)
        X = [x for y in yq, x in xq]
        Y = [y for y in yq, x in xq]

        U = similar(X)
        W = similar(X)
        for j in eachindex(X)
            dx1, dx2 = dynamics(X[j], Y[j])
            U[j] = dx1
            W[j] = dx2
        end

        if normalize_arrows
            Un = similar(U)
            Wn = similar(W)
            for j in eachindex(U)
                s = sqrt(U[j]^2 + W[j]^2)
                if s < 1e-12
                    Un[j] = 0.0
                    Wn[j] = 0.0
                else
                    Un[j] = U[j] / s
                    Wn[j] = W[j] / s
                end
            end
            quiver!(plt, vec(X), vec(Y),
                quiver=(arrow_scale .* vec(Un), arrow_scale .* vec(Wn)),
                linewidth=1.2,
                color=arrow_color,
                label=false
            )
        else
            quiver!(plt, vec(X), vec(Y),
                quiver=(arrow_scale .* vec(U), arrow_scale .* vec(W)),
                linewidth=1.2,
                color=arrow_color,
                label=false
            )
        end
    end

    xlims!(plt, xmin, xmax)
    ylims!(plt, ymin, ymax)

    if show_plot
        display(plt)
    end
    return plt
end