# ── variational MPO × MPS fit (QSMPSLib VariTS, MPS version of `fit_mpo_product`) ──
#
# Same two-site scheme as the MPO × MPO fit of the XTRG code, with the second
# operand and the result being MPSs: |C⟩ ≈ H|B⟩ at bond Nkeep, never forming the
# χ·w bond of the exact product.

# Retag an MPS for the fit: physical legs -> "$p,i", bonds -> "$k,i". The boundary
# legs (vacuum, or the total flux after a rank-3 `apply_op`) keep their tags, so
# ket and bra boundaries contract exactly as in `overlap` and the result inherits
# the input's flux.
function _fit_mps(ψ::TLMps, k::AbstractString, p::AbstractString)
    N = length(ψ)
    return map(1:N) do i
        q = setitag(ψ.TLA[i], "$p,$i"; itag="site,$i")
        i > 1 && (q = setitag(q, "$k,$(i-1)"; itag=_mps_bond(i - 1)))
        i < N && (q = setitag(q, "$k,$i"; itag=_mps_bond(i)))
        to_concrete(q)
    end
end

# Boundary leg of a fit-tagged MPS tensor at site i: the one that is neither physical nor a bond.
function _fit_bnd(q, i, N, k, p)
    inner = [findleg(q; itag="$p,$i")]
    i > 1 && push!(inner, findleg(q; itag="$k,$(i-1)"))
    i < N && push!(inner, findleg(q; itag="$k,$i"))
    return only(setdiff(eachindex(q.inds), inner))
end

# MPO for the fit: '+' (bra) leg -> "fit,o,i", '-' (ket) leg -> "fit,m,i", bonds
# "fit,a,i"; the vacuum boundary legs are dropped.
function _fit_mpo(H::TLMpo)
    N = length(H.TLA)
    return map(1:N) do i
        q = H.TLA[i]
        q = setitag(q, findleg(q; itag="site,$i", dir='+'), "fit,o,$i")
        q = setitag(q, findleg(q; itag="site,$i", dir='-'), "fit,m,$i")
        i > 1 && (q = setitag(q, "fit,a,$(i-1)"; itag=_mpo_bond(i - 1)))
        i < N && (q = setitag(q, "fit,a,$i"; itag=_mpo_bond(i)))
        i == 1 && (q = deleteSingleton(q, findleg(q; itag="dummy_l")))
        i == N && (q = deleteSingleton(q, findleg(q; itag="dummy_r")))
        to_concrete(q)
    end
end

# Kept weight Zd = Σ d·σ² of a bond (QSMPSLib's convergence measure).
_zd(S) = sum(d * s^2 for (s, d) in _weighted_singvals(S); init=0.0)

"""
    fit_apply(H::TLMpo, ψ::TLMps; Nkeep, cutoff=0.0, max_sweeps=3, tol=1e-12, init=ψ)
        -> (φ, max_dw, sweeps)

Two-site variational fit of `|φ⟩ ≈ H|ψ⟩` at bond `Nkeep`; `cutoff` is the
discarded weight (see [`truncated_svd`](@ref)). As in QSMPSLib `VariTS` (and
`fit_mpo_product`), the R→L half-sweep's kept weights `Zd = Σ d·σ²` stop the fit
once `(max Zd - min Zd)/max Zd < tol`, after at most `max_sweeps` sweeps.

`init` seeds the fitted state (default: `ψ` itself); it must carry `ψ`'s total
flux, and may come from a `dmrg!`/`tdvp!` run while `ψ` does not. The result is not
normalised (`‖φ‖² = ⟨ψ|H²|ψ⟩`), carries `ψ`'s boundary legs (so its flux) and
comes out with its canonical center on site 1.
"""
function fit_apply(H::TLMpo, ψ::TLMps; Nkeep::Integer, cutoff::Float64=0.0,
                   max_sweeps::Integer=3, tol::Float64=1e-12, init::TLMps=ψ)
    N = length(ψ)
    N == length(H.TLA) == length(init) || throw(ArgumentError(
        "fit_apply: length mismatch (H $(length(H.TLA)), ψ $N, init $(length(init)))"))
    a = _fit_mpo(H)
    b = _fit_mps(ψ, "fit,b", "fit,m")
    c = _fit_mps(init, "fit,c", "fit,o")
    _boundary_qlabel(init) == _boundary_qlabel(ψ) || throw(ArgumentError(
        "fit_apply: init and ψ carry different total flux"))
    # `init` only seeds the first environments, which contract its boundary legs with
    # ψ's. dmrg!/tdvp! retag and flip the right boundary of the state they evolve, so
    # the two need not agree: give init's boundary legs ψ's tag and direction.
    for i in (1, N)
        l, r = _fit_bnd(c[i], i, N, "fit,c", "fit,o"), _fit_bnd(b[i], i, N, "fit,b", "fit,m")
        q = setitag(c[i], l, String(b[i].inds[r].itags))
        q.inds[l].dir == b[i].inds[r].dir || (q = legflip(q, l))
        c[i] = to_concrete(q)
    end
    # Caches as in `fit_mpo_product`:
    #   X[n] = (L[n]·a[n])·b[n]   (c_{n-1}, o_n, a_n, b_n)      T = X[n]·Y[n+1]
    #   Y[n] = (R[n+1]·a[n])·b[n]  (a_{n-1}, b_{n-1}, o_n, c_n)  L[n+1] = X[n]·c[n]', R[n] = Y[n]·c[n]'
    L = Vector{Any}(nothing, N + 1); R = Vector{Any}(nothing, N + 1)
    X = Vector{Any}(nothing, N); Y = Vector{Any}(nothing, N)
    xof(n) = to_concrete(L[n] === nothing ? a[n] * b[n] : (L[n] * a[n]) * b[n])
    yof(n) = to_concrete(R[n+1] === nothing ? a[n] * b[n] : (R[n+1] * a[n]) * b[n])
    # Left legs of the two-site tensor at bond n; at n = 1 that is ψ's left boundary leg.
    lb = String(b[1].inds[_fit_bnd(b[1], 1, N, "fit,b", "fit,m")].itags)
    leftlegs(T, n) = (findleg(T; itag=n > 1 ? "fit,c,$(n-1)" : lb), findleg(T; itag="fit,o,$n"))
    for n in N:-1:2
        Y[n] = yof(n); R[n] = to_concrete(Y[n] * c[n]')
    end
    zd = zeros(N - 1)
    max_dw = 0.0; sweeps = 0
    for sweep in 1:max_sweeps
        max_dw = 0.0
        for n in 1:N-1                                   # L → R
            X[n] = xof(n)
            T = to_concrete(X[n] * Y[n+1])
            (; U, S, Vd) = truncated_svd(T, leftlegs(T, n), "fitL", "fitR"; Nkeep, cutoff)
            max_dw = max(max_dw, _dw(T, U, S, Vd))
            c[n] = to_concrete(setitag(U, "fit,c,$n"; itag="fitL"))
            c[n+1] = to_concrete(setitag(S * Vd, "fit,c,$n"; itag="fitL"))
            L[n+1] = to_concrete(X[n] * c[n]')
        end
        for n in N-1:-1:1                                # R → L
            Y[n+1] = yof(n + 1)
            T = to_concrete(X[n] * Y[n+1])
            (; U, S, Vd) = truncated_svd(T, leftlegs(T, n), "fitL", "fitR"; Nkeep, cutoff)
            max_dw = max(max_dw, _dw(T, U, S, Vd))
            zd[n] = _zd(S)
            c[n] = to_concrete(setitag(U * S, "fit,c,$n"; itag="fitR"))
            c[n+1] = to_concrete(setitag(Vd, "fit,c,$n"; itag="fitR"))
            R[n+1] = to_concrete(Y[n+1] * c[n+1]')
        end
        sweeps = sweep
        (maximum(zd) - minimum(zd)) / maximum(zd) < tol && break
    end
    φ = map(1:N) do i                                    # back to the MPS tags
        q = setitag(c[i], "site,$i"; itag="fit,o,$i")
        i > 1 && (q = setitag(q, _mps_bond(i - 1); itag="fit,c,$(i-1)"))
        i < N && (q = setitag(q, _mps_bond(i); itag="fit,c,$i"))
        to_concrete(q)
    end
    return TLMps(φ, 1), max_dw, sweeps
end
