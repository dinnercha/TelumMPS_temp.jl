# ── MPO × MPS application: exact, zip-up, variational fit ─────────────────────
#
# All three work on retagged copies of the operands, so the MPO and MPS legs can
# never collide: MPO bra/ket legs "ap,o,i"/"ap,m,i", MPO bonds "ap,a,i", input MPS
# physical "ap,m,i" and bonds "ap,b,i", result bonds "ap,c,i".

# Retag an MPS: physical legs -> "$p,i", bonds -> "$k,i". The boundary legs (vacuum,
# or the total flux after a rank-3 `apply_op`) keep their tags, so they contract
# (or carry over to the result) exactly as in `overlap`.
function _tag_mps(ψ::TLMps, k::AbstractString, p::AbstractString)
    N = length(ψ)
    return Vector{TLArray}(map(1:N) do i
        q = setitag(ψ.TLA[i], "$p,$i"; itag="site,$i")
        i > 1 && (q = setitag(q, "$k,$(i-1)"; itag=_mps_bond(i - 1)))
        i < N && (q = setitag(q, "$k,$i"; itag=_mps_bond(i)))
        to_concrete(q)
    end)
end

# Boundary leg of a retagged MPS tensor at site i: the one that is neither physical nor a bond.
function _tag_bnd(q, i, N, k, p)
    inner = [findleg(q; itag="$p,$i")]
    i > 1 && push!(inner, findleg(q; itag="$k,$(i-1)"))
    i < N && push!(inner, findleg(q; itag="$k,$i"))
    return only(setdiff(eachindex(q.inds), inner))
end

# Retag an MPO: '+' (bra) leg -> "ap,o,i", '-' (ket) leg -> "ap,m,i", bonds
# "ap,a,i"; the vacuum boundary legs are dropped.
function _tag_mpo(H::TLMpo)
    N = length(H.TLA)
    return Vector{TLArray}(map(1:N) do i
        q = H.TLA[i]
        q = setitag(q, findleg(q; itag="site,$i", dir='+'), "ap,o,$i")
        q = setitag(q, findleg(q; itag="site,$i", dir='-'), "ap,m,$i")
        i > 1 && (q = setitag(q, "ap,a,$(i-1)"; itag=_mpo_bond(i - 1)))
        i < N && (q = setitag(q, "ap,a,$i"; itag=_mpo_bond(i)))
        i == 1 && (q = deleteSingleton(q, findleg(q; itag="dummy_l")))
        i == N && (q = deleteSingleton(q, findleg(q; itag="dummy_r")))
        to_concrete(q)
    end)
end

# Result tensors back to the MPS tags: "ap,o,i" -> "site,i", "ap,c,i" -> "bond,m,i".
function _untag(c)
    N = length(c)
    return Vector{TLArray}(map(1:N) do i
        q = setitag(c[i], "site,$i"; itag="ap,o,$i")
        i > 1 && (q = setitag(q, _mps_bond(i - 1); itag="ap,c,$(i-1)"))
        i < N && (q = setitag(q, _mps_bond(i); itag="ap,c,$i"))
        to_concrete(q)
    end)
end

# Same leg with its dual flag set; `legflip` dualizes, this only relabels.
function _setdual(q::TLArray, leg::Int, dual::Bool)
    idx = q.inds[leg]
    newidx = TLIndex(idx.itags, idx.dir, idx.plev, idx.lock, dual)
    return TLArray(q, ntuple(k -> k == leg ? newidx : q.inds[k], length(q.inds)))
end

# Right-to-left truncating sweep of a left-canonical chain (center N -> 1), so every
# cut is an optimal SVD truncation. Returns the largest discarded weight.
function _sweep_truncate!(A; Nkeep, cutoff)
    mdw = 0.0
    for i in length(A):-1:2
        bt = _mps_bond(i - 1); q = A[i]
        (; U, S, Vd) = truncated_svd(q, (findleg(q; itag=bt),); Nkeep, cutoff)
        mdw = max(mdw, _dw(q, U, S, Vd))
        A[i]   = setitag(Vd, bt; itag="svdR")
        A[i-1] = setitag(A[i-1] * (U * S), bt; itag="svdR")
    end
    return mdw
end

# Exact: H[i]·ψ[i] with the (MPO, MPS) bond pair fused into one leg.
function _apply_exact(H::TLMpo, ψ::TLMps; Nkeep, cutoff)
    N = length(ψ)
    a, b = _tag_mpo(H), _tag_mps(ψ, "ap,b", "ap,m")
    c = Vector{TLArray}(undef, N)
    FR = nothing
    for i in 1:N
        q = to_concrete(a[i] * b[i])
        if i > 1
            # Left fusion = conjugate of the previous right fusion, so the two are exact
            # inverses. A second `getIdentity` is not: on an incoming leg it dualizes and
            # re-sorts the qlabels, reversing the order of fused U(1) blocks (SU(2) is
            # self-dual, so only abelian bonds show it).
            F = to_concrete(FR')
            for t in ("ap,a,$(i-1)", "ap,b,$(i-1)")
                F = _setdual(F, findleg(F; itag=t), q.inds[findleg(q; itag=t)].dual)
            end
            t = "ap,c,$(i-1)"
            F = _setdual(F, findleg(F; itag=t), FR.inds[findleg(FR; itag=t)].dual)
            q = to_concrete(q * F)
        end
        if i < N
            FR = getIdentity((q, findleg(q; itag="ap,a,$i")), (q, findleg(q; itag="ap,b,$i"));
                             itag="ap,c,$i")
            q = to_concrete(q * FR)
        end
        c[i] = q
    end
    φ = _untag(c)
    Nkeep === nothing && cutoff <= 0 && return TLMps(φ, 0), 0.0
    for i in 1:N-1
        _move_right!(φ, i; cutoff=0.0, Nkeep=nothing)
    end
    return TLMps(φ, 1), _sweep_truncate!(φ; Nkeep, cutoff)
end

# Zip-up: left -> right, contract one site into the carried S·V† and split it again with
# a truncated SVD (the bonds are never fused), then an optimal right -> left SVD sweep.
# The local truncations are near-optimal only on a right-canonical ψ, so a copy is
# gauged first when needed.
function _apply_zipup(H::TLMpo, ψ::TLMps; Nkeep, cutoff)
    N = length(ψ)
    ψ.oc == 1 || (ψ = move_canonical!(deepcopy(ψ), 1))
    a, b = _tag_mpo(H), _tag_mps(ψ, "ap,b", "ap,m")
    lb = String(b[1].inds[_tag_bnd(b[1], 1, N, "ap,b", "ap,m")].itags)
    c = Vector{TLArray}(undef, N)
    mdw = 0.0; C = nothing
    for i in 1:N
        T = to_concrete(i == 1 ? a[1] * b[1] : (C * a[i]) * b[i])
        if i == N
            c[N] = T
            break
        end
        left = (findleg(T; itag=i == 1 ? lb : "ap,c,$(i-1)"), findleg(T; itag="ap,o,$i"))
        (; U, S, Vd) = truncated_svd(T, left, "ap,c,$i", "ap,r"; Nkeep, cutoff)
        mdw = max(mdw, _dw(T, U, S, Vd))
        c[i] = to_concrete(U)
        C = to_concrete(S * Vd)
    end
    φ = _untag(c)
    return TLMps(φ, 1), max(mdw, _sweep_truncate!(φ; Nkeep, cutoff))
end

# Kept weight Zd = Σ d·σ² of a bond (QSMPSLib's convergence measure).
_zd(S) = sum(d * s^2 for (s, d) in _weighted_singvals(S); init=0.0)

# Variational fit, the MPS version of the XTRG `fit_mpo_product` (QSMPSLib VariTS).
function _apply_fit(H::TLMpo, ψ::TLMps; Nkeep, cutoff, max_sweeps, tol, init)
    N = length(ψ)
    length(init) == N || throw(ArgumentError("apply_mpo: init has length $(length(init)), ψ $N"))
    a, b = _tag_mpo(H), _tag_mps(ψ, "ap,b", "ap,m")
    c = _tag_mps(init, "ap,c", "ap,o")
    _boundary_qlabel(init) == _boundary_qlabel(ψ) || throw(ArgumentError(
        "apply_mpo: init and ψ carry different total flux"))
    # `init` only seeds the first environments, which contract its boundary legs with
    # ψ's. dmrg!/tdvp! retag and flip the right boundary of the state they evolve, so
    # the two need not agree: give init's boundary legs ψ's tag and direction.
    for i in (1, N)
        l, r = _tag_bnd(c[i], i, N, "ap,c", "ap,o"), _tag_bnd(b[i], i, N, "ap,b", "ap,m")
        q = setitag(c[i], l, String(b[i].inds[r].itags))
        q.inds[l].dir == b[i].inds[r].dir || (q = legflip(q, l))
        c[i] = to_concrete(q)
    end
    # Caches, as in `fit_mpo_product`:
    #   X[n] = (L[n]·a[n])·b[n]   (c_{n-1}, o_n, a_n, b_n)      T = X[n]·Y[n+1]
    #   Y[n] = (R[n+1]·a[n])·b[n]  (a_{n-1}, b_{n-1}, o_n, c_n)  L[n+1] = X[n]·c[n]', R[n] = Y[n]·c[n]'
    L = Vector{Any}(nothing, N + 1); R = Vector{Any}(nothing, N + 1)
    X = Vector{Any}(nothing, N); Y = Vector{Any}(nothing, N)
    xof(n) = to_concrete(L[n] === nothing ? a[n] * b[n] : (L[n] * a[n]) * b[n])
    yof(n) = to_concrete(R[n+1] === nothing ? a[n] * b[n] : (R[n+1] * a[n]) * b[n])
    # Left legs of the two-site tensor at bond n; at n = 1 that is ψ's left boundary leg.
    lb = String(b[1].inds[_tag_bnd(b[1], 1, N, "ap,b", "ap,m")].itags)
    leftlegs(T, n) = (findleg(T; itag=n > 1 ? "ap,c,$(n-1)" : lb), findleg(T; itag="ap,o,$n"))
    for n in N:-1:2
        Y[n] = yof(n); R[n] = to_concrete(Y[n] * c[n]')
    end
    zd = zeros(N - 1)
    max_dw = 0.0
    for sweep in 1:max_sweeps
        max_dw = 0.0
        for n in 1:N-1                                   # L → R
            X[n] = xof(n)
            T = to_concrete(X[n] * Y[n+1])
            (; U, S, Vd) = truncated_svd(T, leftlegs(T, n), "fitL", "fitR"; Nkeep, cutoff)
            max_dw = max(max_dw, _dw(T, U, S, Vd))
            c[n] = to_concrete(setitag(U, "ap,c,$n"; itag="fitL"))
            c[n+1] = to_concrete(setitag(S * Vd, "ap,c,$n"; itag="fitL"))
            L[n+1] = to_concrete(X[n] * c[n]')
        end
        for n in N-1:-1:1                                # R → L
            Y[n+1] = yof(n + 1)
            T = to_concrete(X[n] * Y[n+1])
            (; U, S, Vd) = truncated_svd(T, leftlegs(T, n), "fitL", "fitR"; Nkeep, cutoff)
            max_dw = max(max_dw, _dw(T, U, S, Vd))
            zd[n] = _zd(S)
            c[n] = to_concrete(setitag(U * S, "ap,c,$n"; itag="fitR"))
            c[n+1] = to_concrete(setitag(Vd, "ap,c,$n"; itag="fitR"))
            R[n+1] = to_concrete(Y[n+1] * c[n+1]')
        end
        (maximum(zd) - minimum(zd)) / maximum(zd) < tol && break
    end
    return TLMps(_untag(c), 1), max_dw
end

"""
    apply_mpo(H::TLMpo, ψ::TLMps; alg=:fit, Nkeep=nothing, cutoff=0.0,
              max_sweeps=3, tol=1e-12, init=ψ) -> (φ, max_dw)

`|φ⟩ ≈ H|ψ⟩`. `Nkeep` caps the bond dimension (in multiplets) and `cutoff` is the
discarded weight per cut (see [`truncated_svd`](@ref)); `max_dw` is the largest
discarded weight met. `ψ` is not modified.

- `alg=:exact` — site-by-site contraction with the MPS and MPO bonds fused (bond
  `D·w`); compressed by an SVD sweep only when `Nkeep` or `cutoff` is given.
- `alg=:zipup` — left→right sweep contracting one site at a time and truncating as
  it goes (the `D·w` bond is never formed), then an optimal right→left SVD sweep.
- `alg=:fit` — two-site variational fit (QSMPSLib VariTS) seeded by `init`; stops
  once the R→L sweep's kept weights `Zd = Σ d·σ²` agree to `tol`
  (`(max Zd - min Zd)/max Zd < tol`), after at most `max_sweeps` sweeps. `init`
  must carry `ψ`'s total flux; it may come from a `dmrg!`/`tdvp!` run while `ψ`
  does not. `max_sweeps`, `tol` and `init` are used by `:fit` only.

`φ` carries `ψ`'s boundary legs (so its total flux), is not normalised
(`‖φ‖² = ⟨ψ|H²|ψ⟩`), and has its canonical center on site 1 — except `:exact`
without truncation, which is left uncanonicalized (`oc = 0`).
"""
function apply_mpo(H::TLMpo, ψ::TLMps; alg::Symbol=:fit, Nkeep::Union{Nothing, Integer}=nothing,
                   cutoff::Float64=0.0, max_sweeps::Integer=3, tol::Float64=1e-12, init::TLMps=ψ)
    length(H.TLA) == length(ψ) || throw(ArgumentError(
        "apply_mpo: length mismatch (H $(length(H.TLA)), ψ $(length(ψ)))"))
    alg === :exact && return _apply_exact(H, ψ; Nkeep, cutoff)
    alg === :zipup && return _apply_zipup(H, ψ; Nkeep, cutoff)
    alg === :fit   && return _apply_fit(H, ψ; Nkeep, cutoff, max_sweeps, tol, init)
    throw(ArgumentError("apply_mpo: alg must be :exact, :zipup or :fit; got :$alg"))
end
