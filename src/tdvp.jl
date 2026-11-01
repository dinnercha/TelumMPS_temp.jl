using Printf
using Serialization
using Dates

# TDVP time evolution for the TLMps/TLMpo wrapper. Algorithm follows
# ITensor_Clifford/src/CATDVP/TDVP.jl, but the effective Hamiltonian comes from
# the MPO environments (as in dmrg2s.jl) instead of a term-wise h_eff.
#
# Environment stack `Hrl` (length N+2), same convention as the DMRG engine:
#   left  env of site j = Hrl[j]     (sites 1..j-1 contracted)
#   right env of site j = Hrl[j+2]   (sites j+1..N contracted)
# DMRG sweeps right->left first, so `_getHrl` seeds the stack with LEFT
# environments; TDVP starts at site 1, so `_tdvp_prep!` seeds it with RIGHT
# environments instead.

# H_eff applied to a window tensor: contract L, then the MPO tensors one at a
# time, then R. `_eigs_GS` pre-fuses the MPO tensors (W1*W2) instead; that costs
# O(w³d⁴) up front and O(D²d⁴w²) per matvec, while this order stays at
# O(D²d³w²) and keeps each MPO tensor's block sparsity intact.
_applyH(Hl, Ws, Hr, A) = foldl(*, Ws; init = Hl * A) * Hr

# exp(τ·A)·x by Arnoldi with full (twice-repeated) Gram-Schmidt. H_eff is
# Hermitian so the projected matrix is tridiagonal up to round-off; the full T
# is kept anyway since k ≤ nKrylov is tiny. Convergence uses Saad's a-posteriori
# estimate h_{j+1,j}·|[exp(τT)]_{j,1}|, relative to ‖x‖ so the criterion does not
# depend on how the input is scaled — the reference engine instead
# rebuilds the Krylov combination every iteration to compare it with the last,
# which costs one extra vector sum per step.
#
# `lanczos=true` uses the Hermitian three-term recurrence instead: one inner product
# and one (three-term) TLArray sum per Krylov vector rather than 2j of each. With
# SU(2) TLArrays every sum/inner product pays a w-matrix alignment, which made the
# Gram-Schmidt ~85% of a thermal-TDVP sweep at small bond dimension.
function _expm_krylov(applyA, x, τ::Number; nKrylov::Int=20, tol::Float64=1e-12,
                      lanczos::Bool=false)
    β = norm(x)
    β == 0 && return x
    V = Any[x / β]
    # Real x with real τ keeps the whole recursion real; only a complex τ (real
    # time) promotes, and then only in the final combination.
    T = zeros(promote_type(eltype(x), typeof(τ)), nKrylov, nKrylov)
    k = nKrylov
    for j in 1:nKrylov
        w = applyA(V[j])
        if lanczos
            α = real((V[j]' * w)[])
            T[j, j] = α
            w = to_concrete(j == 1 ? w - α * V[j] : sum((w, -α * V[j], -T[j, j-1] * V[j-1])))
        else
            for _ in 1:2, i in 1:j                 # reorthogonalize twice
                h = (V[i]' * w)[]
                T[i, j] += h
                w = w - h * V[i]
            end
        end
        nw = norm(w)
        # Both tests are relative. `nw` is on the scale of the unit vector V[j]
        # (β divided out: with β kept in, a thermal MPO with ‖ρ‖ ~ 2^(N/2) at
        # N = 100 stopped Arnoldi at j = 1). Saad's estimate nw·|e_jᵀ exp(τT)e₁|
        # must also be divided by ‖exp(τT)e₁‖, the size of the result: in
        # imaginary time that norm is ~e^{|τ|·|E|} ≫ 1, the absolute estimate
        # never fell below tol, every call ran all nKrylov steps, and the backward
        # bond step e^{+|h|K} was resolved far up its spectrum, amplifying
        # high-energy components ~e^{10–25} into the next SVD truncation (full
        # re-orthogonalisation gives the same result, so not an orthogonality
        # problem; early stopping regularises) — a one-sided tanTRG step Δβ = 1 over-cooled
        # 4x4 Hubbard (E/N −4.391 vs −4.352 at β = 4, D128; QSMPSLib uses the
        # relative |c_end|/‖c‖). In real time ‖exp(τT)e₁‖ = 1, unchanged.
        c = j == nKrylov ? nothing : exp(τ * T[1:j, 1:j])[:, 1]
        if j == nKrylov || nw <= tol || nw * abs(c[j]) <= tol * norm(c)
            k = j
            break
        end
        T[j+1, j] = nw
        lanczos && (T[j, j+1] = nw)
        push!(V, w / nw)
    end
    c = exp(τ * T[1:k, 1:k])[:, 1] .* β
    # Vector, not Tuple: a k-tuple compiles a new `_sum_tlarrays_aligned` (2–3 s each) for every Krylov
    # dimension k the first time it occurs — the β = 3 step of a tanTRG run (first k = 14, 15) took 25 s vs 5.
    return to_concrete(k == 1 ? c[1] * V[1] : sum([c[i] * V[i] for i in 1:k]))
end

# Break the MPS/MPO shared "dummy_r" boundary tag (idempotent: `dmrg!` may have
# done it already), then build the RIGHT environment stack.
function _tdvp_prep!(MPS, MPO)
    N = length(MPS)
    Hrl = Vector{TLArray}(undef, N + 2)

    fresh = findleg(MPS[N]; itag="dummy_r") !== nothing   # false once prepared
    fresh && (MPS[N] = setitag(MPS[N], "SRight"; itag="dummy_r"))
    li  = findleg(MPS[N]; itag="SRight")
    cap = to_concrete(addSingleton(getIdentity((MPS[N], li); itag="SRight"), 3;
                                   itag="dummy_r", dir='+'))
    if fresh
        MPS[N]   = legflip(MPS[N]; itag="SRight")
        Hrl[N+2] = legflip(cap; itag="SRight")
    else
        # Already prepared (e.g. by `dmrg!`): the boundary leg is flipped, so the
        # cap must be too. `getIdentity` mirrors that leg but hands its partner
        # back undualled — flip just the partner to land on the same cap the
        # fresh path builds.
        pl = only(l for l in findlegs(cap; itag="SRight") if !cap.inds[l].dual)
        Hrl[N+2] = legflip(cap, pl)
    end

    # Both branches above end on a `legflip` of the cap: a no-op on a vacuum
    # boundary, but on a flux-carrying state (anything `apply_op` returns for a
    # rank-3 IROP, e.g. `c|GS⟩` for a Green's function) it drags in that irrep's
    # 1j factor and every right environment below inherits it. Correct the cap
    # against the final MPS[N] before the recursion. See `_cap_flux_fix`.
    length(MPS[N].inds) == 3 &&
        (Hrl[N+2] = _cap_flux_fix(Hrl[N+2], MPS[N], _mps_bond(N - 1)))

    li = findleg(MPS[1]; itag="SLeft")
    Hrl[1] = addSingleton(getIdentity((MPS[1]', li); itag="SLeft"), 3;
                          itag="dummy_l", dir='-')

    for i in N:-1:2
        Hrl[i+1] = (Hrl[i+2] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag="bond,m,$(i-1)")
    end
    return Hrl
end

# ⟨H⟩/⟨ψ|ψ⟩ with the center on site 1 (where every sweep ends).
function _tdvp_energy(MPS, MPO, Hrl)
    HA = _applyH(Hrl[1], (MPO[1],), Hrl[3], MPS[1])
    return real((MPS[1]' * HA)[]) / norm(MPS[1])^2
end

# Two-site sweep: forward exp(h·H_eff) on each bond, backward exp(-h·H_eff) on
# the single site left behind. Center starts and ends at 1; every bond is
# visited once per direction, so one call advances the state by 2h.
function _sweep_2s_tdvp!(MPS, MPO, Hrl, Nkeep; h, nKrylov, tol, cutoff, lanczos=false)
    N = length(MPO); mdw = 0.0

    for i in 1:N-1                                     # left -> right
        M = _expm_krylov(A -> _applyH(Hrl[i], (MPO[i], MPO[i+1]), Hrl[i+3], A),
                         MPS[i] * MPS[i+1], h; nKrylov, tol, lanczos)
        lids = _tdvp_left_legs(MPS[i], M, i)
        (; U, S, Vd) = truncated_svd(M, lids, "bond,m,$i,left", "bond,m,$i,right";
                                     Nkeep=Nkeep, cutoff=cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        MPS[i]   = removeitag(U, "left")
        MPS[i+1] = removeitag(S * Vd, "left")
        Hrl[i+1] = (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag="bond,m,$i")
        i < N-1 && (MPS[i+1] = _expm_krylov(
            A -> _applyH(Hrl[i+1], (MPO[i+1],), Hrl[i+3], A), MPS[i+1], -h; nKrylov, tol, lanczos))
    end

    for i in N-1:-1:1                                  # right -> left
        M = _expm_krylov(A -> _applyH(Hrl[i], (MPO[i], MPO[i+1]), Hrl[i+3], A),
                         MPS[i] * MPS[i+1], h; nKrylov, tol, lanczos)
        lids = _tdvp_left_legs(MPS[i], M, i)
        (; U, S, Vd) = truncated_svd(M, lids, "bond,m,$i,left", "bond,m,$i,right";
                                     Nkeep=Nkeep, cutoff=cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        MPS[i]   = removeitag(U * S, "right")
        MPS[i+1] = removeitag(Vd, "right")
        Hrl[i+2] = (Hrl[i+3] * MPS[i+1]) * MPO[i+1] * lock(MPS[i+1]'; itag="bond,m,$i")
        i > 1 && (MPS[i] = _expm_krylov(
            A -> _applyH(Hrl[i], (MPO[i],), Hrl[i+2], A), MPS[i], -h; nKrylov, tol, lanczos))
    end

    return mdw
end

# Left-side SVD legs of a two-site TDVP window.  Ordinary MPS tensors have one
# physical leg; thermal MPO states have two.  The bond tags are the only legs
# that are not physical, so deriving the selection from the local tensor keeps
# the TDVP engine rank-generic without changing the old rank-3 path.
function _tdvp_left_legs(A, M, i)
    rb = findleg(A; itag="bond,m,$i")
    phys = [A.inds[l].itags for l in eachindex(A.inds)
            if l != rb && !(i > 1 && l == findleg(A; itag="bond,m,$(i-1)")) &&
               !(i == 1 && l == findleg(A; itag="SLeft"))]
    tags = i == 1 ? vcat(["SLeft"], phys) : vcat(["bond,m,$(i-1)"], phys)
    return vcat((collect(findlegs(M; itag=t)) for t in tags)...)
end

# Single-site sweep: forward exp(h·H_eff) on each site, backward exp(-h·K_eff)
# on the bond tensor between them. Bond dimension is fixed by the input MPS.
#
# The zero-site environment is built from the *untagged* SVD factor (legs still
# carry "svdL"/"svdR"), so the bond tensor's two legs stay distinguishable and
# K_eff contracts unambiguously with no lock juggling; the same tensor is then
# retagged into the environment stack, so nothing is contracted twice.
function _sweep_1s_tdvp!(MPS, MPO, Hrl; h, nKrylov, tol, lanczos=false)
    N = length(MPO)

    for i in 1:N-1                                     # left -> right
        M = _expm_krylov(A -> _applyH(Hrl[i], (MPO[i],), Hrl[i+2], A), MPS[i], h;
                         nKrylov, tol, lanczos)
        bt   = "bond,m,$i"
        rb   = findleg(M; itag=bt)
        left = Tuple(l for l in 1:length(M.inds) if l != rb)
        (; U, S, Vd) = truncated_svd(M, left)
        Lz = (Hrl[i] * U) * MPO[i] * lock(U'; itag="svdL")        # 0-site left env
        C  = _expm_krylov(A -> (Lz * A) * Hrl[i+2], S * Vd, -h; nKrylov, tol, lanczos)
        MPS[i]   = setitag(U, bt; itag="svdL")
        MPS[i+1] = setitag(C * MPS[i+1], bt; itag="svdL")
        Hrl[i+1] = setitag(Lz, bt; itag="svdL")
    end

    # turning point: site N takes both half-steps at once
    MPS[N] = _expm_krylov(A -> _applyH(Hrl[N], (MPO[N],), Hrl[N+2], A), MPS[N], 2h;
                          nKrylov, tol, lanczos)

    for i in N:-1:2                                    # right -> left
        bt = "bond,m,$(i-1)"
        lb = findleg(MPS[i]; itag=bt)
        (; U, S, Vd) = truncated_svd(MPS[i], (lb,))
        Rz = (Hrl[i+2] * Vd) * MPO[i] * lock(Vd'; itag="svdR")    # 0-site right env
        C  = _expm_krylov(A -> (Hrl[i] * A) * Rz, U * S, -h; nKrylov, tol, lanczos)
        MPS[i]   = setitag(Vd, bt; itag="svdR")
        Hrl[i+1] = setitag(Rz, bt; itag="svdR")
        MPS[i-1] = _expm_krylov(A -> _applyH(Hrl[i-1], (MPO[i-1],), Hrl[i+1], A),
                                setitag(MPS[i-1] * C, bt; itag="svdR"), h; nKrylov, tol, lanczos)
    end

    return 0.0
end

# Controlled bond expansion (Gleis, Li, von Delft, PRL 130, 246402; CBE-TDVP: Li, Gleis, von Delft 2024).
# Before each one-site update the bond the sweep heads into is enlarged by `dex` states taken from the
# discarded space, chosen by "shrewd selection" (supplement Table I) to carry the weight of H|ψ⟩ in
# Ā ⊗ B̄ — the part of the two-site tangent space a one-site update cannot reach — at one-site cost:
# no two-site tensor is ever formed. Kept and added states are orthonormal, so ψ is unchanged; the
# one-site update then evolves into the new directions and its SVD trims the bond back to Nkeep.
# Every step below uses only (D·d) × (D·w)-sized objects. Same scheme as QSMPSLib CBE1SiQS (which does the
# SVDs as Gram-matrix eigendecompositions): preselect D' = Nkeep/w, expand the bond to Nkeep + dex.

_nmult(q, l) = sum(last, q.spaces[l])                  # multiplets on leg l
# Preselection keeps D' = κ·Nkeep/w (QSMPSLib/paper: κ = 1; default here 3). At small Nkeep, D' of a few multiplets loses
# accuracy at low T; κ trades cost (steps c, d scale with κ) for selection quality.
const CBE_PRESELECT = Ref(3.0)   # 4x4 Hubbard D=32: κ=1 → ΔE/N≈1e-2 at β=8 vs two-site, κ=3 → 1.5e-4, +8% time

# Direct-sum the new block `T` onto `K` along bond `bt` (leg order and direction of K).
function _cbe_oplus(K, T, bt)
    k = findleg(K; itag=bt)
    T.inds[findleg(T; itag=bt)].dir == K.inds[k].dir || (T = legflip(T; itag=bt))
    perm = Tuple(findleg(T; itag=K.inds[l].itags, dir=K.inds[l].dir) for l in eachindex(K.inds))
    return to_concrete(oplus(K, to_concrete(permutedims(T, perm)), k))
end
_others(q, l) = Tuple(k for k in eachindex(q.inds) if k != l)

# Isometry factor (`side` = :U or :Vd) of q keeping only singular values > thr (absolute); `nothing` if none.
# Projected isometries have σ ∈ [0, 1]: σ ≈ 0 means "was inside the kept space", and normalizing such
# round-off would break orthogonality to it — exactly what happens once a bond is full (Ā = ∅).
function _cbe_iso(q, left, tl, tr, side; thr=1e-6)
    norm(q) <= thr && return nothing
    f = truncated_svd(q, left, tl, tr)
    k = count(x -> x[1] > thr, _weighted_singvals(f.S))
    k == 0 && return nothing
    return getfield(truncated_svd(q, left, tl, tr; Nkeep=k), side)
end

# right -> left, center MPS[i] = Λ B, left-orthonormal A = MPS[i-1]: truncated complement Ã ⊥ A on bond i-1.
function _cbe_expand_left!(MPS, MPO, Hrl, i, dex, Nkeep)
    A, C = MPS[i-1], MPS[i]; bt = "bond,m,$(i-1)"; lb = findleg(C; itag=bt)
    B  = truncated_svd(C, (lb,)).Vd                                     # C = Λ B
    Rt = (C * MPO[i]) * Hrl[i+2]                                        # Λ B W R
    Ro = Rt - (Rt * B') * B                                             # onto B̄
    norm(Ro) <= 1e-13 * norm(Rt) && return
    (; U, S) = truncated_svd(Ro, (findleg(Ro; itag=bt),), "cbe,a,l", "cbe,a,r")       # (a) canonicalize right side
    Lt = ((Hrl[i-1] * A) * MPO[i-1]) * (U * S)                          # L W A U S
    Lo = Lt - A * (A' * Lt)                                             # onto Ā
    norm(Lo) <= 1e-13 * norm(Lt) && return
    w  = _nmult(MPO[i-1], findleg(MPO[i-1]; itag="bond,w,$(i-1)"))
    r  = truncated_svd(Lo, _others(Lo, findleg(Lo; itag="cbe,a,r")), "cbe,b,l", "cbe,b,r";
                       Nkeep=max(1, ceil(Int, CBE_PRESELECT[] * Nkeep / w)))   # (b) preselect D' = κ Dmax/w
    us = r.U * r.S
    Ah = _cbe_iso(us, Tuple(_tdvp_left_legs(A, us, i - 1)), "cbe,c,l", "cbe,c,r", :U; thr=1e-12 * norm(us))  # (c) Â^pr
    Ah === nothing && return
    Ah = _cbe_iso(Ah - A * (A' * Ah), _others(Ah, findleg(Ah; itag="cbe,c,l")), "cbe,d,l", "cbe,d,r", :U)  # A†Â = 0
    Ah === nothing && return
    Ct = (((Hrl[i-1] * A) * MPO[i-1]) * Ah' * C * MPO[i]) * Hrl[i+2]   # Â† L W A Λ B W R
    Co = Ct - (Ct * B') * B
    norm(Co) <= 1e-13 * norm(Ct) && return
    u  = truncated_svd(Co, (findleg(Co; itag="cbe,d,l"),), "cbe,e,l", "cbe,e,r";
                       Nkeep=dex + max(0, Nkeep - _nmult(C, lb)), cutoff=1e-20).U   # (d) final: grow to Nkeep+dex
    Ax = _cbe_oplus(A, setitag(Ah * u, bt; itag="cbe,e,l"), bt)
    MPS[i]   = to_concrete(unlock(lock(Ax'; itag=bt) * (A * C); itag=bt))
    MPS[i-1] = Ax
    Hrl[i]   = (Hrl[i-1] * Ax) * MPO[i-1] * lock(Ax'; itag=bt)
end

# left -> right, center MPS[i] = A Λ, right-orthonormal B = MPS[i+1]: mirror image, complement B̃ ⊥ B on bond i.
function _cbe_expand_right!(MPS, MPO, Hrl, i, dex, Nkeep)
    C, B = MPS[i], MPS[i+1]; bt = "bond,m,$i"; rb = findleg(C; itag=bt)
    A  = truncated_svd(C, _others(C, rb)).U                              # C = A Λ
    Lt = (Hrl[i] * C) * MPO[i]                                           # L W A Λ
    Lo = Lt - A * (A' * Lt)                                              # onto Ā
    norm(Lo) <= 1e-13 * norm(Lt) && return
    (; S, Vd) = truncated_svd(Lo, _others(Lo, findleg(Lo; itag=bt)), "cbe,a,l", "cbe,a,r")   # (a)
    Rt = (((S * Vd) * B) * MPO[i+1]) * Hrl[i+3]                         # S V† B W R
    Ro = Rt - (Rt * B') * B                                              # onto B̄
    norm(Ro) <= 1e-13 * norm(Rt) && return
    w  = _nmult(MPO[i], findleg(MPO[i]; itag="bond,w,$i"))
    r  = truncated_svd(Ro, (findleg(Ro; itag="cbe,a,l"),), "cbe,b,l", "cbe,b,r";
                       Nkeep=max(1, ceil(Int, CBE_PRESELECT[] * Nkeep / w)))   # (b) preselect
    sv = r.S * r.Vd
    Bh = _cbe_iso(sv, (findleg(sv; itag="cbe,b,l"), findleg(sv; itag="bond,w,$i")), "cbe,c,l", "cbe,c,r", :Vd;
                  thr=1e-12 * norm(sv))                                  # (c) B̂^pr
    Bh === nothing && return
    Bh = _cbe_iso(Bh - (Bh * B') * B, (findleg(Bh; itag="cbe,c,r"),), "cbe,d,l", "cbe,d,r", :Vd)   # B̂ B† = 0
    Bh === nothing && return
    Ct = (Hrl[i] * C) * MPO[i] * ((((Hrl[i+3] * B) * MPO[i+1]) * Bh'))  # L W A Λ (B W R B̂†)
    Co = Ct - A * (A' * Ct)
    norm(Co) <= 1e-13 * norm(Ct) && return
    v  = truncated_svd(Co, (findleg(Co; itag="cbe,d,r"),), "cbe,e,l", "cbe,e,r";
                       Nkeep=dex + max(0, Nkeep - _nmult(C, rb)), cutoff=1e-20).U   # (d)
    Bx = _cbe_oplus(B, setitag(v * Bh, bt; itag="cbe,e,l"), bt)
    MPS[i]   = to_concrete(unlock((C * B) * lock(Bx'; itag=bt); itag=bt))
    MPS[i+1] = Bx
    Hrl[i+2] = (Hrl[i+3] * Bx) * MPO[i+1] * lock(Bx'; itag=bt)
end

# One-site TDVP sweep (as _sweep_1s_tdvp!) with a CBE step before every forward site update and an
# Nkeep/cutoff-truncated SVD after it.
function _sweep_cbe_tdvp!(MPS, MPO, Hrl, Nkeep; h, nKrylov, tol, cutoff, lanczos=false,
                          dex::Int=max(1, cld(Nkeep, 10)))
    N = length(MPO); mdw = 0.0

    for i in 1:N-1                                     # left -> right
        _cbe_expand_right!(MPS, MPO, Hrl, i, dex, Nkeep)
        M = _expm_krylov(A -> _applyH(Hrl[i], (MPO[i],), Hrl[i+2], A), MPS[i], h; nKrylov, tol, lanczos)
        bt = "bond,m,$i"; rb = findleg(M; itag=bt)
        (; U, S, Vd) = truncated_svd(M, Tuple(l for l in eachindex(M.inds) if l != rb); Nkeep, cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        Lz = (Hrl[i] * U) * MPO[i] * lock(U'; itag="svdL")
        C  = _expm_krylov(A -> (Lz * A) * Hrl[i+2], S * Vd, -h; nKrylov, tol, lanczos)
        MPS[i]   = setitag(U, bt; itag="svdL")
        MPS[i+1] = setitag(C * MPS[i+1], bt; itag="svdL")
        Hrl[i+1] = setitag(Lz, bt; itag="svdL")
    end

    _cbe_expand_left!(MPS, MPO, Hrl, N, dex, Nkeep)          # turning point: both half-steps on site N
    MPS[N] = _expm_krylov(A -> _applyH(Hrl[N], (MPO[N],), Hrl[N+2], A), MPS[N], 2h; nKrylov, tol, lanczos)

    for i in N:-1:2                                    # right -> left
        bt = "bond,m,$(i-1)"
        (; U, S, Vd) = truncated_svd(MPS[i], (findleg(MPS[i]; itag=bt),); Nkeep, cutoff)
        mdw = max(mdw, _dw(MPS[i], U, S, Vd))
        Rz = (Hrl[i+2] * Vd) * MPO[i] * lock(Vd'; itag="svdR")
        C  = _expm_krylov(A -> (Hrl[i] * A) * Rz, U * S, -h; nKrylov, tol, lanczos)
        MPS[i]   = setitag(Vd, bt; itag="svdR")
        Hrl[i+1] = setitag(Rz, bt; itag="svdR")
        MPS[i-1] = setitag(MPS[i-1] * C, bt; itag="svdR")
        i > 2 && _cbe_expand_left!(MPS, MPO, Hrl, i - 1, dex, Nkeep)
        MPS[i-1] = _expm_krylov(A -> _applyH(Hrl[i-1], (MPO[i-1],), Hrl[i+1], A), MPS[i-1], h;
                                nKrylov, tol, lanczos)
    end

    return mdw
end

function _print_tdvp_options(mode, N, dt, nsteps, Nkeep, cutoff, nKrylov, tol,
                             normalize, obs, save_path, save_option, init_energy)
    bar = "═"^65
    println("╔" * bar)
    @printf("║  TDVP   ·   mode = %-3s   ·   N = %d\n", mode, N)
    @printf("║  %s\n", Dates.format(now(), "yyyy-mm-dd (E)  HH:MM:SS"))
    println("╟" * "─"^65)
    @printf("║  init_energy = %.10f\n", init_energy)
    @printf("║  step exp(dt·H), dt = %s   ·   nsteps = %d   ·   T = %s\n",
            string(dt), nsteps, string(dt * nsteps))
    @printf("║  Nkeep = %-5s   cutoff (discarded weight) = %.1e\n",
            Nkeep === nothing ? "—" : string(Nkeep), cutoff)
    @printf("║  Krylov: nKrylov = %d, tol = %.1e   ·   normalize = %s\n",
            nKrylov, tol, normalize)
    obs === nothing || println("║  observables: ", join(first.(obs), ",  "))
    save_path === nothing ||
        println("║  saving ", save_option == "all" ? "every step → $save_path/" :
                                                  "final result → $save_path")
    println("╚" * bar)
    flush(stdout)
end

# `save`, when given, is a closure taking the finished record and returning the
# number of bytes it put on disk. It runs here rather than at the call site so
# the record is complete before it is serialized, and so its cost can be timed
# and printed inside the step block.
function _report_step!(history, si, nsteps, t, E, mdw, mbond, st, elapsed, MPS, N, obs,
                       verbose::Bool=true; save=nothing)
    rec = Dict{String,Any}("step"=>si, "t"=>t, "E"=>E, "dw"=>mdw, "chi"=>mbond, "time"=>st)
    if verbose
        eta = 0 < si < nsteps ? (elapsed / si) * (nsteps - si) : 0.0
        @printf("  ┌ step %3d/%3d   t = %-18s [%s]\n", si, nsteps, string(t),
                Dates.format(now(), "HH:MM:SS"))
        @printf("  │  E = %.10f\n", E)
        @printf("  │  %s  dw = %.2e      t %.2fs · elapsed %.1fs · ETA %.1fs\n",
                _chi_str(MPS, mbond), mdw, st, elapsed, eta)
    end
    if obs !== nothing                        # recorded even when quiet
        snap = TLMps(deepcopy(MPS), 1)        # measure without disturbing the TDVP gauge
        ov = Dict{String,Any}(); parts = String[]
        for (name, f) in obs
            v = f(snap); ov[name] = v
            push!(parts, @sprintf("%s = %s", name, _fmt_obs(v)))
        end
        verbose && println("  │  ", join(parts, "   "))
        rec["obs"] = ov
    end
    if save !== nothing
        stime = @elapsed bytes = save(rec)
        rec["save_time"] = stime; rec["save_bytes"] = bytes
        verbose && @printf("  │  saved %s in %.2fs  (%s/s)\n", _fmt_bytes(bytes), stime,
                           _fmt_bytes(stime > 0 ? bytes / stime : 0.0))
    end
    verbose && (println("  └"); flush(stdout))   # cluster stdout is block-buffered
    push!(history, rec)
    return history
end

function _fmt_bytes(n)
    n < 1024 && return @sprintf("%.0f B", n)
    for (u, d) in (("KB", 2.0^10), ("MB", 2.0^20), ("GB", 2.0^30))
        n < 1024 * d && return @sprintf("%.1f %s", n / d, u)
    end
    return @sprintf("%.2f TB", n / 2.0^40)
end

_fmt_obs(v::Real)    = @sprintf("%.6g", v)
_fmt_obs(v::Complex) = @sprintf("%.6g%+.6gim", real(v), imag(v))
_fmt_obs(v)          = string(v)

"""
    tdvp!(mps::TLMps, mpo::TLMpo, dt::Number, nsteps::Int, Nkeep=nothing;
          mode="2s", cutoff=0.0, nKrylov=20, tol=1e-12, normalize=false,
          obs=nothing, save_path=nothing, save_option="final", meta_out=false,
          verbose=true) -> (mps, E)

Time-dependent variational principle evolution, mutating `mps` in place. One
step applies `exp(dt·H)`; `nsteps` steps advance the state to `T = nsteps·dt`.
Returns `(mps, E)` with `E = ⟨H⟩` after the last step, or `(mps, E, meta)` when
`meta_out=true`. Use `tdvp` to run on a copy instead.

Sign convention matches `ITensorMPS.tdvp` — the generator is `dt·H`, so

- real time `|ψ(t)⟩ = e^{-iHt}|ψ⟩`  →  `dt = -im*Δt`
- imaginary time (cooling)          →  `dt = -Δτ`, usually with `normalize=true`

`mode` selects the update:
- `"2s"` two-site (default): the bond can grow, truncated by `Nkeep`/`cutoff`.
  Needed whenever the initial state is not already entangled enough (a product
  state under `"1s"` stays a product state).
- `"1s"` single-site: bond dimension is frozen at the input MPS's, no
  truncation error, and the evolution is exactly unitary/norm-preserving.
- `"cbe"` one-site TDVP with controlled bond expansion: before each site update the
  bond ahead grows by `cld(Nkeep, 10)` states from the discarded⊗discarded part of
  H|ψ⟩, then the update's SVD trims to `Nkeep`/`cutoff`. Bonds adapt like `"2s"`
  without the two-site Krylov.

Each step is a full left→right→left sweep applying `dt/2` per direction, i.e.
the standard second-order symmetric scheme, so the error per step is `O(dt³)`
and `O(dt²)` accumulated. `nKrylov`/`tol` control the Krylov exponential of the
local effective Hamiltonian; `cutoff` is the discarded weight (see
[`truncated_svd`](@ref)) and only bites in `"2s"`/`"cbe"`.

`lanczos=true` exponentiates with the Hermitian Lanczos recurrence instead of
Arnoldi with full re-orthogonalization (much cheaper with SU(2) TLArrays; H_eff is
Hermitian for any Hermitian MPO).

`normalize=true` renormalizes the state after every step (required for
imaginary time, harmless but unnecessary for real time).

Real time needs complex tensors: when `mps` holds real ones and `dt` is complex,
the promoted copy is what gets evolved and returned, and the input `mps` is left
at its initial state. Always keep the returned object — `mps, E = tdvp!(mps, …)`.

`obs` registers per-step measurements: a vector of `name => f` where `f(m::TLMps)`
returns a number, evaluated on a fresh snapshot. Per-step values are in
`metadata.history`, which is filled whether or not `verbose` printing is on.

`save_path` + `save_option` control what hits disk:

- `"final"` (default): `save_path` is a **file** holding `(metadata, mps, mpo)`,
  rewritten in place after every step, so it doubles as a checkpoint — a run
  killed mid-flight leaves the last completed step on disk, with
  `metadata.steps_completed` saying which one. `metadata.history` carries the
  per-step scalars.
- `"all"`: `save_path` is a **directory** (created if missing), and every time
  step's state is kept:

      save_path/step_0.jls      (metadata = t=0 history record + metadata["run"]
                                 = run parameters, mps = t=0 state, mpo)
      save_path/step_1.jls      (metadata = that step's history record, mps)
      ...
      save_path/final.jls       (metadata, mps, mpo)  — same as "final" mode

  `metadata.history` starts at the t=0 record (`step = 0`), so trajectories run
  `t = 0 … dt·nsteps`.

  Disk cost is `nsteps ×` one MPS, so watch `nsteps · N · χ²`.

    obs = ["Sz1" => m->observe(m, Sz, 1)]
    tdvp!(mps, mpo, -0.05im, 40, 100; obs=obs, cutoff=1e-10)
    tdvp!(mps, mpo, -0.05im, 40, 100; save_path="run1", save_option="all")

    traj = [deserialize(joinpath("run1", f)).mps
            for f in sort(filter(startswith("step_"), readdir("run1")))]
"""
function tdvp!(mps::TLMps, mpo::TLMpo, dt::Number, nsteps::Int,
               Nkeep::Union{Nothing,Int}=nothing;
               mode::AbstractString="2s", cutoff::Float64=0.0, nKrylov::Int=20,
               tol::Float64=1e-12, normalize::Bool=false, obs=nothing,
               save_path=nothing, save_option::AbstractString="final",
               meta_out::Bool=false, verbose::Bool=true, lanczos::Bool=false)
    MPO = mpo.TLA
    N = length(MPO); @assert length(mps.TLA) == N
    mode in ("1s", "2s", "cbe") ||
        throw(ArgumentError("mode must be \"1s\", \"2s\", or \"cbe\"; got \"$mode\""))
    mode == "cbe" && Nkeep === nothing &&
        throw(ArgumentError("mode=\"cbe\" needs an integer Nkeep"))
    save_option in ("final", "all") ||
        throw(ArgumentError("save_option must be \"final\" or \"all\"; got \"$save_option\""))
    save_option == "all" && save_path === nothing &&
        throw(ArgumentError("save_option=\"all\" needs save_path (a directory)"))
    save_steps = save_option == "all"

    move_canonical!(mps, 1)                    # TDVP sweeps start at the left edge
    RT = promote_type(eltype(mps.TLA[1]), typeof(dt))
    if RT !== eltype(mps.TLA[1])               # real tensors, complex dt: promote once
        mps = TLMps(TLArray[to_concrete(A * one(RT)) for A in mps.TLA], mps.oc)
    end
    MPS = mps.TLA
    Hrl = _tdvp_prep!(MPS, MPO)
    measure_energy = length(MPS[1].inds) == 3
    init_energy = measure_energy ? _tdvp_energy(MPS, MPO, Hrl) : NaN
    verbose && _print_tdvp_options(mode, N, dt, nsteps, Nkeep, cutoff, nKrylov, tol,
                                   normalize, obs, save_path, save_option, init_energy)

    h = dt / 2
    history = Dict{String,Any}[]
    E = init_energy; t0 = time()
    # "final" mode rewrites its single file every step, so it is also a checkpoint.
    save_final = save_path !== nothing && !save_steps
    save_final && mkpath(dirname(abspath(save_path)))
    _meta(si, E, elapsed) =
        (; N, dt, nsteps, Nkeep, mode, cutoff, nKrylov, tol, normalize, obs, save_path,
         save_option, init_energy, final_energy=E, total_time=dt*si, elapsed,
         steps_completed=si, history)

    save_steps && mkpath(save_path)
    total_bytes = 0.0
    # step 0 = run header + t=0 state/obs + mpo
    save0 = !save_steps ? nothing : function (rec)
        rec0 = copy(rec)                       # same Dict shape as every other step file
        rec0["run"] = (; N, dt, nsteps, Nkeep, mode, cutoff, nKrylov, tol, normalize,
                       init_energy,
                       obs_names=obs === nothing ? String[] : String.(first.(obs)))
        f = joinpath(save_path, "step_0.jls")
        serialize(f, (; metadata=rec0, mps=deepcopy(mps), mpo))
        return filesize(f)
    end
    _report_step!(history, 0, nsteps, zero(dt), init_energy, 0.0, _maxbond(MPS),
                  0.0, 0.0, MPS, N, obs, verbose; save=save0)   # t=0 record heads the history
    total_bytes += get(history[end], "save_bytes", 0.0)
    for si in 1:nsteps
        st = @elapsed begin
            mdw = mode == "2s" ?
                _sweep_2s_tdvp!(MPS, MPO, Hrl, Nkeep; h, nKrylov, tol, cutoff, lanczos) :
                mode == "cbe" ?
                _sweep_cbe_tdvp!(MPS, MPO, Hrl, Nkeep; h, nKrylov, tol, cutoff, lanczos) :
                _sweep_1s_tdvp!(MPS, MPO, Hrl; h, nKrylov, tol, lanczos)
            normalize && (MPS[1] = to_concrete(MPS[1] / norm(MPS[1])))
            measure_energy && (E = _tdvp_energy(MPS, MPO, Hrl))
        end
        mps.oc = 1
        saver = save_steps ? (rec -> (f = joinpath(save_path, "step_$(si).jls");
                                      serialize(f, (; metadata=rec, mps=deepcopy(mps)));
                                      filesize(f))) :
                save_final ? (_ -> (serialize(save_path,
                                        (; metadata=_meta(si, E, time()-t0), mps, mpo));
                                    filesize(save_path))) : nothing
        _report_step!(history, si, nsteps, dt * si, E, mdw, _maxbond(MPS), st,
                      time()-t0, MPS, N, obs, verbose; save=saver)
        total_bytes += get(history[end], "save_bytes", 0.0)
    end
    elapsed = time() - t0
    verbose && (@printf("  ▸ done: T = %s   E = %.10f   (%.1fs total)\n",
                        string(dt * nsteps), E, elapsed); flush(stdout))
    if verbose && save_path !== nothing
        save_total = sum(r -> get(r, "save_time", 0.0), history; init=0.0)
        @printf("  ▸ saved: %s written in %.1fs (%.1f%% of wall)%s\n",
                _fmt_bytes(total_bytes), save_total, 100 * save_total / max(elapsed, eps()),
                save_steps ? @sprintf("  ·  %d files in %s/", length(history), save_path) :
                             "  ·  1 file, overwritten each step")
        flush(stdout)
    end
    mps.oc = 1
    meta = meta_out || save_path !== nothing ? _meta(nsteps, E, elapsed) : nothing
    if save_path !== nothing
        f = save_steps ? joinpath(save_path, "final.jls") : save_path
        mkpath(dirname(abspath(f)))           # save_path's folder need not exist yet
        serialize(f, (; metadata=meta, mps, mpo))
    end
    return meta_out ? (mps, E, meta) : (mps, E)
end

"""
    tdvp(mps::TLMps, mpo::TLMpo, dt, nsteps, Nkeep=nothing; kwargs...)

Non-mutating [`tdvp!`](@ref): evolves a deep copy of `mps`.
"""
tdvp(mps::TLMps, mpo::TLMpo, dt::Number, nsteps::Int, Nkeep=nothing; kwargs...) =
    tdvp!(deepcopy(mps), mpo, dt, nsteps, Nkeep; kwargs...)
