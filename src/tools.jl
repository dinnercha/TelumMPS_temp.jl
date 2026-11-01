
function init_MPS(MPO::Vector{<:TLArray}, Nkeep::Int, Nkeep_last::Int=1; tol=0.0,
                  target=nothing)
    N = length(MPO); MPS = Vector{TLArray{Float64, 3}}(undef, N)
    zq = zero_qlabels(MPO[1])
    Aprev = getvac(MPO[1], ("SLeft", "SLeft"))
    Hprev = addSingleton(Aprev, 3; itag="dummy_l", dir='-')
    E, sp = nothing, nothing

    # Identity-closure vectors: rv[i] is the trace component of MPO[i+1..N]
    # on bond i, so that Hnow*rv[i] is the block Hamiltonian of sites 1..i in
    # ANY MPO gauge. For FSM-gauge MPOs rv[i] = e_1 and this reproduces the
    # old first-vacuum-copy extraction; SVD-compressed MPOs (auto_mpo) rotate
    # the bond vacuum sector, where the old extraction picks a wrong operator
    # and strands DMRG in a wrong charge sector.
    tracephys(i) = getIdentity((MPO[i], findleg(MPO[i]; itag="site,$i", dir='-'));
                               itag="site,$i") * MPO[i]
    rv = Vector{TLArray}(undef, max(N - 1, 0))
    if N > 1
        TN = tracephys(N)
        dl = findleg(TN; itag="dummy_r")
        rv[N-1] = deleteSingleton(getsub(TN, dl, x -> x == zq ? 1 : nothing), dl)
        for i in N-1:-1:1
            i < N - 1 && (rv[i] = tracephys(i+1) * rv[i+1])
            nr = norm(rv[i])
            # nr ≈ 0: no term of H acts as identity on sites i+1..N (e.g.
            # traceless H with no term completed left of bond i). The block
            # Hamiltonian of 1..i is then exactly zero — keep rv = 0 so
            # Hmat = 0 (degenerate step, matches the FSM-gauge behavior).
            # A genuine identity channel gives nr = O(d^(N-i)) ≫ 1e-8, while
            # SVD-compression junk sits at ~1e-16.
            rv[i] = nr < 1e-8 ? 0.0 * rv[i] : rv[i] / nr
        end
    end

    for i=1:N
        li = findleg(Aprev; itag=i==1 ? "SLeft" : "bond,m", dir='-')
        # Locate the ket ("site,i", dir '-') leg by tag: SVD-compressed MPOs
        # come out with the bond legs first, so leg 2 is not always physical.
        pl = findleg(MPO[i]; itag="site,$i", dir='-')
        Anow = getIdentity((Aprev, li), (MPO[i], pl); itag="SL,$i")
        Hnow = Anow' * lock(Anow * Hprev * MPO[i]; itag="SL,$i")

        if i == N
            bli = findleg(Hnow; itag="dummy_r")
            Hmat = deleteSingleton(getsub(Hnow, bli, x->x==zq ? 1 : nothing), bli)
            # Canonical ensemble: keep only the target global qlabel sector on the
            # last state leg, so the MPS right boundary carries a fixed charge
            # (and spin) instead of the variationally lowest one.
            if !isnothing(target)
                # preserve_space=true: only the sector data is filtered, the cached
                # leg-space list stays intact so `Anow * ek.V` still matches legs.
                Hmat = getsub(Hmat, x -> x == target ? Colon() : nothing;
                              itag="SL,$i", preserve_space=true)
                # nsectors counts nonzero sectors — the old dropped-sector count.
                Telum.nsectors(Hmat) == 0 && error(
                    "init_MPS: target sector $target absent at the last bond; " *
                    "raise Nkeep (currently $Nkeep) or check the target qlabel.")
            end
        else
            Hmat = Hnow * rv[i]
        end
        e = eigen((Hmat + Hmat') / 2)

        ek, _ = discard_eigen(e, i==N ? Nkeep_last : Nkeep, i==N ? "dummy_r" : "bond,m,$i", "SD,$i"; tol)
        MPS[i] = Aprev = Anow * ek.V
        if i < N Hprev = lock(ek.V * Hnow; itag="bond,m") * ek.V'
        else E, sp = ek.eig_list[1][1], ek.eig_list[1][3] end
    end
    return TLMps(MPS, N), E, sp
end

# Distance between two qlabels: sum of |ΔDynkin| over all symmetries. Each site
# can move a bond label by at most `step` (below), which is what makes the
# feasibility prune in `_walk_sector` work for U1, SU(2) and their products.
_qdist(a, b) = sum(sum(abs, ai .- bi) for (ai, bi) in zip(a, b))

# Depth-first walk building the χ=1 fusion path. At site i the reachable bond
# sectors are exactly `Anow.spaces[3]` — the fusion of the current bond with the
# local space — so no symmetry algebra is needed here, Telum's `getIdentity`
# supplies the fusion rule. Candidates too far from the target to be recovered
# by the remaining sites are pruned; the rest are tried nearest-first, and the
# walk backtracks on a dead end (parity traps: e.g. an all-spin-1/2 chain cannot
# sit on total spin 0 with an odd number of sites left).
function _walk_sector(MPO, i::Int, Aprev, target, step::Int, MPS)
    N = length(MPO)
    li = findleg(Aprev; itag = i == 1 ? "SLeft" : "bond,m", dir='-')
    pl = findleg(MPO[i]; itag="site,$i", dir='-')
    Anow = getIdentity((Aprev, li), (MPO[i], pl); itag="SL,$i")
    r = N - i
    cands = sort([a for a in first.(Anow.spaces[3]) if _qdist(a, target) <= r * step];
                 by = a -> _qdist(a, target))
    for a in cands
        A = to_concrete(setitag(getsub(Anow, 3, x -> x == a ? Colon() : nothing),
                                3, i == N ? "dummy_r" : "bond,m,$i"))
        MPS[i] = A
        (i == N || _walk_sector(MPO, i + 1, A, target, step, MPS)) && return true
    end
    return false
end

"""
    init_MPS_sector(MPO, target) -> TLMps

Bond-dimension-1 MPS sitting exactly in the global `target` qlabel sector, built
by fusing the local spaces left to right along a single fusion path — no
Hamiltonian, no eigensolve, so no `Nkeep` to compete with the sector (which is
what forces `init_MPS(...; target=...)` to run at a large `Nkeep`).

Every tensor is an isometry restricted to one output sector, so the chain comes
out left-canonical (`oc = N`). Use it as a DMRG seed; being χ=1 it is a colder
start than `init_MPS`, so prefer `mode="2s"` (or the `1s+` subspace expansion).
"""
function init_MPS_sector(MPO::Vector{<:TLArray}, target)
    N = length(MPO)
    N >= 1 || throw(ArgumentError("init_MPS_sector needs at least one site"))
    zq = zero_qlabels(MPO[1])
    pl1 = findleg(MPO[1]; itag="site,1", dir='-')
    step = maximum(_qdist(first(s), zq) for s in MPO[1].spaces[pl1])
    MPS = Vector{TLArray{Float64, 3}}(undef, N)
    Aprev = getvac(MPO[1], ("SLeft", "SLeft"))
    _walk_sector(MPO, 1, Aprev, target, step, MPS) || error(
        "init_MPS_sector: no fusion path reaches target $target on $N sites " *
        "(check the qlabel, its parity, and that it is within reach of the local space)")
    return TLMps(MPS, N)
end

init_MPS_sector(MPO::TLMpo, target) = init_MPS_sector(MPO.TLA, target)

# Convenience: accept a TLMpo directly.
init_MPS(MPO::TLMpo, Nkeep::Int, Nkeep_last::Int=1; tol=0.0, target=nothing) =
    init_MPS(MPO.TLA, Nkeep, Nkeep_last; tol=tol, target=target)

"""
    truncated_svd(q, left, tags...; cutoff=0.0, Nkeep=nothing)

`svd` truncated by *discarded weight*, the criterion ITensor's `cutoff` uses:

    dw = Σ_{i>k} d_i σ_i² / Σ_j d_j σ_j²  =  1 - Σ_{i≤k} λ_i  ≤  cutoff

where `λ_i` are the reduced-density-matrix eigenvalues at the cut and `d_i` is
the irrep dimension of the sector each singular value sits in — so an SU(2)
multiplet is counted with its `2S+1` degeneracy, as it must be, since that
eigenvalue really occurs `2S+1` times. For abelian symmetries every `d_i = 1`.

This is NOT Telum's own `svd` cutoff, which keeps `σ > cutoff·σ_max`. The two
are far apart in practice: on an L=50 Heisenberg mid-bond, `1e-12` means 622
kept values under Telum's rule and about 92 under this one. Every `cutoff`
throughout TelumMPS is this discarded-weight one, so the same number handed
to Telum and to ITensor now truncates in the same place.

`Nkeep` caps the count as well (degeneracy ignored, matching Telum); whichever
of the two binds first wins. `cutoff = 0` keeps everything the decomposition
produces and costs nothing extra.
"""
function truncated_svd(q, left, tags...; cutoff::Float64=0.0, Nkeep=nothing)
    cutoff <= 0 && return svd(q, left, tags...; cutoff=0.0, Nkeep=Nkeep)
    # ponytail: decompose once to read the spectrum, once more to cut at k. The
    # SVD is not the dominant cost of a two-site update; if it ever is, slice
    # U/S/Vd in place with getsub instead of re-decomposing.
    full = svd(q, left, tags...; cutoff=0.0, Nkeep=nothing)
    sv = _weighted_singvals(full.S)
    isempty(sv) && return full
    tot = sum(d * s^2 for (s, d) in sv)
    k = 1                                   # everything discardable -> keep one
    if tot > 0
        acc = 0.0
        for i in length(sv):-1:1            # accumulate from the tail
            acc += sv[i][2] * sv[i][1]^2
            if acc > cutoff * tot           # element i cannot be dropped
                k = i
                break
            end
        end
    end
    Nkeep === nothing || (k = min(k, Nkeep))
    return svd(q, left, tags...; cutoff=0.0, Nkeep=k)
end

# (σ, irrep dimension) for every singular value of an SVD's `S`, descending. σ is the
# full singular value, carried by each of the d states of the multiplet: Telum's CGTs are
# normalised (a 2-leg CGT is δ/√d), so the stored reduced entry is σ√d.
function _weighted_singvals(S)
    syms = symm(S)
    out = Tuple{Float64, Float64}[]
    for si in Telum.sector_slots(S)
        Telum.is_sector_active(S, si) || continue
        d = Float64(_irrepdim(syms, Telum.sector_qlabel(S, si, 1)))
        rmt = S.RMTs[si]
        for j in 1:size(rmt, 1)
            push!(out, (abs(rmt[j, j]) / sqrt(d), d))
        end
    end
    sort!(out; by = first, rev = true)
    return out
end

# Bond tag between MPS sites i and i+1 (matches the TLMps construction scheme).
_mps_bond(i::Int) = "bond,m,$i"

"""
    _cap_flux_fix(cap, AN, bondtag) -> cap

Rescale the right-boundary environment cap so it closes the bra/ket boundary
legs exactly the way `overlap` does (`A[N]' * A[N]`).

Both `_getHrl` and `_tdvp_prep!` build that cap as `getIdentity` on the boundary
leg followed by a `legflip`. The flip is structural — without it the cap's second
leg has the wrong direction and will not contract the bra — but flipping a leg
that carries a *genuine* irrep drags along that irrep's 1j factor ((-1)^{2j} for
SU(2)), and nothing in the recursion cancels it. On an ordinary MPS the boundary
is the 1-dim vacuum, the flip is the identity and this never mattered. On a state
with nonzero total flux — anything `apply_op` returns for a rank-3 IROP, e.g.
`c|GS⟩` for a Green's function — every environment came out with a spurious
factor: measured -1 for a j=1/2 (one-electron) flux, i.e. `⟨φ|Σn_i|φ⟩ = -(N-1)`
while `observe`/`overlap` (which never build a cap) were exact.

The factor is measured rather than derived so this stays symmetry-agnostic: build
the boundary contraction with the cap and without it, and take the ratio. A
vacuum boundary gives `s == 1` exactly, so flux-free runs are unchanged.
"""
function _cap_flux_fix(cap::AbstractTLArray, AN::AbstractTLArray, bondtag::AbstractString)
    dr = findleg(cap; itag="dummy_r")
    dr === nothing && return cap
    bra = lock(AN'; itag=bondtag)                  # keep the bond leg open on the bra side
    ref = to_concrete(bra * AN)                    # what `overlap` contracts at site N
    vraw = to_concrete(bra * (cap * AN))           # what the environment does
    via = to_concrete(deleteSingleton(vraw, findleg(vraw; itag="dummy_r")))
    nr, nv = norm(ref), norm(via)
    (nr == 0 || nv == 0) && return cap
    s = nv / nr
    norm(via - s * ref) > norm(via + s * ref) && (s = -s)
    return isapprox(s, 1; atol=1e-12) ? cap : to_concrete(cap / s)
end

# Left-canonicalize site i (SVD), absorb S*Vd into site i+1. Center i -> i+1.
# `bond` maps bond index -> itag, so the same movers serve MPS and MPO chains.
function _move_right!(A::Vector{<:TLArray}, i::Int; cutoff, Nkeep, bond::Function=_mps_bond)
    bt = bond(i)
    q = A[i]
    rb = findleg(q; itag=bt)                       # right bond -> site i+1
    left = Tuple(l for l in 1:length(q.inds) if l != rb)
    (; U, S, Vd) = truncated_svd(q, left; cutoff=cutoff, Nkeep=Nkeep)
    A[i]   = setitag(U, bt; itag="svdL")           # new left-canonical site
    nb     = (S * Vd) * A[i+1]                      # old bond bt consumed; svdL carried
    A[i+1] = setitag(nb, bt; itag="svdL")
    return A
end

# Right-canonicalize site i (SVD), absorb U*S into site i-1. Center i -> i-1.
function _move_left!(A::Vector{<:TLArray}, i::Int; cutoff, Nkeep, bond::Function=_mps_bond)
    bt = bond(i-1)
    q = A[i]
    lb = findleg(q; itag=bt)                        # left bond -> site i-1
    (; U, S, Vd) = truncated_svd(q, (lb,); cutoff=cutoff, Nkeep=Nkeep)
    A[i]   = setitag(Vd, bt; itag="svdR")           # new right-canonical site
    nb     = A[i-1] * (U * S)                        # old bond bt consumed; svdR carried
    A[i-1] = setitag(nb, bt; itag="svdR")
    return A
end

"""
    move_canonical!(mps::TLMps, j; cutoff=0.0, Nkeep=nothing) -> TLMps

Move the orthogonality (canonical) center to site `j` by sequential SVD gauge
steps, mutating `mps` in place (both `mps.TLA` and `mps.oc`). Returns `mps`.

If `mps.oc == 0` (undefined) the chain is first left-canonicalized 1→N to
establish a center at N. `cutoff`/`Nkeep` are forwarded to `svd`; the defaults
drop only numerical-zero singular values (effectively an exact gauge move).
"""
function move_canonical!(mps::TLMps, j::Integer;
                         cutoff::Float64=0.0, Nkeep::Union{Nothing, Int}=nothing)
    A = mps.TLA; N = length(A)
    1 <= j <= N || throw(ArgumentError("target center $j out of range 1:$N"))
    oc = mps.oc
    if oc == 0
        for i in 1:N-1
            _move_right!(A, i; cutoff=cutoff, Nkeep=Nkeep)
        end
        oc = N
    end
    while oc < j; _move_right!(A, oc; cutoff=cutoff, Nkeep=Nkeep); oc += 1; end
    while oc > j; _move_left!(A, oc; cutoff=cutoff, Nkeep=Nkeep); oc -= 1; end
    mps.oc = j
    return mps
end

# Retag an operator's physical ("site") legs onto MPS site i, leaving any IROP
# ("op") leg untouched.
_op_at(op::AbstractTLArray, i::Int) = replaceitag(op, "site" => "site,$i")

"""
    apply_op(mps::TLMps, op, i; cutoff=0.0, Nkeep=nothing, target=nothing, jw=nothing)
        -> TLMps

Apply a local rank-2 scalar operator or rank-3 IROP at site `i` without
mutating `mps`. A rank-2 operator leaves the canonical center at `i`. For a
rank-3 IROP, the open operator leg is carried to the right by SVD and fused
into the right boundary, leaving an ordinary rank-3 MPS with center `N`.

`target`, when given, selects one physical (unflipped) qlabel sector of the
fused right boundary. Normalization is the caller's responsibility.

`jw` is the local fermion-parity operator (`q.Z`), and is **required for a
fermionic rank-3 `op`**. Fusing the operator leg into the right boundary places
the created/annihilated mode last in the Jordan-Wigner ordering, while `op` acted
at site `i`; moving it there costs the string `∏_{k>i} Z_k`. Without it the
returned MPS is `(∏_{k>i} Z_k)·op_i|ψ⟩`, not `op_i|ψ⟩`. That difference is
invisible to `overlap` and to `observe` of any parity-even local operator (the
strings cancel between bra and ket), which is why it hid for so long — but it
flips the sign of every Hamiltonian hopping term that touches site `i`, so a
Green's function built from an undressed `c|GS⟩` evolves under the wrong `H`.
Leave `jw=nothing` for a bosonic IROP (`q.S`): there is no string to apply.
`Z` is unitary and diagonal, so the canonical form survives the dressing.
"""
function apply_op(mps::TLMps, op::AbstractTLArray, i::Integer;
                  cutoff::Float64=0.0, Nkeep::Union{Nothing, Int}=nothing,
                  target=nothing, jw=nothing)
    N = length(mps)
    1 <= i <= N || throw(ArgumentError("site $i out of range 1:$N"))
    site = Int(i)
    rank = length(op.inds)
    rank in (2, 3) || throw(ArgumentError(
        "apply_op takes a rank-2 scalar operator or rank-3 IROP; got rank $rank"))

    ψ = deepcopy(mps)
    move_canonical!(ψ, site; cutoff=cutoff, Nkeep=Nkeep)
    A = TLArray[to_concrete(a) for a in ψ.TLA]

    if rank == 2
        A[site] = to_concrete(_op_at(op, site) * A[site])
        return TLMps(A, site)
    end

    flux_tag = "apply_op_flux"
    sitelegs = findlegs(op; itag="site")
    length(sitelegs) == 2 || throw(ArgumentError(
        "rank-3 apply_op requires two physical legs tagged site"))
    flux_leg = only(setdiff(1:3, sitelegs))
    B = to_concrete(setitag(_op_at(op, site), flux_leg, flux_tag) * A[site])
    for k in site:N-1
        leftleg = k == 1 ? findleg(B; itag="SLeft") : findleg(B; itag=_mps_bond(k - 1))
        k == 1 && leftleg === nothing && (leftleg = findleg(B; itag="dummy_l"))
        leftleg === nothing && error("apply_op: left bond leg not found at site $k")
        left = (leftleg, findleg(B; itag="site,$k"))
        bt = _mps_bond(k)
        (; U, S, Vd) = truncated_svd(B, left, "$bt,applyL", "$bt,applyR";
                                     cutoff=cutoff, Nkeep=Nkeep)
        A[k] = removeitag(U, "applyL")
        B = removeitag(to_concrete((S * Vd) * A[k+1]), "applyL")
    end

    boundary_tag = findleg(B; itag="dummy_r") !== nothing ? "dummy_r" : "SRight"
    boundary = findleg(B; itag=boundary_tag)
    boundary === nothing && error("apply_op: right boundary leg not found")
    boundary_itag = String(B.inds[boundary].itags)
    old_dir = B.inds[boundary].dir
    fuse_tag = "apply_op_boundary"
    F = getIdentity((B, findleg(B; itag=flux_tag)), (B, boundary); itag=fuse_tag)
    A[N] = setitag(to_concrete(B * F), boundary_itag; itag=fuse_tag)

    if target !== nothing
        new_boundary = findleg(A[N]; itag=boundary_itag)
        A[N] = to_concrete(getsub(A[N], new_boundary,
            qlabel -> qlabel == target ? Colon() : nothing))
        Telum.nsectors(A[N]) > 0 || throw(ArgumentError(
            "target sector $target is absent from the fused right boundary"))
    end
    new_boundary = findleg(A[N]; itag=boundary_itag)
    A[N].inds[new_boundary].dir == old_dir || (A[N] = legflip(A[N], new_boundary))

    # Jordan-Wigner string for the mode that just got parked at the right
    # boundary: it is ordered after every site, `op` acted at `site`, so the
    # sites it was carried across pick up a parity factor each. Z is diagonal
    # and unitary, so left-canonicality of sites site+1..N-1 is preserved.
    if jw !== nothing
        for k in site+1:N
            A[k] = to_concrete(_op_at(jw, k) * A[k])
        end
    end
    return TLMps(A, N)
end

# Right-boundary leg of the last MPS tensor: the one carrying the state's total
# symmetry flux (vacuum for an ordinary MPS, nontrivial after a rank-3 apply_op).
function _boundary_leg(A::AbstractTLArray)
    l = findleg(A; itag="dummy_r")
    l === nothing && (l = findleg(A; itag="SRight"))
    l === nothing && error("no right boundary leg (dummy_r / SRight) found")
    return l
end

_bonddir(A::AbstractTLArray, k::Int) = A.inds[findleg(A; itag=_mps_bond(k))].dir

function _boundary_qlabel(mps::TLMps)
    A = mps.TLA[end]
    l = _boundary_leg(A)
    length(A.spaces[l]) == 1 || throw(ArgumentError(
        "right boundary carries $(length(A.spaces[l])) sectors, so the total " *
        "flux is ambiguous"))
    return A.spaces[l][1][1]
end

"""
    overlap(a::TLMps, b::TLMps) -> Number

`⟨a|b⟩`, contracted site by site. Neither state is mutated and no canonical form
is assumed — the result is gauge invariant.

Both states may carry a nontrivial total flux (as produced by a rank-3
[`apply_op`](@ref)); their boundary legs then contract to a singlet. States in
different flux sectors are orthogonal, so `0` is returned without contracting.
"""
function overlap(a::TLMps, b::TLMps)
    N = length(a)
    N == length(b) || throw(ArgumentError(
        "length mismatch: $(N) vs $(length(b))"))
    A, B = a.TLA, b.TLA
    _boundary_qlabel(a) == _boundary_qlabel(b) ||
        return zero(promote_type(eltype(A[1]), eltype(B[1])))
    # Bra and ket bond legs are paired by itag, so the two states must share the
    # bond-direction convention — a bond left of the canonical center carries the
    # flipped one (Telum's svd always hands out '-'). Re-gauge copies to the right
    # edge when they disagree; both states are left untouched.
    if any(k -> _bonddir(A[k], k) != _bonddir(B[k], k), 1:N-1)
        a = move_canonical!(deepcopy(a), N)
        b = move_canonical!(deepcopy(b), N)
        A, B = a.TLA, b.TLA
    end
    E = lock(A[1]'; itag=_mps_bond(1)) * B[1]
    for k in 2:N-1
        E = lock(A[k]'; itag=_mps_bond(k)) * unlock(E * B[k]; itag=_mps_bond(k-1))
    end
    return (A[N]' * unlock(E * B[N]; itag=_mps_bond(N-1)))[]
end

"""
    observe(mps::TLMps, op, i; cutoff=0.0, Nkeep=nothing) -> Number

Expectation value `⟨ψ|op_i|ψ⟩` of a single on-site operator at site `i`, for
a rank-2 scalar operator or a rank-3 IROP.

Rank 2 (scalar/singlet operator): the plain expectation value. The canonical
center is moved to `i` first, so the rest of the chain contracts to the identity
(no zip-up error accumulation). Mutates `mps` (center position).

Rank 3 (IROP, e.g. an SU(2) spin operator `q.S`): the open symmetry leg cannot
close against an identity environment, so what is returned is the Wigner-Eckart
**reduced** matrix element `⟨ψ‖op_i‖ψ⟩` — the operator's flux is absorbed by the
state's own total flux (`op_i|ψ⟩` is projected back onto `ψ`'s sector). Hence:

- on a flux-free state (e.g. an SU(2) singlet ground state) it is exactly `0`,
  which is the selection rule, not a failure;
- on a flux-carrying state — one built by `apply_op(ψ, q.S, j)` — it is nonzero,
  and `Σ_i ⟨ψ‖S_i‖ψ⟩` reproduces the reduced matrix element of the total spin
  (`-√2` in the `S = 1` sector), independent of `ψ` within that sector.

It is *not* `⟨S^z_i⟩`, which is not an SU(2) invariant and carries an extra
Clebsch factor. For rotationally invariant quantities use [`observe_cor`](@ref)
(`observe_cor(ψ, q.S, i, q.S', j)` = `⟨S_i·S_j⟩`). The rank-3 path runs an
`apply_op` sweep from `i` to `N` and leaves `mps` untouched.
"""
function observe(mps::TLMps, op::AbstractTLArray, i::Integer; cutoff::Float64=0.0, Nkeep=nothing)
    rank = length(op.inds)
    rank in (2, 3) || throw(ArgumentError(
        "observe takes a rank-2 (scalar) operator or a rank-3 IROP; got rank $rank."))
    if rank == 3
        φ = try
            apply_op(mps, op, i; cutoff=cutoff, Nkeep=Nkeep, target=_boundary_qlabel(mps))
        catch e
            # Selection rule: the operator's flux cannot bring the state back to
            # its own sector (IROP on a singlet), so the element is exactly zero.
            (e isa ArgumentError && occursin("target sector", e.msg)) || rethrow()
            return zero(eltype(mps.TLA[i]))
        end
        return overlap(φ, mps)
    end
    move_canonical!(mps, i; cutoff=cutoff, Nkeep=Nkeep)
    A = mps.TLA[i]
    return (A' * (_op_at(op, i) * A))[]
end

"""
    observe_cor(mps::TLMps, op1, i, op2, j; cutoff=0.0, Nkeep=nothing) -> Float64

Two-point correlation `⟨ψ| op1_i op2_j |ψ⟩`. For an SU(2) spin correlation pass
the IROP and its conjugate so the two operator legs contract to a singlet (the
`S_i · S_j` dot product), e.g. `observe_cor(mps, q.S, i, q.S', j)`.

Moves the canonical center to `min(i,j)`; sites outside `[min,max]` then
contract to the identity. The window is swept by a transfer matrix that keeps
the running tensor at rank 3 (ket bond, bra bond, operator leg), so cost stays
flat in `|i-j|` rather than growing. Handles `i == j` (on-site, e.g.
`⟨S_i·S_i⟩`). Mutates `mps`.
"""

function observe_cor(mps::TLMps, op1::AbstractTLArray, i::Integer,
                     op2::AbstractTLArray, j::Integer; cutoff::Float64=0.0, Nkeep=nothing)
    a, b = minmax(i, j)
    move_canonical!(mps, a; cutoff=cutoff, Nkeep=Nkeep)
    A = mps.TLA
    if a == b
        return (A[a]' * (_op_at(op2, j) * (_op_at(op1, i) * A[a])))[]
    end
    opa = a == i ? op1 : op2                         # operator sitting at the near site
    opb = b == i ? op1 : op2                         # operator sitting at the far site
    # Site a: insert opa, close the left bond (identity), keep the right bond open.
    # Lock the bra's right bond so the ket-chain contraction stays unambiguous.
    E = lock(A[a]'; itag=_mps_bond(a)) * (_op_at(opa, a) * A[a])
    for k in a+1:b-1                                 # bare transfer (identity sites)
        tmp = unlock(E * A[k]; itag=_mps_bond(k-1))
        E = lock(A[k]'; itag=_mps_bond(k)) * tmp
    end
    # Site b: insert opb (its operator leg contracts the carried one → singlet),
    # then close the right bond (identity) to a scalar.
    tmp = unlock(_op_at(opb, b) * (E * A[b]); itag=_mps_bond(b-1))
    return (A[b]' * tmp)[]
end
