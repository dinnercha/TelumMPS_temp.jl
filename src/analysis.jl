"""
    bonddims(mps::TLMps; full=false) -> Vector{Int}

Bond dimensions of `mps`; entry `i` is the bond between sites `i` and `i+1`
(length `N-1`).

By default the *reduced* (multiplet) dimension is returned: the sum of RMT
dimensions over the sectors of that bond, i.e. what the `TLMps` display shows.
With `full=true` the *original* (non-symmetric) dimension is returned instead:
each sector is weighted by the dimension of its irrep, `prod_n dim(S_n, q_n)`,
which is `2S+1` for an SU(2) spin sector. For purely abelian symmetries
(Z, U1) every irrep is 1-dimensional and both agree.
"""
bonddims(mps::TLMps; full::Bool=false) =
    map(i -> bonddims(mps, i; full=full), 1:length(mps.TLA)-1)

"""
    bonddims(mps::TLMps, i::Integer; full=false) -> Int

Dimension of the single bond `i` (sites `i` | `i+1`).
"""
function bonddims(mps::TLMps, i::Integer; full::Bool=false)
    N = length(mps.TLA)
    1 <= i <= N-1 || throw(BoundsError("bond $i outside 1:$(N-1)"))
    t = mps.TLA[i]
    syms = symm(t)
    return sum(d * (full ? _irrepdim(syms, q) : 1)
               for (q, d) in t.spaces[findleg(t; itag=_mps_bond(i))]; init=0)
end

# Dimension of the irrep labelled by `q` in the original (non-symmetric) basis:
# 2S+1 for an SU(2) sector, 1 for any abelian (Z, U1) one.
_irrepdim(syms, q) = prod(dimension(syms[n], q[n]) for n in eachindex(syms))

"""
    entanglement_entropy(mps::TLMps; normalize=true, cutoff=0.0) -> Vector{Float64}

Von Neumann entanglement entropy (in nats) across each bond of `mps`; entry `i`
is the cut between sites `1..i` and `i+1..N` (length `N-1`).

The MPS is gauged bond by bond (on a copy — `mps` is left untouched) and the
entropy is taken from the SVD spectrum, `S = -Σ_k d_k p_k log p_k`, where `p_k`
is a reduced density-matrix eigenvalue and `d_k` the irrep dimension of its
sector, i.e. SU(2) multiplets are counted with their `2S+1` degeneracy.

`normalize=true` (default) rescales the spectrum to `Σ_k d_k p_k = 1` at every
bond, so an unnormalized MPS still gives the right entropy. Set it to `false`
to use the raw singular values — meaningful only for `‖mps‖ = 1`, and a useful
check that the state really is normalized.
"""
function entanglement_entropy(mps::TLMps; normalize::Bool=true, cutoff::Float64=0.0)
    A = copy(mps.TLA)                 # gauge steps replace entries, not the originals
    move_canonical!(TLMps(A, mps.oc), 1; cutoff=cutoff)
    syms = symm(A[1])
    return map(i -> _cut_entropy!(A, i, syms, normalize, cutoff), 1:length(A)-1)
end

"""
    entanglement_entropy(mps::TLMps, i::Integer; normalize=true, cutoff=0.0) -> Float64

Entropy of the single bond `i` only (sites `1..i` | `i+1..N`). Gauges up to site
`i` and takes one SVD instead of sweeping the whole chain.
"""
function entanglement_entropy(mps::TLMps, i::Integer; normalize::Bool=true,
                              cutoff::Float64=0.0)
    N = length(mps.TLA)
    1 <= i <= N-1 || throw(BoundsError("bond $i outside 1:$(N-1)"))
    A = copy(mps.TLA)
    move_canonical!(TLMps(A, mps.oc), i; cutoff=cutoff)
    return _cut_entropy!(A, i, symm(A[1]), normalize, cutoff)
end

# Entropy of bond `i`, center assumed on site `i`; leaves the center on `i+1`.
function _cut_entropy!(A, i::Int, syms, normalize::Bool, cutoff::Float64)
    bt = _mps_bond(i)
    rb = findleg(A[i]; itag=bt)
    left = Tuple(l for l in eachindex(A[i].inds) if l != rb)
    (; U, S, Vd) = truncated_svd(A[i], left; cutoff=cutoff)
    A[i]   = setitag(U, bt; itag="svdL")
    A[i+1] = setitag((S * Vd) * A[i+1], bt; itag="svdL")
    return _vn_entropy(S, syms, normalize)
end

function _vn_entropy(S, syms, normalize::Bool)
    ps = Float64[]; ds = Float64[]
    # Telum keeps zero sectors as inert slots (iszero/isdefined bitvectors) instead
    # of dropping them, so iterate slots and skip the inactive ones; qlabels is a
    # per-sector tuple, not a (leg, sector) matrix.
    for si in Telum.sector_slots(S)
        Telum.is_sector_active(S, si) || continue
        d = _irrepdim(syms, Telum.sector_qlabel(S, si, 1))
        rmt = S.RMTs[si]                                # diagonal, (m, m, 1...)
        for j in 1:size(rmt, 1)
            push!(ps, rmt[j, j]^2 / d); push!(ds, d)   # full σ² per state (reduced entry is σ√d)
        end
    end
    Z = normalize ? sum(ds .* ps; init=0.0) : 1.0
    Z == 0 && return 0.0
    return -sum(d * (p/Z) * log(p/Z) for (d, p) in zip(ds, ps) if p > 0; init=0.0)
end
