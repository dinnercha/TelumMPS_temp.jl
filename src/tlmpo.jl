



struct TLMpo{T <: TLArray}
    TLA::Vector{T}
    oc::Int   # orthogonality (canonical) center; 0 = none/undefined
end

function TLMpo(TLA::Vector{T}, oc::Integer=0) where T <: TLArray
    return TLMpo{T}(TLA, oc)
end

"""
    TLMpo(W1, Wi, WL, N)

Build a length-`N` MPO from three template tensors: the first site `W1`,
the (repeated) bulk site `Wi`, and the last site `WL`. Each template is
re-tagged per site. Leg convention (matching `init_MPS`): legs are ordered
`(phys, phys, left-bond, right-bond)`; bonds are tagged `dummy_l` / `OB,i` /
`dummy_r`, physical legs `S,i`.
"""
function TLMpo(W1::TLArray, Wi::TLArray, WL::TLArray, N::Integer)
    N >= 2 || throw(ArgumentError("TLMpo needs N >= 2, got N=$N"))
    sitetags(i) = ("site,$i", "site,$i",
                   i == 1 ? "dummy_l"  : "bond,w,$(i-1)",
                   i == N ? "dummy_r" : "bond,w,$i")
    sites = map(1:N) do i
        W = i == 1 ? W1 : i == N ? WL : Wi
        TLArray(W, sitetags(i))
    end
    return TLMpo(sites)
end

"""
    TLMpo(Wi, N; leftbond=3, rightbond=4)
    
Build a length-`N` MPO from the bulk tensor `Wi` alone. The first and last
sites are generated automatically by projecting the boundary bond legs of `Wi`
onto the vacuum sector: the left bond (`leftbond`) keeps its last vacuum copy,
the right bond (`rightbond`) its first. Leg convention as in `TLMpo(W1, Wi, WL, N)`.
"""
function TLMpo(Wi::TLArray, N::Integer; leftbond::Int=3, rightbond::Int=4)
    zq = zero_qlabels(Wi)
    W1 = getsub(Wi, leftbond,  x -> x == zq ? -1 : nothing)  # last vacuum copy
    WL = getsub(Wi, rightbond, x -> x == zq ?  1 : nothing)  # first vacuum copy
    return TLMpo(W1, Wi, WL, N)
end

import Base: show

function Base.show(io::IO, mpo::TLMpo)
    print(io, "TLMpo(length=$(length(mpo.TLA)))")
end

function Base.show(io::IO, ::MIME"text/plain", mpo::TLMpo)
    N = length(mpo.TLA)
    println(io, "TLMpo (L=$N)")
    println(io, "  canonical center: ", mpo.oc == 0 ? "none" : string(mpo.oc))

    if N == 0
        println(io, "  (empty)")
        return
    end

    # Symmetry info, read from the first tensor.
    syms = symm(mpo.TLA[1])
    sym_names = join((totxt(s) for s in syms), " ⊗ ")
    println(io, "  symmetry: ", sym_names, "  (", length(syms), " symmetr",
            length(syms) == 1 ? "y" : "ies", ")")

    # Reduced dim (sum of RMT dims) and #sectors for one leg's space list.
    legdim(sp) = (sum(d for (_, d) in sp; init=0), length(sp))

    # Max reduced bond dim across all legs of all sites.
    maxbond = maximum(s -> maximum(l -> legdim(l)[1], s.spaces; init=0), mpo.TLA; init=0)
    println(io, "  max bond dim (reduced): ", maxbond)

    println(io, " i,tag (D(*)|#of sectors): ")
    println(io)

    half_lines = 4

    for i in 1:N
        if N > 2 * half_lines + 1
            if i > half_lines && i <= N - half_lines
                if i == half_lines + 1
                    println(io, "  ⋮")
                end
                continue
            end
        end

        t = mpo.TLA[i]
        inds_info = t.inds

        # Per leg: "tag<dir>"  with reduced-dim and sector count.
        leg_strs = map(eachindex(inds_info)) do l
            ind = inds_info[l]
            d, ns = legdim(t.spaces[l])
            string(getfield(ind, 1), getfield(ind, 2), " (", d, "|", ns, "s)")
        end

        marker = i == mpo.oc ? "  ← oc" : ""
        println(io, "  [", lpad(i, ndigits(N)), "] ", join(leg_strs, "  "), marker)
    end
end

# ── rank-4 MPO building blocks ───────────────────────────────────────────────
# Convention (shared with TLMpo, auto_mpo and the manual FSM scripts in
# Scripts/): legs are (site, site, left, right); the left bond has dir '+', the
# right bond dir '-'. The operator leg of an IROP therefore has exactly one
# bond it can sit on without a legflip -- its own direction picks it: '+' goes
# left (terminal tensor), '-' goes right (leading tensor).

# Operator leg of a rank-3 IROP: the one whose tag differs from the physical
# pair (both physical legs carry the same tag, e.g. "site").
function _opleg(A::TLArray)
    t = ntuple(l -> A.inds[l].itags, 3)
    t[1] == t[2] && return 3
    t[1] == t[3] && return 2
    t[2] == t[3] && return 1
    throw(ArgumentError("get_rank4: cannot tell the operator leg from tags $t; " *
                        "the two physical legs must share a tag"))
end

"""
    get_rank4(op)              -> rank-4 (site, site, left, right)
    get_rank4(op, side)        -> force `:left` or `:right`
    get_rank4(q::NamedTuple[, side]) -> NamedTuple of rank-4 tensors

Rank-4 MPO building block from a local operator, in the house convention
`(site, site, left, right)` with left bond dir `'+'` and right bond dir `'-'`.

A rank-2 scalar (`q.I`, `q.Z`, a density, …) gets vacuum singletons on both
bond legs. For a rank-3 IROP the operator leg's own direction picks its bond —
`'+'` goes left (terminal tensor), `'-'` goes right (leading tensor) — so
nothing is `legflip`ped: `q.F` (`'-'`) comes out leading and `q.F'Z` (`'+'`)
terminal, exactly the pair one hopping direction needs.

Passing `side` forces the other placement, `legflip`ping the operator leg to
match the bond. That is not an escape hatch but a real need: the h.c. of a
hopping term puts the conjugate operator on the *first* site, and conjugate
operators naturally point the other way — `(F†)_i ⊗ (ZF)_j` needs `F†` on the
right bond and `ZF` on the left, both against their own direction. This is the
`legflip` the hand-written FSM scripts do.

The operator leg is located by tag, so a fresh contraction product with legs
`(site, op, site)` needs no `permutedims` first.

Passing a whole `getLocalSpace` result converts every rank-2/3 entry and keeps
the keys; anything else is dropped:

    q4 = get_rank4(q)                                  # (Z, I, F, S), by direction
    term(-t, i => q4.F, i+1 => get_rank4(q.F' * lock(q.Z,2)))            # c_i c†_j
    term(-t, i => dag(get_rank4(q.F, :left)),
             i+1 => get_rank4(lock(q.Z,1) * q.F, :left))                 # h.c.
"""
function get_rank4(op::AbstractTLArray, side::Union{Symbol, Nothing}=nothing)
    side === nothing || side in (:left, :right) || throw(ArgumentError(
        "get_rank4: side must be :left or :right, got :$side"))
    A = to_concrete(op)
    n = length(A.inds)
    n == 2 && return to_concrete(addSingleton(A, (3, 4); itag=("left", "right"),
                                              dir=('+', '-')))
    n == 3 || throw(ArgumentError(
        "get_rank4: op must be rank-2 or rank-3, got rank $n"))
    p = _opleg(A)
    d = side === nothing ? A.inds[p].dir : (side === :left ? '+' : '-')
    if A.inds[p].dir != d
        A = to_concrete(legflip(A, p))
    end
    p == 3 || (A = to_concrete(permutedims(A, (filter(!=(p), (1, 2, 3))..., p))))
    B = d == '+' ?
        setitag(addSingleton(A; itag="right", dir='-'), 3, "left") :
        setitag(addSingleton(A, 3; itag="left", dir='+'), 4, "right")
    return to_concrete(B)
end

function get_rank4(q::NamedTuple, side::Union{Symbol, Nothing}=nothing)
    ops = Pair{Symbol, TLArray}[]
    for (k, v) in pairs(q)
        v isa AbstractTLArray && length(Telum.inds(v)) in (2, 3) &&
            push!(ops, k => get_rank4(v, side))
    end
    isempty(ops) && throw(ArgumentError("get_rank4: no rank-2/3 operators in $(keys(q))"))
    return NamedTuple(ops)
end

"""
    dag(W)

Hermitian conjugate of a rank-4 `(site, site, left, right)` MPO tensor, back in
the same convention. Telum's `'` is a plain conjugation, so the two physical
legs are swapped to undo the implied transpose, and the two bond legs are
swapped as well — which is also what keeps the `'+'`/`'-'` convention after the
conjugation flipped every direction. A leading tensor therefore comes back as a
terminal one, i.e. `dag(W)` is the h.c. partner to place at the other end of the
bond: `term(J, i => S4d, j => dag(S4d))` builds `J S_i·S_j†`.
"""
function dag(W::AbstractTLArray)
    A = to_concrete(W)
    length(A.inds) == 4 || throw(ArgumentError(
        "dag: expects a rank-4 (site, site, left, right) tensor, got rank $(length(A.inds))"))
    B = permutedims(A', (2, 1, 4, 3))
    return to_concrete(setitag(setitag(B, 3, "left"), 4, "right"))
end
