# AutoMPO: term-by-term Hamiltonian MPO construction, following Alice's
# accumulate-and-compress strategy (github.com/Ideogenesis-AI/Alice):
# each term is a full-length product MPO (identity fill + operator tensors);
# terms are stacked into the running MPO by direct sum (oplus) on the bond
# legs, then compressed by two SVD sweeps. All symmetry sector arithmetic —
# including nonabelian SU(2) — is delegated to Telum's oplus/svd.
#
# Tensor convention (same as the manual FSM construction in Scripts/):
# operator tensors are rank-4 with legs (site, site, left, right); unused
# bond legs are vacuum singletons (left dir '+', right dir '-'). A two-site
# term is a leading tensor (op leg → right bond) at site i and a terminal
# tensor (op leg → left bond) at site j; Jordan-Wigner/Z dressing is the
# caller's responsibility, exactly as in the manual construction.

struct MPOTerm
    coeff::Float64
    ops::Vector{Pair{Int, TLArray}}   # site => rank-4 (site, site, left, right)
end

"""
    term(coeff, i => A, j => B, ...) -> MPOTerm

Operator-string term. Sites must be strictly increasing. Gaps between listed
sites are filled automatically: with the plain identity when the crossing bond
is vacuum, and with a pass-through tensor (physical identity ⊗ identity on the
open bond space) when it is not, so a range-`r` term such as
`term(J, i => S4d, i+r => Sdag4d)` needs no intermediate tensors. Only the
operator content of an intermediate site is not guessed — for a fermionic
string list the Z-dressed tensor at every intermediate site explicitly.
"""
function term(coeff::Real, ops::AbstractVector{<:Pair{Int, <:AbstractTLArray}})
    isempty(ops) && throw(ArgumentError("term needs at least one site => op pair"))
    sites = first.(ops)
    (issorted(sites) && allunique(sites)) || throw(ArgumentError(
        "term sites must be strictly increasing, got $sites"))
    # Telum's conj/scale/permute return lazy TLArrayViews; concretize at the
    # boundary so everything downstream stays a plain TLArray.
    return MPOTerm(coeff, [Pair{Int, TLArray}(s, to_concrete(op)) for (s, op) in ops])
end

# The varargs form is the convenient one to write by hand, but splatting a
# run-time-length vector into it makes Julia compile a fresh specialization for
# every arity — on a 150-site Jordan-Wigner string that is minutes of LLVM time
# before a single tensor is touched. Collect once, then hand the vector form a
# single signature to compile.
term(coeff::Real, ops::Pair{Int, <:AbstractTLArray}...) =
    term(coeff, Pair{Int, TLArray}[s => to_concrete(op) for (s, op) in ops])

"""
    onsite(coeff, op, i) -> MPOTerm

Single-site term from a rank-2 scalar operator (vacuum bond legs added here).
"""
onsite(coeff::Real, op::AbstractTLArray, i::Integer) = term(coeff,
    i => addSingleton(op, (3, 4); itag=("left", "right"), dir=('+', '-')))

# Leg is a 1-dim vacuum singleton?
_isvac(t::TLArray, leg::Int) = length(t.spaces[leg]) == 1 &&
    t.spaces[leg][1][1] == zero_qlabels(t) && t.spaces[leg][1][2] == 1

_mpo_bond(i::Int) = "bond,w,$i"

"""
    auto_mpo(terms, N, I4d; cutoff=1e-13, Nkeep=nothing) -> TLMpo

Build a length-`N` Hamiltonian MPO from `MPOTerm`s. `I4d` is the rank-4 site
identity with vacuum bond legs (same object the manual FSM construction
uses). Terms with zero coefficient are skipped; the rest are accumulated by
`oplus` on the bond legs, then compressed by a left-then-right SVD sweep
(`cutoff`/`Nkeep` forwarded to `svd`).

`compact_every` compresses the running MPO every that many accumulated terms
(Alice's `compact_every`, same default 10), keeping the peak accumulation bond
bounded instead of growing with the term count; `compact_every=0` disables it.

Accumulation and compression run on `H / ‖I‖_F^N`, with the factor put back on
every site at the end. This is not cosmetic: unnormalised, the identity channel
at a bond carries Frobenius weight `d^(N/2)` while an operator channel carries
`O(1)`, so the running MPO is badly conditioned and `compact!`'s repeated SVD
sweeps amplify round-off by that condition number. Measured on an XY chain plus
staggered field, the unnormalised build was off by 5.5e+1 in `⟨H⟩` at N = 56 and
1.0e+2 at N = 96 (bond collapsing to 2 instead of 4); normalised it is exact to
8e-13 and 8.8e-12, and N = 40 builds ~10x faster.
"""
function auto_mpo(terms::AbstractVector{MPOTerm}, N::Integer, I4d_in::AbstractTLArray;
                  cutoff::Float64=1e-13, Nkeep::Union{Nothing, Int}=nothing,
                  compress::Bool=true, compact_every::Int=10)
    N >= 2 || throw(ArgumentError("auto_mpo needs N >= 2, got N=$N"))
    # Work on H / λ^N with λ = ‖I‖_F = √d, i.e. unit-norm identity fill, and put
    # λ back on every site at the end. Without this the identity channel at a
    # bond carries weight d^(N/2) while an operator channel carries O(1), so the
    # accumulated MPO is badly conditioned and the repeated SVD sweeps of
    # `compact!` amplify round-off by that condition number: at N = 56 the built
    # MPO was off by 5.5e+1 (and by 2.8e-7 even with every truncation disabled,
    # which is how we know it is conditioning and not truncation). Normalised,
    # the same build is exact to 8e-13 and ~10x faster.
    λ = norm(I4d_in)
    I4d = to_concrete(I4d_in / λ)
    active = [t for t in terms if t.coeff != 0.0]
    isempty(active) && throw(ArgumentError("no terms with nonzero coefficient"))

    sitetags(i) = ("site,$i", "site,$i",
                   i == 1 ? "dummy_l" : "bond,w,$(i-1)",
                   i == N ? "dummy_r" : "bond,w,$i")

    # Pass-through tensor for a nontrivial bond crossing an identity-filled
    # site: physical identity ⊗ identity on that bond space — the same object
    # the manual FSM uses for its long-range transfer entry (see
    # Scripts/SU2_test_SpinOrbit_model.jl, `I4d_S = q.I ⊗ id_S`). Cached per
    # leading operator, since every term of a given range reuses it.
    I2 = nothing
    ptcache = IdDict{TLArray, TLArray}()
    passthrough(op::TLArray) = get!(ptcache, op) do
        I2 === nothing && (I2 = deleteSingleton(I4d, (3, 4)))
        I2 ⊗ getIdentity((op, 4))
    end

    function term_chain(t::MPOTerm)
        _isvac(t.ops[1][2], 3) || throw(ArgumentError(
            "term's first tensor (site $(t.ops[1][1])) must have a vacuum left bond"))
        _isvac(t.ops[end][2], 4) || throw(ArgumentError(
            "term's last tensor (site $(t.ops[end][1])) must have a vacuum right bond"))
        chain = Vector{TLArray}(undef, N)
        for i in 1:N
            chain[i] = TLArray(I4d, sitetags(i))
        end
        for (k, (s, op)) in enumerate(t.ops)
            1 <= s <= N || throw(ArgumentError("term site $s outside 1:$N"))
            W = k == length(t.ops) ? t.coeff * op : op   # coeff on the last op
            chain[s] = TLArray(to_concrete(W / λ), sitetags(s))
        end
        for k in 1:length(t.ops)-1
            s, sn = t.ops[k][1], t.ops[k+1][1]
            (sn > s + 1 && !_isvac(t.ops[k][2], 4)) || continue
            P = passthrough(t.ops[k][2])
            for i in s+1:sn-1
                chain[i] = TLArray(P, sitetags(i))
            end
        end
        return chain
    end

    sweep_right!(A) = (for i in 1:N-1
        _move_right!(A, i; bond=_mpo_bond, cutoff=cutoff, Nkeep=Nkeep)
    end; A)
    sweep_left!(A) = (for i in N:-1:2
        _move_left!(A, i; bond=_mpo_bond, cutoff=cutoff, Nkeep=Nkeep)
    end; A)

    # A sweep leaves each bond in the direction convention of whichever move
    # touched it last (Telum's svd always hands out a '-' bond leg): ending on
    # `sweep_right!` reproduces the ('-', '+') convention `term_chain` builds
    # with, ending on `sweep_left!` gives the flipped one. Intermediate
    # compaction must end right so the next `oplus` still lines up; the final
    # compression keeps the original left-then-right order (oc = 1).
    compress!(A) = sweep_left!(sweep_right!(A))
    compact!(A) = sweep_right!(sweep_left!(A))

    # Sweeps also migrate the bond legs to the front; `oplus` is positional, so
    # restore (site, site, left, right) before accumulating on.
    function reorder!(A)
        for i in 1:N
            tags = sitetags(i)
            lb = findleg(A[i]; itag=tags[3])
            rb = findleg(A[i]; itag=tags[4])
            (lb, rb) == (3, 4) && continue
            ph = Tuple(l for l in 1:4 if l != lb && l != rb)
            A[i] = to_concrete(permutedims(A[i], (ph[1], ph[2], lb, rb)))
        end
        return A
    end

    mpo = term_chain(active[1])
    nacc = length(active) - 1
    for (k, t) in enumerate(active[2:end])
        c = term_chain(t)
        mpo[1] = oplus(mpo[1], c[1], (4,))
        for i in 2:N-1
            mpo[i] = oplus(mpo[i], c[i], (3, 4))
        end
        mpo[N] = oplus(mpo[N], c[N], (3,))
        # Skip on the last term — the final compression below covers it.
        if compress && compact_every > 0 && k % compact_every == 0 && k != nacc
            reorder!(compact!(mpo))
        end
    end
    unscale!(A) = (for i in 1:N; A[i] = to_concrete(λ * A[i]); end; A)
    compress || return TLMpo(unscale!(mpo), 0)
    compress!(mpo)
    return TLMpo(unscale!(mpo), 1)
end
