



mutable struct TLMps{T <: TLArray}
    TLA::Vector{T}
    oc::Int   # orthogonality (canonical) center; 0 = none/undefined
end

function TLMps(TLA::Vector{T}, oc::Integer=0) where T <: TLArray
    return TLMps{T}(TLA, oc)
end

"""
    TLMps(A1, Ai, AL, N; oc=0)

Build a length-`N` MPS from three template tensors: the first site `A1`,
the (repeated) bulk site `Ai`, and the last site `AL`. Each template is
re-tagged per site. Legs are ordered `(phys, left-bond, right-bond)`; bonds
are tagged `SLeft` / `bond,m,i` / `dummy_r`, the scheme `init_MPS` builds and
`dmrg`/`tdvp` expect (`TLMpo` uses `dummy_l` and bond label `w`), physical legs
`site,i`. `oc` records the canonical center (`0`: unknown, `move_canonical!`
then left-canonicalizes first).
"""
function TLMps(A1::TLArray, Ai::TLArray, AL::TLArray, N::Integer; oc::Integer=0)
    N >= 2 || throw(ArgumentError("TLMps needs N >= 2, got N=$N"))
    sitetags(i) = ("site,$i",
                   i == 1 ? "SLeft"    : "bond,m,$(i-1)",
                   i == N ? "dummy_r" : "bond,m,$i")
    sites = map(1:N) do i
        A = i == 1 ? A1 : i == N ? AL : Ai
        TLArray(A, sitetags(i))
    end
    return TLMps(sites, oc)
end


import Base: show, length

# Number of site tensors (chain length).
Base.length(mps::TLMps) = length(mps.TLA)

function Base.show(io::IO, mps::TLMps)
    print(io, "TLMps(length=$(length(mps.TLA)))")
end

function Base.show(io::IO, ::MIME"text/plain", mps::TLMps)
    N = length(mps.TLA)
    println(io, "TLMps (L=$N)", "|  center : ", mps.oc == 0 ? "none" : string(mps.oc))

    if N == 0
        println(io, "  (empty)")
        return
    end

    # Symmetry info, read from the first tensor.
    syms = symm(mps.TLA[1])
    sym_names = join((totxt(s) for s in syms), " ⊗ ")
    println(io, "  symmetry: ", sym_names, "  (", length(syms), " symmetr",
            length(syms) == 1 ? "y" : "ies", ")")

    # Reduced dim (sum of RMT dims) and #sectors for one leg's space list.
    legdim(sp) = (sum(d for (_, d) in sp; init=0), length(sp))

    # Max reduced bond dim across all legs of all sites.
    maxbond = maximum(s -> maximum(l -> legdim(l)[1], s.spaces; init=0), mps.TLA; init=0)
    println(io, "  max bond dim (reduced): ", maxbond)
    println(io)
    println(io, " i,tag (D(*)|#of sectors): ", maxbond)
    

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

        t = mps.TLA[i]
        inds_info = t.inds

        # Per leg: "tag<dir>"  with reduced-dim and sector count.
        leg_strs = map(eachindex(inds_info)) do l
            ind = inds_info[l]
            d, ns = legdim(t.spaces[l])
            string(getfield(ind, 1), getfield(ind, 2), " (", d, "|", ns, "s)")
        end

        marker = i == mps.oc ? "  ← oc" : ""
        println(io, "  [", lpad(i, ndigits(N)), "] ", join(leg_strs, "  "), marker)
    end
end