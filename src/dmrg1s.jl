# Single-site ground-state DMRG (+ optional subspace expansion / DMRG3S,
# Hubig et al. 2015). Shares the engine (envs, Lanczos, reporting) with dmrg2s.jl.

# Build a zero TLArray shaped like T but with leg `leg` replaced by `newspace`.
_zero_like(T, leg::Int, newspace) =
    Telum._zero_tlarray_with_spaces(symm(T), T.inds,   # renamed from _zero_qspace_with_spaces in Telum 0.2
        ntuple(k -> k == leg ? newspace : T.spaces[k], length(T.inds)))

# Subspace-expansion term P = α·L·M·W, fused so it can be direct-summed onto the
# bond `bt` of M. Returns (Mexp, neighbor_pad_space) where Mexp has the enlarged
# bond and neighbor_pad_space is the qlabel space of the added block (to zero-pad
# the neighbor's matching bond).
function _expand_term(M, Henv, W, bt, wt, α)
    P = α * ((Henv * M) * W)               # legs: M-bond bt, phys, mpo-bond wt, opposite bra bond
    F  = getIdentity((P, findleg(P; itag=bt)), (P, findleg(P; itag=wt)); itag="exp_fuse")
    PC = setitag(P * F, bt; itag="exp_fuse")      # fuse (M-bond ⊕ mpo-bond) -> new "bt" block
    mleg = findleg(M; itag=bt)
    PC.inds[findleg(PC; itag=bt)].dir == M.inds[mleg].dir || (PC = legflip(PC; itag=bt))
    Mexp = oplus(M, PC, mleg)
    return Mexp, PC.spaces[findleg(PC; itag=bt)]
end

# Single-site sweep (right->left then left->right), center starts/ends at N.
# Mutates MPS, Hrl. Returns (E, max_discarded_weight).
function _sweep_1s!(MPS, MPO, Hrl, Nkeep; tol, nKrylov, cutoff, expand::Bool, α::Float64=0.0)
    N = length(MPO); E = 0.0; mdw = 0.0

    for i in N:-1:2                                    # right -> left (right-canonicalize i)
        M, E = _eigs_GS(Hrl[i], [MPO[i]], Hrl[i+2], MPS[i]; tol, nKrylov)
        bt = "bond,m,$(i-1)"                           # bond toward i-1
        if expand
            M, padsp = _expand_term(M, Hrl[i+2], MPO[i], bt, "bond,w,$(i-1)", α)  # opposite-side env
            MPS[i-1] = oplus(MPS[i-1], _zero_like(MPS[i-1], findleg(MPS[i-1]; itag=bt), padsp), findleg(MPS[i-1]; itag=bt))
        end
        lb = findleg(M; itag=bt)
        (; U, S, Vd) = truncated_svd(M, (lb,); Nkeep=Nkeep, cutoff=cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        MPS[i]   = setitag(Vd, bt; itag="svdR")
        MPS[i-1] = setitag(MPS[i-1] * (U * S), bt; itag="svdR")
        Hrl[i+1] = (Hrl[i+2] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=bt)
    end

    for i in 1:N-1                                     # left -> right (left-canonicalize i)
        M, E = _eigs_GS(Hrl[i], [MPO[i]], Hrl[i+2], MPS[i]; tol, nKrylov)
        bt = "bond,m,$i"                               # bond toward i+1
        if expand
            M, padsp = _expand_term(M, Hrl[i], MPO[i], bt, "bond,w,$i", α)
            MPS[i+1] = oplus(MPS[i+1], _zero_like(MPS[i+1], findleg(MPS[i+1]; itag=bt), padsp), findleg(MPS[i+1]; itag=bt))
        end
        rb = findleg(M; itag=bt)
        left = Tuple(l for l in 1:length(M.inds) if l != rb)
        (; U, S, Vd) = truncated_svd(M, left; Nkeep=Nkeep, cutoff=cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        MPS[i]   = setitag(U, bt; itag="svdL")
        MPS[i+1] = setitag((S * Vd) * MPS[i+1], bt; itag="svdL")
        Hrl[i+1] = (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=bt)
    end

    return E, mdw
end
