using Printf
using Serialization
using Dates

# Ground-state DMRG for the TLMps/TLMpo wrapper, ported from
# Tutorial/3_tutorial_DMRG.ipynb. Tag scheme of this codebase:
#   MPS phys "site,i", bond "bond,m,i", left boundary "SLeft", right "SRight"
#   MPO phys "site,i", bond "bond,w,i", boundaries "dummy_l"/"dummy_r"
#
# This file holds the shared engine (environments, Lanczos, reporting, the
# dispatcher dmrg!) and the two-site sweep. Single-site (+subspace expansion)
# lives in dmrg1s.jl.

# Build right-environment stack Hrl (length N+2). Convention used by the sweeps:
#   left  env of site j = Hrl[j]
#   right env of site j = Hrl[j+2]
# Hrl[1]/Hrl[end] are the boundary caps.
function _getHrl(MPO::Vector{<:TLArray}, MPS::Vector{<:TLArray})
    N = length(MPS)
    Hrl = Vector{TLArray}(undef, N + 2)

    li = findleg(MPS[1]; itag="SLeft")
    left_id = getIdentity((MPS[1]', li); itag="SLeft")
    Hrl[1] = addSingleton(left_id, 3; itag="dummy_l", dir='-')   # contracts MPO[1] dummy_l '+'

    li = findleg(MPS[end]; itag="SRight")
    right_id = getIdentity((MPS[end], li); itag="SRight")
    Hrl[end] = addSingleton(right_id, 3; itag="dummy_r", dir='+')  # contracts MPO[N] dummy_r '-'

    for i in 1:N
        Hrl[i+1] = MPS[i]' * lock(Hrl[i] * MPO[i] * MPS[i]; itag="bond,m,$i")
    end
    return Hrl
end

# Lanczos for the effective Hamiltonian (verbatim from the tutorial;
# tag-agnostic, contraction by index matching). `Hs` is the list of MPO tensors
# in the active window (1 for single-site, 2 for two-site).
function _eigs_GS(Hl, Hs, Hr, M; tol=1e-8, nKrylov=5)
    As = Vector{TLArray}(undef, nKrylov)
    As[1] = M / norm(M)
    αs = zeros(nKrylov); βs = zeros(nKrylov-1)
    cnt = 0

    H_tot = 1.0

    for H in Hs H_tot=H_tot * H end

    for i in 1:nKrylov
        Amul = Hl * As[i]
        
        Amul = H_tot * Amul
        Amul = Amul * Hr
        αs[i] = (As[i]' * Amul)[]
        cnt += 1
        if i < nKrylov
            for _=1:2, j=1:i Amul -= (As[j]' * Amul)[] * As[j] end
            Anorm = norm(Amul)
            if Anorm < tol break end
            As[i+1] = Amul / Anorm
            βs[i] = Anorm
        end
    end
    Hkrylov = SymTridiagonal(αs[1:cnt], βs[1:cnt-1])
    # Ritz value = ⟨Anew|H|Anew⟩ (fully reorthogonalized basis), so no extra
    # H application: exactly nKrylov applications per local problem.
    θ, V = eigen(Hkrylov); vec = V[:, 1]
    Anew = sum(vec[i] * As[i] for i in 1:cnt)
    return Anew, θ[1]
end

# Discarded weight of a truncated SVD M ≈ U*S*Vd: 1 - ‖U S Vd‖²/‖M‖².
function _dw(M, U, S, Vd)
    nm = norm(M)
    nm == 0 && return 0.0
    return max(0.0, 1 - (norm(U * S * Vd) / nm)^2)
end

# Max bond dimension across MPS bonds. Default is the reduced (multiplet) count;
# `full=true` weights each sector by its irrep dimension (2S+1 for an SU(2)
# sector), i.e. the bond dimension the state would have without the symmetry.
_maxbond(MPS; full::Bool=false) =
    maximum(s -> _maxleg(symm(s), s.spaces, full), MPS; init=0)

_maxleg(syms, spaces, full) = maximum(spaces; init=0) do l
    sum(d * (full ? _irrepdim(syms, q) : 1) for (q, d) in l; init=0)
end

# "χ = 80" normally, "χ* = 80 (χ = 160)" when a non-abelian (SU(2)) symmetry is
# present and the reduced count no longer equals the original dimension.
function _chi_str(MPS, mbond)
    all(isabelian, symm(MPS[1])) && return @sprintf("χ = %-5d", mbond)
    return @sprintf("χ* = %d (χ = %d)", mbond, _maxbond(MPS; full=true))
end

# Break the MPS/MPO shared "dummy_r" boundary tag, build envs, then mirror the
# tutorial's rightmost-leg flip. Returns Hrl.
function _dmrg_prep!(MPS, MPO)
    N = length(MPS)
    # Idempotent (warm restart, e.g. a Nkeep ramp of repeated dmrg! calls): undo
    # the previous call's retag + flip so MPS[N] looks fresh again.
    findleg(MPS[N]; itag="dummy_r") === nothing &&
        (MPS[N] = setitag(legflip(MPS[N]; itag="SRight"), "dummy_r"; itag="SRight"))
    MPS[N] = setitag(MPS[N], "SRight"; itag="dummy_r")
    Hrl = _getHrl(MPO, MPS)
    MPS[N]   = legflip(MPS[N]; itag="SRight")
    Hrl[end] = legflip(Hrl[end]; itag="SRight")
    # The flip above is a no-op on a vacuum boundary but carries a 1j factor when
    # the state has a nonzero total flux; correct the cap against the final
    # MPS[N]. Only the right envs (rebuilt from Hrl[end] during the sweeps) are
    # affected — the left envs built in `_getHrl` contract bra and ket boundary
    # legs directly and were always right. See `_cap_flux_fix`.
    Hrl[end] = _cap_flux_fix(Hrl[end], MPS[N], _mps_bond(N - 1))
    return Hrl
end

# Two-site sweep (right->left then left->right). Mutates MPS, Hrl.
# Returns (E, max_discarded_weight).
function _sweep_2s!(MPS, MPO, Hrl, Nkeep; tol, nKrylov, cutoff)
    N = length(MPO); E = 0.0; mdw = 0.0
    for i in N-1:-1:1                                  # right -> left
        M, E = _eigs_GS(Hrl[i], [MPO[i], MPO[i+1]], Hrl[i+3], MPS[i]*MPS[i+1]; tol, nKrylov)
        tags = i == 1 ? ("SLeft", "site,$i") : ("bond,m,$(i-1)", "site,$i")
        lids = [findleg(M; itag=t) for t in tags]
        (; U, S, Vd) = truncated_svd(M, lids, "bond,m,$i,left", "bond,m,$i,right"; Nkeep=Nkeep, cutoff=cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        MPS[i]   = removeitag(U * S, "right")
        MPS[i+1] = removeitag(Vd, "right")
        Hrl[i+2] = (Hrl[i+3] * MPS[i+1]) * MPO[i+1] * lock(MPS[i+1]'; itag="bond,m,$i")
    end
    for i in 1:N-1                                     # left -> right
        M, E = _eigs_GS(Hrl[i], [MPO[i], MPO[i+1]], Hrl[i+3], MPS[i]*MPS[i+1]; tol, nKrylov)
        tags = i == 1 ? ("SLeft", "site,$i") : ("bond,m,$(i-1)", "site,$i")
        lids = [findleg(M; itag=t) for t in tags]
        (; U, S, Vd) = truncated_svd(M, lids, "bond,m,$i,left", "bond,m,$i,right"; Nkeep=Nkeep, cutoff=cutoff)
        mdw = max(mdw, _dw(M, U, S, Vd))
        MPS[i]   = removeitag(U, "left")
        MPS[i+1] = removeitag(S * Vd, "left")
        Hrl[i+1] = (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag="bond,m,$i")
    end
    return E, mdw
end

function _print_options(mode, N, Nkeep, cutoff, nKrylov, tol, Nsweep, Etol, α, α_decay,
                        obs, save_path, init_energy)
    bar = "═"^65
    println("╔" * bar)
    @printf("║  DMRG   ·   mode = %-3s   ·   N = %d\n", mode, N)
    @printf("║  %s\n", Dates.format(now(), "yyyy-mm-dd (E)  HH:MM:SS"))
    println("╟" * "─"^65)
    @printf("║  init_energy = %.10f\n", init_energy)
    @printf("║  Nkeep = %-5d   cutoff (discarded weight) = %.1e\n", Nkeep, cutoff)
    @printf("║  Lanczos: nKrylov = %d, tol = %.1e\n", nKrylov, tol)
    @printf("║  Nsweep = %-4d   convergence |dE| < %.1e\n", Nsweep, Etol)
    mode == "1s+" && @printf("║  subspace expansion: α = %.1e, decay = %.2f\n", α, α_decay)
    obs === nothing || println("║  observables: ", join(first.(obs), ",  "))
    save_path === nothing || println("║  saving result → ", save_path)
    println("╚" * bar)
    flush(stdout)
end

# Print one sweep block (+obs on a snapshot) and accumulate history.
function _report_sweep!(history, si, Nsweep, E, dE, mdw, mbond, st, elapsed, MPS, N, obs)
    eta = si < Nsweep ? (elapsed / si) * (Nsweep - si) : 0.0
    @printf("  ┌ sweep %2d/%2d %s[%s]\n", si, Nsweep, " "^28, Dates.format(now(), "HH:MM:SS"))
    @printf("  │  E = %.10f      dE = %+.3e\n", E, dE)
    @printf("  │  %s  dw = %.2e      t %.2fs · elapsed %.1fs · ETA %.1fs\n",
            _chi_str(MPS, mbond), mdw, st, elapsed, eta)
    rec = Dict{String,Any}("sweep"=>si, "E"=>E, "dE"=>dE, "dw"=>mdw, "chi"=>mbond, "time"=>st)
    if obs !== nothing
        snap = TLMps(deepcopy(MPS), N)        # measure without disturbing DMRG gauge
        ov = Dict{String,Any}()
        parts = String[]
        for (name, f) in obs
            v = f(snap); ov[name] = v
            push!(parts, @sprintf("%s = %.6g", name, v))
        end
        println("  │  ", join(parts, "   "))
        rec["obs"] = ov
    end
    println("  └"); flush(stdout)              # cluster stdout is block-buffered
    push!(history, rec)
    return history
end

"""
    dmrg!(mps::TLMps, mpo::TLMpo, Nkeep, Nsweep; mode="2s", cutoff=0.0, nKrylov=5,
          tol=1e-8, Etol=1e-10, obs=nothing, save_path=nothing,
          α=1e-4, α_decay=0.9, meta_out=false) -> (mps, E)

Ground-state DMRG, mutating `mps` in place (center ends at site N). Returns
`(mps, E)`, or `(mps, E, meta)` when `meta_out=true`.

Use `dmrg` instead to run on a copy and leave the input `mps` unchanged.
`meta` is a named tuple containing all calculation parameters, initial/final
energies, completed sweep count, elapsed time, and sweep history.

`mode` selects the update:
- `"2s"`  two-site (default; truncates to `Nkeep`/`cutoff`).
- `"1s"`  strictly single-site (bond dimension fixed by the input MPS).
- `"1s+"` single-site with subspace expansion (DMRG3S, Hubig et al. 2015): the
  bond is enriched with `α`-weighted Hamiltonian directions so it can grow up to
  `Nkeep` at single-site cost. `α` decays by `α_decay` each sweep.

Truncation knobs (independent of `tol`, which is the Lanczos convergence
threshold): `Nkeep` caps the bond dimension (in multiplets) and `cutoff` is the
*discarded weight* per SVD, `Σ_{i>k} d_i σ_i² / Σ_j d_j σ_j² ≤ cutoff` with
`d_i` the irrep dimension of the sector each `σ_i` sits in — the same criterion
ITensor's `cutoff` uses, see [`truncated_svd`](@ref). Whichever of the two binds
first wins. The realized truncation is reported each sweep as the discarded
weight `dw`.

`Etol` stops the sweeps early once `|dE| < Etol`.

`obs` registers per-sweep measurements: a vector of `name => f` where
`f(m::TLMps)` returns a number, evaluated on a fresh snapshot. `save_path` (if
given) serializes the final `(metadata, mps, mpo)` result. Per-sweep values
(sweep, E, dE, dw, chi, time, obs) are available as `metadata.history`.

    obs = ["S3·S4" => m->observe_cor(m, q.S, 3, q.S', 4)]
    dmrg!(mps, mpo, 60, 10; mode="1s+", obs=obs, save_path="run.jls")
"""
function dmrg!(mps::TLMps, mpo::TLMpo, Nkeep::Int, Nsweep::Int;
               mode::AbstractString="2s", cutoff::Float64=0.0, nKrylov::Int=5,
               tol::Float64=1e-8, Etol::Float64=0.0, obs=nothing, save_path=nothing,
               α::Float64=1e-4, α_decay::Float64=0.9, meta_out::Bool=false)
    MPS, MPO = mps.TLA, mpo.TLA
    N = length(MPO); @assert length(MPS) == N
    mode in ("1s", "1s+", "2s") || throw(ArgumentError("mode must be \"1s\", \"1s+\", or \"2s\"; got \"$mode\""))

    Hrl = _dmrg_prep!(MPS, MPO)
    init_energy = real(deleteSingleton(Hrl[N+1], findleg(Hrl[N+1]; itag="dummy_r"))[])
    _print_options(mode, N, Nkeep, cutoff, nKrylov, tol, Nsweep, Etol, α, α_decay,
                   obs, save_path, init_energy)

    history = Dict{String,Any}[]
    E = 0.0; Eprev = Inf; αcur = α; t0 = time()
    for si in 1:Nsweep
        st = @elapsed begin
            if mode == "2s"
                E, mdw = _sweep_2s!(MPS, MPO, Hrl, Nkeep; tol, nKrylov, cutoff)
            elseif mode == "1s"
                E, mdw = _sweep_1s!(MPS, MPO, Hrl, Nkeep; tol, nKrylov, cutoff, expand=false)
            else  # "1s+"
                E, mdw = _sweep_1s!(MPS, MPO, Hrl, Nkeep; tol, nKrylov, cutoff, expand=true, α=αcur)
                αcur *= α_decay
            end
        end
        mps.oc = N
        dE = E - Eprev; Eprev = E
        _report_sweep!(history, si, Nsweep, E, dE, mdw, _maxbond(MPS), st,
                       time()-t0, MPS, N, obs)
        if abs(dE) < Etol
            @printf("  ✓ converged: |dE| = %.2e < %.2e\n", abs(dE), Etol)
            break
        end
    end
    elapsed = time() - t0
    @printf("  ▸ done: E = %.10f   (%.1fs total)\n", E, elapsed)
    flush(stdout)
    mps.oc = N
    meta = meta_out || save_path !== nothing ?
        (; N, Nkeep, Nsweep, mode, cutoff, nKrylov, tol, Etol, obs, save_path,
         α, α_decay, init_energy, final_energy=E,
         sweeps_completed=length(history), elapsed, history) : nothing
    if save_path !== nothing
        mkpath(dirname(abspath(save_path)))   # save_path's folder need not exist yet
        serialize(save_path, (; metadata=meta, mps, mpo))
    end
    return meta_out ? (mps, E, meta) : (mps, E)
end

"""
    dmrg(mps::TLMps, mpo::TLMpo, Nkeep, Nsweep; kwargs...)

Non-mutating [`dmrg!`](@ref): runs DMRG on a deep copy of `mps`.
"""
dmrg(mps::TLMps, mpo::TLMpo, Nkeep::Int, Nsweep::Int; kwargs...) =
    dmrg!(deepcopy(mps), mpo, Nkeep, Nsweep; kwargs...)
