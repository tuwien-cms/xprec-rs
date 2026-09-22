# Cross-language micro-benchmark harness (Julia / MultiFloats.jl).
#
# The input generation and the timing methodology are shared with
# `benches/core.rs` and `bench/python/bench.py`; the canonical specification
# lives in `bench/README.md`.  Deliberately dependency-free apart from
# MultiFloats: one CSV row per `(operation, mode)`.
#
# Usage:
#
#     taskset -c 2 julia --project=bench/julia bench/julia/bench.jl \
#         --out bench/out/julia.csv
#
# Copyright (C) 2023-2025 Markus Wallerberger and others
# SPDX-License-Identifier: MIT

using MultiFloats
using Printf

# ---------------------------------------------------------------------------
# Shared input specification

const SPLITMIX_INC = 0x9E3779B97F4A7C15
const FNV_OFFSET = 0xcbf29ce484222325
const FNV_PRIME = 0x00000100000001b3
const POWI_EXP = 3

function splitmix64(state::UInt64)
    state += SPLITMIX_INC
    z = state
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return state, z ⊻ (z >> 31)
end

function fnv1a64(s::AbstractString)
    h = FNV_OFFSET
    for b in codeunits(s)
        h ⊻= UInt64(b)
        h *= FNV_PRIME
    end
    return h
end

# Uniform value in [1, 2) built from the 52 low bits of `u`.
@inline unit12(u::UInt64) = reinterpret(Float64, 0x3FF0000000000000 | (u >> 12))

# Uniform value in [-0.5, 0.5).
@inline signed12(u::UInt64) = unit12(u) - 1.5

# Exact power of two 2^-k for k in [0, 21).
@inline pow2_neg(k::Int) = 1.0 / Float64(UInt64(1) << k)

# ---------------------------------------------------------------------------
# Operations

const OPS = ["noop", "add", "sub", "mul", "div", "sqrt", "cbrt",
             "exp", "exp2", "log", "log2", "log10", "powi", "powf"]

const BINARY = Set(["add", "sub", "mul", "div", "powf"])

# The input transforms cover every operation of the shared specification, not
# only the ones this harness measures, so that the three generators stay
# literally identical and the checksum comparison keeps its meaning when an
# operation is added here later.
function input_a(op::AbstractString, u::UInt64)
    if op == "exp" || op == "exp2"
        return signed12(u) * 32.0
    elseif op in ("sin", "cos", "tan", "atan", "atan2")
        return signed12(u) * 8.0
    elseif op in ("sinh", "cosh", "tanh")
        return signed12(u) * 4.0
    elseif op == "expm1" || op == "log1p"
        return signed12(u) * 0.0625
    else
        return unit12(u)
    end
end

function input_b(op::AbstractString, i::Int, u::UInt64)
    if op == "add" || op == "sub"
        return unit12(u) * pow2_neg((i - 1) % 21)
    elseif op == "atan2"
        return signed12(u) * 8.0
    elseif op == "powf"
        return signed12(u) * 4.0
    else
        return unit12(u)
    end
end

@inline function _hash_update(hash::UInt64, x::Float64)
    bits = reinterpret(UInt64, x)
    for k in 0:7
        hash ⊻= (bits >> (8 * k)) & UInt64(0xFF)
        hash *= FNV_PRIME
    end
    return hash
end

function generate(op::AbstractString, n::Int)
    state = fnv1a64(op)
    hash = FNV_OFFSET

    a = Vector{Float64}(undef, n)
    for i in 1:n
        state, u = splitmix64(state)
        a[i] = input_a(op, u)
        hash = _hash_update(hash, a[i])
    end

    b = Vector{Float64}(undef, n)
    for i in 1:n
        state, u = splitmix64(state)
        b[i] = input_b(op, i, u)
        hash = _hash_update(hash, b[i])
    end

    return a, b, hash
end

# ---------------------------------------------------------------------------
# Evaluation
#
# The dispatch on the operation name happens once, outside the timed loop
# (function barrier); the returned closure is specialised by the compiler.

function op_function(op::AbstractString)
    if op == "noop"
        (a, b) -> a
    elseif op == "add"
        (a, b) -> a + b
    elseif op == "sub"
        (a, b) -> a - b
    elseif op == "mul"
        (a, b) -> a * b
    elseif op == "div"
        (a, b) -> a / b
    elseif op == "sqrt"
        (a, b) -> sqrt(a)
    elseif op == "cbrt"
        (a, b) -> cbrt(a)
    elseif op == "exp"
        (a, b) -> exp(a)
    elseif op == "exp2"
        (a, b) -> exp2(a)
    elseif op == "log"
        (a, b) -> log(a)
    elseif op == "log2"
        (a, b) -> log2(a)
    elseif op == "log10"
        (a, b) -> log10(a)
    elseif op == "powi"
        (a, b) -> a^POWI_EXP
    elseif op == "powf"
        (a, b) -> a^b
    else
        error("unknown op $op")
    end
end

# Reduce a benchmark value to a plain Float64 so that the accumulation loop is
# identical for both element types.  `_limbs[1]` is the leading limb.
@inline reduce_value(x::Float64) = x
@inline reduce_value(x::Float64x2) = x._limbs[1]

# ---------------------------------------------------------------------------
# Timing

const SINKS = Ref(0.0)

function median_sample!(samples::Vector{Float64})
    sort!(samples)
    return samples[cld(length(samples), 2)]
end

"""Batched (throughput) form: independent operations, four-way unrolled."""
function time_throughput(n::Int, reps::Int, eval::F) where {F}
    samples = Vector{Float64}(undef, reps)
    for r in 1:reps
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        t0 = time_ns()
        i = 1
        while i + 3 <= n
            s0 += reduce_value(eval(i))
            s1 += reduce_value(eval(i + 1))
            s2 += reduce_value(eval(i + 2))
            s3 += reduce_value(eval(i + 3))
            i += 4
        end
        while i <= n
            s0 += reduce_value(eval(i))
            i += 1
        end
        dt = (time_ns() - t0) / n
        SINKS[] += s0 + s1 + s2 + s3
        samples[r] = dt
    end
    return median_sample!(samples)
end

"""Dependent (latency) form.  Informational only."""
function time_latency(op::AbstractString, a::Vector{T}, b::Vector{T}, reps::Int) where {T}
    n = length(a)
    binary = op in BINARY
    f = op_function(op)
    samples = Vector{Float64}(undef, reps)
    for r in 1:reps
        acc = one(T)
        t0 = time_ns()
        if binary
            for i in 1:n
                acc = f(a[i], acc)
            end
        else
            for i in 1:n
                acc = f(acc, b[i])
            end
        end
        dt = (time_ns() - t0) / n
        SINKS[] += reduce_value(acc)
        samples[r] = dt
    end
    return median_sample!(samples)
end

# ---------------------------------------------------------------------------
# Driver

function main()
    out = "bench/out/julia.csv"
    n = 16384
    reps = 15
    argv = ARGS
    i = 1
    while i <= length(argv)
        if argv[i] == "--out"
            out = argv[i+1]; i += 2
        elseif argv[i] == "--n"
            n = parse(Int, argv[i+1]); i += 2
        elseif argv[i] == "--reps"
            reps = parse(Int, argv[i+1]); i += 2
        else
            error("unknown argument: $(argv[i])")
        end
    end

    mkpath(dirname(out))
    rows = String["impl,op,mode,ns_per_op,checksum"]

    println("impl=multifloats n=$n reps=$reps")
    @printf("%-8s %12s %12s %18s\n", "op", "mf-thr", "mf-lat", "checksum")

    for op in OPS
        a64, b64, checksum = generate(op, n)
        amf = Float64x2.(a64)
        bmf = Float64x2.(b64)

        f = op_function(op)
        thr = time_throughput(n, reps, i -> f(amf[i], bmf[i]))
        lat = time_latency(op, amf, bmf, reps)

        @printf("%-8s %12.4f %12.4f 0x%016x\n", op, thr, lat, checksum)
        cs = @sprintf("0x%016x", checksum)
        push!(rows, "multifloats,$op,throughput,$(round(thr, digits=6)),$cs")
        push!(rows, "multifloats,$op,latency,$(round(lat, digits=6)),$cs")
    end

    open(out, "w") do io
        println(io, "# xprec-rs benchmark harness (julia/MultiFloats)")
        println(io, "# impl=multifloats n=$n reps=$reps powi_exp=$POWI_EXP")
        for row in rows
            println(io, row)
        end
    end
    println("\nwrote $out")
end

main()
