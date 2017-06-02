# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    isunix([os])

Predicate for testing if the OS provides a Unix-like interface.
See documentation in [Handling Operating System Variation](@ref).
"""
function isunix(os::Symbol)
    if iswindows(os)
        return false
    elseif islinux(os) || isbsd(os)
        return true
    else
        throw(ArgumentError("unknown operating system \"$os\""))
    end
end

"""
    islinux([os])

Predicate for testing if the OS is a derivative of Linux.
See documentation in [Handling Operating System Variation](@ref).
"""
islinux(os::Symbol) = (os == :Linux)

"""
    isbsd([os])

Predicate for testing if the OS is a derivative of BSD.
See documentation in [Handling Operating System Variation](@ref).
"""
isbsd(os::Symbol) = (os == :FreeBSD || os == :OpenBSD || os == :NetBSD || os == :DragonFly || os == :Darwin || os == :Apple)

"""
    iswindows([os])

Predicate for testing if the OS is a derivative of Microsoft Windows NT.
See documentation in [Handling Operating System Variation](@ref).
"""
iswindows(os::Symbol) = (os == :Windows || os == :NT)

"""
    isapple([os])

Predicate for testing if the OS is a derivative of Apple Macintosh OS X or Darwin.
See documentation in [Handling Operating System Variation](@ref).
"""
isapple(os::Symbol) = (os == :Apple || os == :Darwin)

"""
    @static

Partially evaluates an expression at parse time.

For example, `@static iswindows() ? foo : bar` will evaluate `iswindows()` and insert either `foo` or `bar` into the expression.
This is useful in cases where a construct would be invalid on other platforms,
such as a `ccall` to a non-existent function.
`@static if isapple() foo end` and `@static foo <&&,||> bar` are also valid syntax.
"""
macro static(ex)
    if isa(ex, Expr)
        if ex.head === :if || ex.head === :&& || ex.head === :||
            cond = eval(__module__, ex.args[1])
            if xor(cond, ex.head === :||)
                return esc(ex.args[2])
            elseif length(ex.args) == 3
                return esc(ex.args[3])
            elseif ex.head === :if
                return nothing
            else
                return cond
            end
        end
    end
    throw(ArgumentError("invalid @static macro"))
end

let KERNEL = ccall(:jl_get_UNAME, Any, ()),
    __module__ = @__MODULE__
    # evaluate the zero-argument form of each of these functions
    # as a function returning a static constant based on the build-time
    # operating-system kernel
    for f in (:isunix, :islinux, :isbsd, :isapple, :iswindows)
        @eval $f() = $(getfield(__module__, f)(KERNEL))
    end
end
