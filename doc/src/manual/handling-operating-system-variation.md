# Handling Operating System Variation

When dealing with platform libraries, it is often necessary to provide special cases for various
platforms. The variable `Sys.KERNEL` can be used to write these special cases. There are several
functions intended to make this easier: `isunix`, `islinux`, `isapple`, `isbsd`, and `iswindows`.
These may be used as follows:

```julia
if iswindows()
    some_complicated_thing(a)
end
```

Note that `islinux` and `isapple` are mutually exclusive subsets of `isunix`. Additionally,
there is a macro `@static` which makes it possible to use these functions to conditionally hide
invalid code, as demonstrated in the following examples.

Simple blocks:

```
ccall((@static iswindows() ? :_fopen : :fopen), ...)
```

Complex blocks:

```julia
@static if islinux()
    some_complicated_thing(a)
else
    some_different_thing(a)
end
```

When chaining conditionals (including if/elseif/end), the `@static` must be repeated for each
level (parentheses optional, but recommended for readability):

```julia
@static iswindows() ? :a : (@static isapple() ? :b : :c)
```
