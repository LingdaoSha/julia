# This file is a part of Julia. License is MIT: https://julialang.org/license

@test !Base.isunix(:Windows)
@test !Base.islinux(:Windows)
@test Base.islinux(:Linux)
@test Base.iswindows(:Windows)
@test Base.iswindows(:NT)
@test !Base.iswindows(:Darwin)
@test Base.isapple(:Darwin)
@test Base.isapple(:Apple)
@test !Base.isapple(:Windows)
@test Base.isunix(:Darwin)
@test Base.isunix(:FreeBSD)
@test_throws ArgumentError Base.isunix(:BeOS)
if !iswindows()
    @test Sys.windows_version() == v"0.0.0"
else
    @test Sys.windows_version() >= v"1.0.0-"
end

@test (@static true ? 1 : 2) === 1
@test (@static false ? 1 : 2) === 2
@test (@static if true 1 end) === 1
@test (@static if false 1 end) === nothing
@test (@static true && 1) === 1
@test (@static false && 1) === false
@test (@static true || 1) === true
@test (@static false || 1) === 1
