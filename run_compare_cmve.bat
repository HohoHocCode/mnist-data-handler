@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

echo Compiling CMVE Verification...
nvcc -o compare_cmve.exe compare_cmve.cu -I. -std=c++17 -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -Xcompiler "/std:c++17" -arch=sm_86 -lcublas

if %errorlevel% neq 0 (
    echo Compilation Failed!
    exit /b %errorlevel%
)

echo Running CMVE Verification...
compare_cmve.exe
