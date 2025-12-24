@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -std=c++17 -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -Xcompiler "/std:c++17" -arch=sm_86 -I"d:\Imputing Library" debug_bgs.cu -o debug_bgs.exe -lcublas -lcusolver
if %ERRORLEVEL% NEQ 0 (
    echo Compilation Failed
    exit /b %ERRORLEVEL%
)
debug_bgs.exe
