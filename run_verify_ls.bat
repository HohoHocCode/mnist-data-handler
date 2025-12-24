@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -std=c++17 -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -Xcompiler "/std:c++17" -arch=sm_86 -I"d:\Imputing Library" verify_ls_only.cu -o verify_ls_only.exe -lcublas -lcusolver
if %ERRORLEVEL% NEQ 0 (
    echo Compilation Failed
    exit /b %ERRORLEVEL%
)
verify_ls_only.exe
