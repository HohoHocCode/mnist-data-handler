@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
echo Starting Final Compilation...
nvcc -std=c++17 -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -Xcompiler "/std:c++17" -arch=sm_86 -I"d:\Imputing Library" -I"d:\Imputing Library\ImputingLibrary" "d:\Imputing Library\run_benchmarks.cu" -o "d:\Imputing Library\impute_final.exe" -lcublas -lcusolver
if %ERRORLEVEL% NEQ 0 (
    echo Compilation Failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo Compilation Succeeded!
dir "d:\Imputing Library\impute_final.exe"
