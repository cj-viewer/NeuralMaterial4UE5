@echo off
rem Launch the D3D12 test renderer against a material package. Runs from anywhere.
rem Examples:
rem   test.bat                                   (default package: PBR\data\material)
rem   test.bat -material trainer\output\my_run   (path relative to where you run this)
rem Extra arguments are passed straight to the renderer (see -material / keys N, D, C).
setlocal
set EXE=%~dp0PBR\build\pbr_d3d12.exe
if not exist "%EXE%" (
    echo Renderer not built yet - run trainer\build_pbr.bat first.
    exit /b 1
)
"%EXE%" -d3d12 -datadir "%~dp0PBR\data" -material "%~dp0PBR\data\material" %*
