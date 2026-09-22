@echo off
rem Build the Nadrin/PBR renderer, D3D12 backend only (no Vulkan SDK needed).
rem Usage: build_pbr.bat   (from any directory)

setlocal
set VSDEV="D:\Tools\VS2022\Common7\Tools\VsDevCmd.bat"
set PBR=%~dp0..\PBR
set OUT=%PBR%\build

call %VSDEV% -arch=x64 -no_logo || exit /b 1
if not exist "%OUT%" mkdir "%OUT%"
pushd "%OUT%"

cl /nologo /std:c++17 /O2 /EHsc /MD /DENABLE_D3D12 /DNOMINMAX /DGLM_ENABLE_EXPERIMENTAL ^
  /I"%PBR%\lib\glfw\include" /I"%PBR%\lib\assimp\include" ^
  /I"%PBR%\lib\glm\include" /I"%PBR%\lib\stb\include" /I"%PBR%\lib\d3dx12" ^
  "%PBR%\src\common\main.cpp" "%PBR%\src\common\application.cpp" ^
  "%PBR%\src\common\image.cpp" "%PBR%\src\common\mesh.cpp" ^
  "%PBR%\src\common\utils.cpp" "%PBR%\src\common\optimus.cpp" ^
  "%PBR%\src\d3d12.cpp" "%PBR%\lib\stb\src\libstb.c" ^
  /Fe:pbr_d3d12.exe ^
  /link /LIBPATH:"%PBR%\lib\glfw\win64" /LIBPATH:"%PBR%\lib\assimp\win64" ^
  glfw3dll.lib assimp.lib d3d12.lib dxgi.lib d3dcompiler.lib ^
  user32.lib gdi32.lib shell32.lib ole32.lib || (popd & exit /b 1)

copy /y "%PBR%\lib\glfw\win64\glfw3.dll" . >nul
copy /y "%PBR%\lib\assimp\win64\assimp.dll" . >nul
popd
echo Built %OUT%\pbr_d3d12.exe
echo Run:  cd %PBR%\data ^&^& ..\build\pbr_d3d12.exe -d3d12
