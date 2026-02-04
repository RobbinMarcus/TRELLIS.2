@echo off
setlocal EnableDelayedExpansion

REM Setup script for TRELLIS.2 on Windows
REM Usage: setup.bat [OPTIONS]

set "NEW_ENV=false"
set "BASIC=false"
set "FLASHATTN=false"
set "CUMESH=false"
set "OVOXEL=false"
set "FLEXGEMM=false"
set "NVDIFFRAST=false"
set "NVDIFFREC=false"
set "HELP=false"

:parse_args
if "%~1"=="" goto :args_parsed
if "%~1"=="--new-env" set "NEW_ENV=true" & shift & goto :parse_args
if "%~1"=="--basic" set "BASIC=true" & shift & goto :parse_args
if "%~1"=="--flash-attn" set "FLASHATTN=true" & shift & goto :parse_args
if "%~1"=="--cumesh" set "CUMESH=true" & shift & goto :parse_args
if "%~1"=="--o-voxel" set "OVOXEL=true" & shift & goto :parse_args
if "%~1"=="--flexgemm" set "FLEXGEMM=true" & shift & goto :parse_args
if "%~1"=="--nvdiffrast" set "NVDIFFRAST=true" & shift & goto :parse_args
if "%~1"=="--nvdiffrec" set "NVDIFFREC=true" & shift & goto :parse_args
if "%~1"=="-h" set "HELP=true" & shift & goto :parse_args
if "%~1"=="--help" set "HELP=true" & shift & goto :parse_args
shift
goto :parse_args
:args_parsed

if "%HELP%"=="true" (
    echo Usage: setup.bat [OPTIONS]
    echo Options:
    echo   -h, --help              Display this help message
    echo   --new-env               Create a new virtual environment
    echo   --basic                 Install basic dependencies
    echo   --flash-attn            Install flash-attention ^(Check prerequisites^)
    echo   --cumesh                Install cumesh
    echo   --o-voxel               Install o-voxel
    echo   --flexgemm              Install flexgemm
    echo   --nvdiffrast            Install nvdiffrast
    echo   --nvdiffrec             Install nvdiffrec
    exit /b 0
)

REM Initialize MSVC environment
where cl >nul 2>nul
if %errorlevel% equ 0 goto :msvc_found
echo MSVC compiler (cl.exe) not found in PATH.
if not exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo Warning: vcvars64.bat not found. Extensions may fail to build.
    goto :msvc_found
)
echo Initializing VS 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
:msvc_found
set DISTUTILS_USE_SDK=1

REM Check for UV
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: 'uv' is not installed or not in PATH. Please install uv first.
    exit /b 1
)

REM Create/Activate Environment
if "%NEW_ENV%"=="true" (
    echo Creating new virtual environment...
    if exist .venv rmdir /s /q .venv
    uv venv .venv --python 3.10
    if %errorlevel% neq 0 (
         echo Failed to create venv. Trying default python...
         uv venv .venv
    )
)

if not exist .venv (
    echo Virtual environment not found. Please run with --new-env to create one, or ensure .venv exists.
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Bypass CUDA version check for extensions
set IGNORE_CUDA_VERSION=1
set TORCH_CUDA_ARCH_LIST=8.9

REM Check for CUDA compiler
nvcc --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Warning: nvcc not found. CUDA extensions might fail to build.
) else (
    echo Found CUDA compiler.
)

REM Install PyTorch if new env
if "%NEW_ENV%"=="true" (
    echo Installing PyTorch...
    REM Using UV pip to install pytorch for CUDA 12.4 as per setup.sh (adjust if needed for 13.0 or use what works)
    uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
)

REM Install Basic Dependencies
if "%BASIC%"=="true" (
    echo Installing basic dependencies...
    uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard psutil
    uv pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
    REM pillow-simd often fails on Windows, skip or rely on pillow (which imagesio pulls)
    uv pip install pillow
    uv pip install kornia timm
)

if not exist extensions mkdir extensions

REM Flash Attention
if "%FLASHATTN%"=="true" (
    echo Installing flash-attn...
    uv pip install flash-attn==2.7.3 --no-build-isolation
    if %errorlevel% neq 0 (
        echo [FLASHATTN] Installation failed. It is common on Windows.
        echo Please try installing manually or use pre-built wheels if available.
    )
)

REM NVDiffRast
if "%NVDIFFRAST%"=="true" (
    echo Installing nvdiffrast...
    if not exist extensions\nvdiffrast (
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git extensions\nvdiffrast
    )
    uv pip install extensions\nvdiffrast --no-build-isolation
)

REM NVDiffRec
if "%NVDIFFREC%"=="true" (
    echo Installing nvdiffrec...
    if not exist extensions\nvdiffrec (
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git extensions\nvdiffrec
    )
    uv pip install extensions\nvdiffrec --no-build-isolation
)

REM CuMesh
if "%CUMESH%"=="true" (
    echo Installing CuMesh...
    if not exist extensions\CuMesh (
        git clone https://github.com/JeffreyXiang/CuMesh.git extensions\CuMesh --recursive
    )
    uv pip install extensions\CuMesh --no-build-isolation
)

REM FlexGEMM
if "%FLEXGEMM%"=="true" (
    echo Installing FlexGEMM...
    if not exist extensions\FlexGEMM (
        git clone https://github.com/JeffreyXiang/FlexGEMM.git extensions\FlexGEMM --recursive
    )
    uv pip install extensions\FlexGEMM --no-build-isolation
)

REM O-Voxel
if "%OVOXEL%"=="true" (
    echo Installing o-voxel...
    if not exist extensions\o-voxel (
        xcopy /E /I /Y o-voxel extensions\o-voxel
    )
    uv pip install extensions\o-voxel --no-build-isolation
)

echo Setup complete.
