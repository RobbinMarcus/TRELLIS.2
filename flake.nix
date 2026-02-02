{
  description = "TRELLIS.2 Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "trellis2-shell";
          buildInputs = with pkgs; [
            python310
            uv
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
            cudaPackages.libcublas
            cudaPackages.libcusolver
            cudaPackages.libcurand
            cudaPackages.libcusparse
            cudaPackages.cuda_nvcc
            ninja
            git
            pkg-config
            gcc
            libGL
            libGLU
            glib
            libjpeg
            zlib
            stdenv.cc.cc.lib
            opencv
            xorg.libxcb
            xorg.libX11
            xorg.libXext
            xorg.libXrender
            libSM
            libICE
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.libGL
              pkgs.glib
              pkgs.zlib
              pkgs.libjpeg
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.cuda_nvcc
              pkgs.opencv
              pkgs.xorg.libxcb
              pkgs.xorg.libX11
              pkgs.xorg.libXext
              pkgs.xorg.libXrender
              pkgs.libSM
              pkgs.libICE
            ]}:/run/opengl-driver/lib:$LD_LIBRARY_PATH
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
            export TORCH_CUDA_ARCH_LIST="8.9"
            export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
            export EXTRA_LDFLAGS="-L/run/opengl-driver/lib"
            
            # Setup venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              uv venv .venv
            fi
            source .venv/bin/activate
            
            echo "TRELLIS.2 Environment Ready"
            echo "Run 'uv pip install -r requirements.txt' or similar to install python deps."
          '';
        };
      });
}
