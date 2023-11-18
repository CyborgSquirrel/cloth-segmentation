{
  description = "A very basic flake";

  nixConfig.extra-substituters = [
    "https://cuda-maintainers.cachix.org"
    "https://numtide.cachix.org"
  ];
  nixConfig.extra-trusted-public-keys = [
    "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    "numtide.cachix.org-1:2ps1kLBUWjxIneOy1Ik6cQjb41X0iXVXeHigGmycPPE="
  ]; 
  
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pylibjpeg-src.url = "github:pydicom/pylibjpeg";
    pylibjpeg-src.flake = false;

    # pylibjpeg-openjpeg-src.url = "github:pydicom/pylibjpeg-openjpeg";
    pylibjpeg-openjpeg-src.url = "https://github.com/pydicom/pylibjpeg-openjpeg.git";
    pylibjpeg-openjpeg-src.flake = false;
    pylibjpeg-openjpeg-src.type = "git";
    pylibjpeg-openjpeg-src.submodules = true;
  };
  
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    # python modules
    pylibjpeg-src,
    pylibjpeg-openjpeg-src,
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          # config.cudaCapabilities = [ "8.6" ];
        };
        pylibjpeg = pkgs.python3.pkgs.buildPythonPackage {
          pname = "pylibjpeg";
          version = "1.4.0";
          src = pylibjpeg-src;
          doCheck = false;
          propagatedBuildInputs = [
            pkgs.python3.pkgs.numpy
          ];
        };
        pylibjpeg-openjpeg = pkgs.python3.pkgs.buildPythonPackage {
          pname = "pylibjpeg-openjpeg";
          version = "1.3.2";
          src = pylibjpeg-openjpeg-src;
          doCheck = false;
          dontUseCmakeConfigure = true;
          nativeBuildInputs = [
            pkgs.cmake
          ];
          # patches = [
          #   ./dontbuild.patch
          # ];
          propagatedBuildInputs = [
            pkgs.python3.pkgs.numpy
            pkgs.openjpeg
          ];
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages
              (pythonPkgs: [
                pylibjpeg
                pylibjpeg-openjpeg

                # pythonPkgs.torch
                pythonPkgs.torchWithCuda
                (pythonPkgs.torchvision.override {
                  torch = pythonPkgs.pytorchWithCuda;
                })

                pythonPkgs.pip

                pythonPkgs.gdown
                pythonPkgs.tensorboardx
                pythonPkgs.opencv4
                # pythonPkgs.torchvision
                pythonPkgs.pandas
                pythonPkgs.polars
                pythonPkgs.pydicom
                pythonPkgs.scikit-image
                pythonPkgs.pylibjpeg-libjpeg

                pythonPkgs.python-lsp-server
                pythonPkgs.pyls-isort
                pythonPkgs.pycodestyle

                pythonPkgs.ipython
              ])
            )
          ];
          # LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";
        };
      }
    );
}
