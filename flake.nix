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
  };
  
  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          # config.cudaCapabilities = [ "8.6" ];
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages
              (pythonPkgs: [
                # pythonPkgs.torch
                pythonPkgs.torchWithCuda
                (pythonPkgs.torchvision.override {
                  torch = pythonPkgs.pytorchWithCuda;
                })

                pythonPkgs.gdown
                pythonPkgs.tensorboardx
                pythonPkgs.opencv4
                # pythonPkgs.torchvision
                pythonPkgs.pandas

                pythonPkgs.python-lsp-server
              ])
            )
          ];
          # LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";
        };
      }
    );
}
