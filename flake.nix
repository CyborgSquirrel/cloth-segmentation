{
  description = "A very basic flake";
  
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
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python311.withPackages
              (pythonPkgs: [
                pythonPkgs.torch
                pythonPkgs.gdown
                pythonPkgs.tensorboardx
                pythonPkgs.opencv4
                pythonPkgs.torchvision
                pythonPkgs.pandas
              ])
            )
          ];
          # LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}";
        };
      }
    );
}
