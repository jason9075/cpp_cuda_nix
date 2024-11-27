{
  description = "Development environment for projects";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ ... }:
    let
      system = "x86_64-linux";
      pkgs = import inputs.nixpkgs {
        inherit system;
        config = { allowUnfree = true; };
      };
      pkgs-unstable = import inputs.nixpkgs-unstable {
        inherit system;
        config = { allowUnfree = true; };
      };
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        name = "cuda-env";
        nativeBuildInputs = with pkgs; [
          gcc12 # default is v13. gcc versions later than 12 are not supported for cuda.
          cmake
          cudatoolkit
          cudaPackages.cudnn
          entr
          addOpenGLRunpath
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          # export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
          echo "Welcome to CUDA project environment!"
        '';
      };
    };
}

# ref: https://nixos.wiki/wiki/CUDA
