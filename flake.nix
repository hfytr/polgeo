# From https://github.com/litchipi/nix-build-templates/blob/6e4961dc56a9bbfa3acf316d81861f5bd1ea37ca/rust/maturin.nix
# See also https://discourse.nixos.org/t/pyo3-maturin-python-native-dependency-management-vs-nixpkgs/21739/2
{
  # Build Pyo3 package
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.rust-overlay.overlays.default ];
        };
        lib = pkgs.lib;

        # Get a custom rust toolchain
        customRustToolchain = pkgs.rust-bin.stable."1.70.0".default;
        craneLib =
          (inputs.crane.mkLib pkgs).overrideToolchain customRustToolchain;

        projectName =
          (craneLib.crateNameFromCargoToml { cargoToml = ./annealer/Cargo.toml; }).pname;
        projectVersion = (craneLib.crateNameFromCargoToml {
          cargoToml = ./annealer/Cargo.toml;
        }).version;

        pythonVersion = pkgs.python310;
        wheelTail =
          "cp310-cp310-manylinux_2_34_x86_64"; # Change if pythonVersion changes
        wheelName = "${projectName}-${projectVersion}-${wheelTail}.whl";

        crateCfg = {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          nativeBuildInputs = [ pythonVersion ];
        };

        # Build the library, then re-use the target dir to generate the wheel file with maturin
        crateWheel = (craneLib.buildPackage (crateCfg // {
          pname = projectName;
          version = projectVersion;
          # cargoArtifacts = crateArtifacts;
        })).overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.maturin ];
          buildPhase = old.buildPhase + ''
            maturin build --offline --target-dir ./annealer/target
          '';
          installPhase = old.installPhase + ''
            cp ./annealer/target/wheels/${wheelName} $out/
          '';
        });
      in rec {
        packages = rec {
          default = crateWheel; # The wheel itself

          # A python version with the library installed
          pythonEnv = pythonVersion.withPackages
            (ps: [ (lib.pythonPackage ps) ] ++ (with ps; [ ipython ]));
        };

        lib = {
          # To use in other builds with the "withPackages" call
          pythonPackage = ps:
            ps.buildPythonPackage rec {
              pname = projectName;
              format = "wheel";
              version = projectVersion;
              src = "${crateWheel}/${wheelName}";
              doCheck = false;
              pythonImportsCheck = [ projectName ];
            };
        };

        devShells = rec {
          rust = pkgs.mkShell {
            name = "rust-env";
            src = ./annealer/.;
            nativeBuildInputs = with pkgs; [ pkg-config rust-analyzer maturin ];
          };
          python = let
          in pkgs.mkShell {
            name = "python-env";
            src = ./annealer.;
            nativeBuildInputs = [ packages.pythonEnv ];
          };
          default = rust;
        };

        apps = rec {
          ipython = {
            type = "app";
            program = "${packages.pythonEnv}/bin/ipython";
          };
          default = ipython;
        };
      });
}
