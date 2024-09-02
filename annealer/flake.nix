{
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

  outputs = { self, nixpkgs, rust-overlay, flake-utils, crane, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustpkg = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.default.override {
          extensions = [ "rust-src" "rust-analyzer" "rustfmt" ];
          targets = [ "arm-unknown-linux-gnueabihf" ];
        });
        lib = pkgs.lib;

        craneLib =
          (crane.mkLib pkgs).overrideToolchain rustpkg;

        projectName =
          (craneLib.crateNameFromCargoToml { cargoToml = ./Cargo.toml; }).pname;
        projectVersion = (craneLib.crateNameFromCargoToml {
          cargoToml = ./Cargo.toml;
        }).version;

        pythonVersion = pkgs.python311;
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
            maturin build --offline --target-dir ./target
          '';
          installPhase = old.installPhase + ''
            cp ./target/wheels/${wheelName} $out/
          '';
        });
      in rec {
        packages = rec {
          default = crateWheel; # The wheel itself

          # A python version with the library installed
          pythonEnv = pythonVersion.withPackages
            (ps: [ (lib.pythonPackage ps) ] ++ (with ps; [
              pandas
              networkx
              black
              isort
              tensorflow
              keras
              geopandas
              shapely
              # gurobipy
              matplotlib
              ipython
            ]));
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

        devShells.default = pkgs.mkShell rec {
          name = "python-dev";
          src = ./.;
          nativeBuildInputs = with pkgs; [
            pkg-config
            maturin
            packages.pythonEnv
            pyright
            # gurobi
          ];
        };

        RUST_BACKTRACE = 1;
        devShells.rust = pkgs.mkShell rec {
          name = "rust-dev";
          src = ./.;
          nativeBuildInputs = with pkgs; [
            pkg-config
            maturin
            packages.pythonEnv
            pyright
            # gurobi
          ];
          buildInputs = with pkgs; [
            openssl
            pkg-config
            rustpkg
          ];
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
