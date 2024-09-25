{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.rust-overlay = {
    url = "github:oxalica/rust-overlay";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  inputs.crane = {
    url = "github:ipetkov/crane";
    inputs.nixpkgs.follows = "nixpkgs";
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
        pythonPackages = pkgs.python311Packages;
        wheelTail =
          "cp311-cp311-linux_x86_64"; # Change if pythonVersion changes
        wheelName = "${projectName}-${projectVersion}-${wheelTail}.whl";

        crateCfg = {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
          nativeBuildInputs = [ pythonVersion ];
        };

        # Build the library, then re-use the target dir to generate the wheel file with maturin
        crateWheel = (craneLib.buildPackage (crateCfg // {
          pname = projectName;
          version = projectVersion;
        })).overrideAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.maturin ];
          buildPhase = old.buildPhase + ''
            maturin build --release --offline --target-dir ./target
          '';
          installPhase = old.installPhase + ''
            ls target/wheels
            cp ./target/wheels/${wheelName} $out/
          '';
        });
      in rec {
        packages = {
          default = crateWheel; # The wheel itself
          # A python version with the library installed
          pythonEnv = pythonVersion.withPackages
            (ps: [ (lib.pythonPackage ps) ] ++ (with ps; [
              networkx
              black
              isort
              geopandas
              shapely
              matplotlib
              gurobipy
            ]));
        };

        lib = {
          # To use in other builds with the "withPackages" call
          pythonPackage = ps:
            ps.buildPythonPackage {
              pname = projectName;
              format = "wheel";
              version = projectVersion;
              src = "${crateWheel}/${wheelName}";
              doCheck = false;
              pythonImportsCheck = [ projectName ];
            };
        };

        devShells.default = pkgs.mkShell {
          name = "python-dev";
          src = ./.;
          nativeBuildInputs = with pkgs; [
            pkg-config
            maturin
            packages.pythonEnv
            pyright
            gurobi
          ];
        };

        devShells.rust = pkgs.mkShell {
          name = "rust-dev";
          src = ./.;
          nativeBuildInputs = with pkgs; [
            pkg-config
            maturin
            packages.pythonEnv
            pyright
          ];
          buildInputs = with pkgs; [
            openssl
            pkg-config
            rustpkg
          ];
        };
      });
}
