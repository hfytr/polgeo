
let
  pkgs = import <nixpkgs> {};
  python = pkgs.python311;
in pkgs.mkShell {
  packages = with pkgs; [
    (python.withPackages (python-pkgs: with python-pkgs; [
      pandas
      pip
      pyyaml
      black
      isort
      tensorflow
      keras
      geopandas
      gurobipy
      scipy
      scikit-learn
      shapely
      matplotlib
      rpy2
    ]))
    R
    python
    gurobi
    pyright
  ];
}
