let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (python-pkgs: with python-pkgs; [
      pandas
      pyyaml
      black
      isort
      tensorflow
      keras
      scikit-learn
      geopandas
      rasterio
      gurobipy
      scipy
      shapely
      matplotlib
      rpy2
      # networkx
      # libpysal
    ]))
    R
    gurobi
    pyright
  ];
}
