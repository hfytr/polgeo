import json
import logging
import numpy as np
import os
import random
import time

import geopandas as gpd
import highspy
import numpy as np
import matplotlib.pyplot as plt
from shapely import unary_union

from annealer import AnnealerService, init_precinct


def get_adj(gdf, unique_col):
    gdf = gdf.to_crs(3857)
    gdf_buffer = gdf.copy(deep=True)
    gdf_buffer["geometry"] = gdf.buffer(1)
    test_intersection = gpd.overlay(gdf_buffer, gdf, how="intersection")
    test_intersection_tuples = tuple(
        zip(test_intersection[unique_col + "_1"], test_intersection[unique_col + "_2"])
    )

    final_dict = {}
    for val in test_intersection_tuples:
        if val[0] != val[1]:
            if val[0] in list(final_dict.keys()):
                holder = final_dict[val[0]]
                holder.append(val[1])
                final_dict[val[0]] = holder
            else:
                final_dict[val[0]] = [val[1]]

    for val in [i for i in gdf[unique_col] if i not in list(final_dict.keys())]:
        final_dict[val] = []
    return final_dict


def get_data(tracts_path, pop_path):
    tracts = gpd.read_file(tracts_path)
    population = (
        gpd.read_file(pop_path)
        .iloc[1:, :][["GEO_ID", "P3_001N"]]
        .rename(columns={"GEO_ID": "GEOID", "P3_001N": "POP"})
    )
    tracts["GEOID"] = tracts["GEOID"].apply(lambda x: int(x))
    population["POP"] = population["POP"].apply(lambda x: int(x))
    population["GEOID"] = population["GEOID"].apply(
        lambda x: int(x[x.find("US") + len("US") :])
    )
    tracts = tracts.merge(population, how="left", on="GEOID")
    return tracts


def shp_from_assign(gdf, col):
    ax = gdf.plot(column=col, legend=True, figsize=(10, 6), cmap="viridis")
    plt.savefig("initial_districts.png", dpi=300)
    plt.show()


def solve_with_tracts():
    FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    tracts = get_data(
        "../data/tl/tl_2023_25_tract.shp", "../data/tl/DECENNIALPL2020.P3-Data.csv"
    )
    adj = get_adj(tracts, "GEOID")
    tracts.set_index("GEOID", inplace=True, drop=True)
    ind_to_geoid = {}
    geoid_to_ind = {}
    adj_list = []
    population_list = []
    for i, key in enumerate(tracts.index):
        ind_to_geoid.update({i: key})
        geoid_to_ind.update({key: i})
    adj_list = [None for _ in range(len(ind_to_geoid))]
    for key, value in adj.items():
        adj_list[geoid_to_ind.get(key)] = [geoid_to_ind.get(v) for v in value]
        population_list.append(tracts.loc[key]["POP"])

    test_adj = {
        ind_to_geoid.get(i): [ind_to_geoid.get(v) for v in val]
        for i, val in enumerate(adj_list)
    }
    graph_adj(tracts, test_adj)

    NUM_DISTRICTS = 9
    POP_THRESH = 0.6
    NUM_THREADS = 8
    shp_path = "../data/initial_assign.shp"
    if not os.path.isfile(shp_path):
        initial_assignment = annealer.init_precinct(
            adj_list, population_list, NUM_DISTRICTS, POP_THRESH, NUM_THREADS
        )
        tracts["INIT_DISTS"] = tracts.index.map(
            {
                ind_to_geoid.get(node): district
                for node, district in enumerate(initial_assignment)
            }
        )
        tracts.to_file(shp_path)
        shp_from_assign(tracts, "INIT_DISTS")
    else:
        tracts = gpd.read_file(shp_path)
        tracts.set_index("GEOID", inplace=True, drop=False)
        initial_assignment = [
            tracts.loc[ind_to_geoid.get(i)]["INIT_DISTS"]
            for i in range(len(population_list))
        ]
        shp_from_assign(tracts, "INIT_DISTS")


T0 = 0.1
INIT_POP_THRESH = 0.01
POP_THRESH = 0.9
NUM_THREADS = 8


def pprint_assignment(assignment: list[int], d: int, width: int, height: int):
    grid = []
    for i, district in enumerate(assignment):
        if i % width == 0:
            grid.append([])
        grid[-1].append(str(round(district)))

    print("\n".join(["".join(gridi) for gridi in grid]))


def run_lp(
    assignment_raw: list[int],
    adj: list[list[int]],
    populations: list[int],
    d: int,
    width: int,
    height: int,
    pop_thresh: float,
):
    h = highspy.Highs()

    inf = highspy.kHighsInf
    n = len(adj)
    highest_pop = h.addVariable(lb=0, ub=inf, type=highspy.HighsVarType.kInteger)
    lowest_pop = h.addVariable(lb=0, ub=inf, type=highspy.HighsVarType.kInteger)
    x, w, y, abs_diff, district_pops = [], [], [], [], []
    for i in range(d):
        district_pops.append(
            h.addVariable(lb=0, ub=inf, type=highspy.HighsVarType.kInteger)
        )
        y.append({})
        x.append([])
        w.append([])
        abs_diff.append([])
        for j in range(n):
            x[-1].append(h.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger))
            w[-1].append(h.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger))
            abs_diff[-1].append(
                h.addVariable(lb=0, ub=1, type=highspy.HighsVarType.kInteger)
            )
            for k in adj[j]:
                y[-1].update(
                    {
                        (j, k): h.addVariable(
                            lb=0, ub=inf, type=highspy.HighsVarType.kContinuous
                        )
                    }
                )

    assignment = [
        [1 if assignment_raw[j] == i else 0 for j in range(n)] for i in range(d)
    ]

    # yes j is the first index, sue me
    # one district per node
    for j in range(n):
        h.addConstr(sum(x[i][j] for i in range(d)) == 1)
    # population balance
    h.addConstr(highest_pop * pop_thresh <= lowest_pop)

    for i in range(d):
        # population constraints
        h.addConstr(district_pops[i] == sum(x[i][j] * populations[j] for j in range(n)))
        h.addConstr(highest_pop >= district_pops[i])
        h.addConstr(lowest_pop <= district_pops[i])

        # one sink per district
        h.addConstr(sum(w[i][j] for j in range(n)) == 1)

        for j in range(n):
            # sink is in district
            h.addConstr(w[i][j] <= x[i][j])

            # set absolute difference with starting assignment
            # used for objective
            if bool(round(assignment[i][j])):
                h.addConstr(abs_diff[i][j] == 1 - x[i][j])
            else:
                h.addConstr(abs_diff[i][j] == x[i][j])

            # net flow constraint
            h.addConstr(
                sum(y[i][(j, k)] for k in adj[j]) - sum(y[i][(k, j)] for k in adj[j])
                >= x[i][j] - (n - d) * w[i][j]
            )
            for k in adj[j]:
                # flow is between nodes of same, correct district
                h.addConstr(y[i][(j, k)] <= x[i][j] * (n - d))
                h.addConstr(y[i][(j, k)] <= x[i][k] * (n - d))

    # as close as possible to input assignment
    h.minimize(sum(abs_diff[i][j] for i in range(d) for j in range(n)))

    if h.getModelStatus() == highspy.HighsModelStatus.kInfeasible:
        print("sol infeasible")
        exit()

    fit = h.getInfo().objective_function_value
    values = list(reversed(h.getSolution().col_value))

    # highs doesn't store variable names, so we need to reconstruct from vector
    highest_pop = values.pop()
    lowest_pop = values.pop()
    for i in range(d):
        district_pops[i] = values.pop()
        for j in range(n):
            x[i][j] = values.pop()
            w[i][j] = values.pop()
            abs_diff[i][j] = values.pop()
            for k in adj[j]:
                y[i][(j, k)] = values.pop()

    solution = [d] * n
    for i in range(d):
        for j in range(n):
            if bool(round(x[i][j])):
                solution[j] = i
    print(fit);
    pprint_assignment(solution, d, width, height)

    return (solution, fit)


def test_grid(width, height, population, num_districts):
    def make_cell(i):
        row = i / width
        col = i % height
        return (
            [
                (col, row),
                (col + 1.0, row),
                (col + 1.0, row + 1.0),
                (col, row + 1.0),
            ],
            [],
        )

    cells = list(map(make_cell, range(width * height)))

    def cell_adj(i):
        row = i // width
        col = i % width
        result = []
        for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if 0 <= row + offset[0] < height and 0 <= col + offset[1] < width:
                result.append((row + offset[0]) * width + (col + offset[1]))
        return result

    adj: list[list[int]] = list(map(cell_adj, range(width * height)))

    assignment = init_precinct(
        adj,
        population,
        num_districts,
        INIT_POP_THRESH,
        NUM_THREADS,
    )

    (assignment, _) = run_lp(
        assignment,
        adj,
        population,
        num_districts,
        width,
        height,
        POP_THRESH,
    )

    annealer = AnnealerService(
        assignment,
        adj,
        cells,
        num_districts,
        population,
        POP_THRESH,
        False,
        T0,
    )
    hist: list[tuple[list[float], float]] = []
    for i in range(10):
        (assignment, anneal_cycle_hist) = annealer.anneal(
            assignment, 100, NUM_THREADS
        )
        print(anneal_cycle_hist)
        pprint_assignment(assignment, num_districts, width, height)
        anneal_cycle_hist = [score for _, _, score in anneal_cycle_hist]
        (assignment, fit) = run_lp(
            assignment,
            adj,
            population,
            num_districts,
            width,
            height,
            POP_THRESH,
        )
        hist.append((anneal_cycle_hist, fit))

    return (assignment, hist)


def fetch_grid_data(width: int, height: int, num_districts: int, path: str, use_precalculated: bool):
    path_exists = os.path.exists(path)
    if path_exists and use_precalculated:
        with open(path, "r") as f:
            data = json.load(f)
            assignment = data["assignment"]
            hist = data["hist"]
            if data["width"] != width or data["height"] != height:
                use_precalculated = False

    if not use_precalculated or not path_exists:
        rand_pop = [round(random.gauss(10, 2)) for _ in range(width * height)]
        print(rand_pop)
        assignment, hist = test_grid(width, height, rand_pop, num_districts)
        with open(path, "w") as f:
            data_json = {
                "assignment": assignment,
                "hist": hist,
                "width": width,
                "height": height,
            }
            json.dump(data_json, f)

    return (assignment, hist)


if __name__ == "__main__":
    side = 10
    assignment, hist = fetch_grid_data(side, side, 3, "../data/solutions.json", False)

    sim_annealer_history = []
    mlp_fit_indices = []
    mlp_fits = []

    current_index = 0

    for annealer_history, mlp_fit in hist:
        sim_annealer_history.extend(annealer_history)
        mlp_fits.append(mlp_fit)
        mlp_fit_indices.append(current_index)
        current_index += len(annealer_history)

    fig, ax1 = plt.subplots()
    ax1.plot(sim_annealer_history, label="Objective Function", color="blue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Simulated Annealer History", color="blue")

    ax2 = ax1.twinx()
    ax2.scatter(mlp_fit_indices, mlp_fits, color="red", label="MILP Fit", zorder=5)
    ax2.set_ylabel("MILP Fit", color="red")

    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Simulated Annealer History with MILP Fit")
    fig.legend(loc="upper right")
    plt.show()
