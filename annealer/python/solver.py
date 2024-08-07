import logging
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import to_rgba
from shapely import unary_union
from shapely.geometry import MultiPolygon, Polygon

import annealer


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


def graph_adj(gdf, adj):
    def create_graph(adj_list):
        G = nx.Graph()
        for key, neighbors in adj_list.items():
            for neighbor in neighbors:
                G.add_edge(key, neighbor)
        return G

    G = create_graph(adj)
    fig, ax = plt.subplots(figsize=(12, 12))

    gdf.plot(ax=ax, edgecolor="black", color="lightgrey")

    pos = {idx: row["geometry"].centroid.coords[0] for idx, row in gdf.iterrows()}

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_color="blue",
        edge_color="red",
        node_size=1,
        font_color="white",
        alpha=0.7,
        edgecolors="black",
    )

    plt.savefig("census_tracts_with_graph.png", dpi=300)


def shp_from_assign(gdf, col):
    ax = gdf.plot(column=col, legend=True, figsize=(10, 6), cmap="viridis")
    plt.savefig("initial_districts.png", dpi=300)
    plt.show()


def objective(
    x: list[list[bool]], ind_to_geoid: dict[int, int], tracts: [gpd.GeoDataFrame]
):
    result = 0
    for district in x:
        dist_start = time.perf_counter() * 1000
        geoid_list = [
            ind_to_geoid[i] for i, included in enumerate(district) if included
        ]
        mk_geoids = time.perf_counter() * 1000
        selected_tracts = tracts[tracts.index.isin(geoid_list)]
        select = time.perf_counter() * 1000
        combined_geometry = unary_union(selected_tracts.geometry)
        combine = time.perf_counter() * 1000
        result += combined_geometry.area / combined_geometry.convex_hull.area
        divide = time.perf_counter() * 1000
        print(
            f"total: {divide - dist_start} mk_geoids: {mk_geoids - dist_start}, select: {select - mk_geoids}, combine: {combine - select}, divide: {divide - combine}, "
        )
    return result


def solve():
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
    T0 = 0.5
    POP_THRESH = 0.6
    NUM_THREADS = 8
    shp_path = "../data/initial_assign.shp"
    if not os.path.isfile(shp_path):
        print(tracts.shape)
        initial_assignment = annealer.init_precinct(
            adj_list, population_list, NUM_DISTRICTS, POP_THRESH, NUM_THREADS
        )
        tracts["INIT_DISTS"] = tracts.index.map(
            {
                ind_to_geoid.get(node): district
                for node, district in enumerate(initial_assignment)
            }
        )
        print(tracts.columns)
        tracts.to_file(shp_path)
        shp_from_assign(tracts, "INIT_DISTS")
    else:
        tracts = gpd.read_file(shp_path)
        tracts.set_index("GEOID", inplace=True, drop=False)
        initial_assignment = [
            tracts.loc[ind_to_geoid.get(i)]["INIT_DISTS"]
            for i in range(len(population_list))
        ]
        # shp_from_assign(tracts, "INIT_DISTS")
        print(tracts.columns)
        print(tracts["INIT_DISTS"])
    annealer.optimize_func(
        objective_raw=lambda x: objective(x, ind_to_geoid, tracts),
        temperature_raw=lambda x: T0 * x,
        precinct_in=initial_assignment,
        adj=adj_list,
        num_districts=NUM_DISTRICTS,
        population=population_list,
        num_steps=100,
        pop_thresh=POP_THRESH,
    )
    logging.shutdown()


if __name__ == "__main__":
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
    T0 = 0.5
    POP_THRESH = 0.6
    NUM_THREADS = 8
    shp_path = "../data/initial_assign.shp"
    if not os.path.isfile(shp_path):
        print(tracts.shape)
        initial_assignment = annealer.init_precinct(
            adj_list, population_list, NUM_DISTRICTS, POP_THRESH, NUM_THREADS
        )
        tracts["INIT_DISTS"] = tracts.index.map(
            {
                ind_to_geoid.get(node): district
                for node, district in enumerate(initial_assignment)
            }
        )
        print(tracts.columns)
        tracts.to_file(shp_path)
        shp_from_assign(tracts, "INIT_DISTS")
    else:
        tracts = gpd.read_file(shp_path)
        tracts.set_index("GEOID", inplace=True, drop=False)
        initial_assignment = [
            tracts.loc[ind_to_geoid.get(i)]["INIT_DISTS"]
            for i in range(len(population_list))
        ]
        # shp_from_assign(tracts, "INIT_DISTS")
        print(tracts.columns)
        print(tracts["INIT_DISTS"])
    annealer.optimize_func(
        objective_raw=lambda x: objective(x, ind_to_geoid, tracts),
        temperature_raw=lambda x: T0 * x,
        precinct_in=initial_assignment,
        adj=adj_list,
        num_districts=NUM_DISTRICTS,
        population=population_list,
        num_steps=100,
        pop_thresh=POP_THRESH,
    )
    logging.shutdown()
