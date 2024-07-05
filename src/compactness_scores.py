from math import pi

import geopandas as gp
import numpy as np
from shapely import LineString, Polygon


def all_measures(d: gp.GeoDataFrame, inplace: bool):
    result = gp.GeoDataFrame()
    # measures = ["convex_hull", "boyce_clark", "reock", "length_width", "polsby_popper"]
    # names longer than 10 characters get truncated in shp files
    measures = ["con_hull", "reock", "len_width", "polsby_pop"]
    for measure in measures:
        if inplace:
            single_measure(d, True, measure)
        else:
            single_measure(result, True, measure)
    if inplace:
        return result


def single_measure(d: gp.GeoDataFrame, inplace: bool, measure: str):
    measure_to_func = {
        "con_hull": __convex_hull,
        # "boyce_clark": __boyce_clark,
        "reock": __reock,
        "len_width": __length_width,
        "polsby_pop": __polsby_popper,
    }
    if inplace:
        d[measure] = measure_to_func[measure](d)
    else:
        return measure_to_func[measure](d)


def __convex_hull(d: gp.GeoDataFrame):
    hull_area = d["geometry"].apply(lambda x: x.convex_hull.area)
    return d.area / hull_area


def __boyce_clark(d: gp.GeoDataFrame):
    def points_on_surface(p: Polygon, num_points: int):
        line = LineString(list(p.exterior.coords))
        dists = np.linspace(0, line.length, num_points)
        return [line.interpolate(dist) for dist in dists]

    def single_avg_dist(geo, num_points):
        pass

    return d.geometry.apply(single_avg_dist, num_points=100)


def __reock(d: gp.GeoDataFrame):
    bounding = d["geometry"].minimum_bounding_circle()
    radius = bounding.boundary.distance(bounding.centroid)
    bounding_area = radius * radius * pi
    return d["area"] / bounding_area


def __length_width(d: gp.GeoDataFrame):
    return (d.bounds["maxx"] - d.bounds["minx"]).mul(
        d.bounds["maxy"] - d.bounds["miny"]
    )


def __polsby_popper(d: gp.GeoDataFrame):
    r = d["perimeter"] / (2 * pi)
    return d["area"] / (r * r * pi)
