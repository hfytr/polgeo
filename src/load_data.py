import geopandas as gp
import pandas as pd
import rasterio
from rasterio.features import rasterize

from compactness_scores import all_measures
from rdata import df_from_rdata


def load_to_file(r_path, output_shp_path, output_csv_path, var_name) -> gp.GeoDataFrame:
    data = df_from_rdata(r_path, var_name)
    data = data.dropna()
    data = data.rename(columns={"compactness": "rank"})
    _parse_district_ids(data, "district")
    data = gp.GeoDataFrame(data)
    data = load_districts(data)
    data["idx"] = range(data.shape[0])
    data = data.set_index("idx")
    pixels = rasterize_data(data)
    pixels.to_csv(output_csv_path)
    all_measures(data, True)
    data.to_file(output_shp_path)
    return data


def load_districts(d: pd.DataFrame) -> gp.GeoDataFrame:
    id_map = {}
    for chamber, path in [
        ["L", "data/districts2010/US_stleg_lo_2010.shp"],
        ["U", "data/districts2010/US_stleg_up_2010.shp"],
        ["C", "data/districts2010/US_cd111th_2010.shp"],
    ]:
        districts: gp.GeoDataFrame = gp.read_file(path)
        id_map.update(_id_geometry_map(districts, chamber))

    filter_df = d["district"].apply(lambda x: x in id_map)
    d["district"] = d["district"][filter_df]
    d = d.dropna()
    d["geometry"] = d["district"].apply(lambda x: id_map[x][0]).copy()
    d["area"] = d["district"].apply(lambda x: id_map[x][1]).copy()
    d = gp.GeoDataFrame(d, geometry=d["geometry"])
    d["perimeter"] = d["geometry"].length
    return d


def _left_pad(x: str) -> str:
    if len(x) > 4:
        raise Exception("x len greater than 4")
    result = "0" * (4 - len(x)) + x
    return result


def _id_geometry_map(districts: pd.DataFrame, chamber: str) -> dict:
    to_return = {}
    if not "CD111FP" in districts.keys():
        districts = districts.rename(
            columns={"SLDLST10": "CD111FP", "SLDUST10": "CD111FP"}
        )
    for row in districts.itertuples():
        key = "_".join(
            [
                _left_pad(getattr(row, "STATEFP10")),
                chamber,
                _left_pad(getattr(row, "CD111FP")),
                "2010",
            ]
        )
        to_return[key] = [getattr(row, "geometry"), getattr(row, "Shape_area")]

    return to_return


def _parse_district_ids(d: pd.DataFrame, key) -> None:
    def parse_single_id(s: str):
        return "_".join(
            [_left_pad(x) if i != 1 else x for i, x in enumerate(s.split("_")[0:4])]
        )

    d[key] = d[key].apply(parse_single_id)


def rasterize_data(d: gp.GeoDataFrame):
    xmin, ymin, xmax, ymax = d.total_bounds
    resolution = 10000

    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)

    transform = rasterio.transform.from_origin(xmin, ymax, resolution, resolution)

    print(d["geometry"].shape)

    raster = d["geometry"].apply(
        lambda x: rasterize(
            [(x, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="float32",
        ).flatten()
    )
    raster = pd.DataFrame(raster.tolist()).add_prefix("p")
    print("raster")
    print(raster)
    print("d")
    print(d)
    print("joined")
    print(d.join(raster))

    d = d.join(raster, how="left")
    print(d)

    return d
