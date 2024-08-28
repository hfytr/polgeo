use annealer::{adjacency::*, init_precinct_with_threads, Annealer};
use anyhow::Result;
use csv::ReaderBuilder;
use dbase::FieldValue;
use geo::{
    algorithm::{bool_ops::BooleanOps, Area, ConvexHull},
    geometry::{Coord, LineString, MultiPolygon, Polygon},
};
use itertools::Itertools;
use shapefile::Shape;
use std::collections::BTreeMap;
use std::fs::File;
use std::iter::zip;

const NUM_DISTRICTS: usize = 9;
const ANNEAL_POP_THRESH: f32 = 0.50;
const T0: f32 = 0.5;
const INIT_POP_THRESH: f32 = 0.25;
const NUM_THREADS: u8 = 8;

fn get_data(
    shp_path: &str,
    id_col: &str,
    pop_path: &str,
    pop_col: &str,
    pop_id_col: &str,
) -> Result<()> {
    let mut reader = shapefile::Reader::from_path(shp_path)?;
    let mut data = Vec::new();
    let pop_data = load_pop_csv(pop_path, pop_col, pop_id_col)?;
    for item in reader.iter_shapes_and_records() {
        let (shape, record) = item?;
        if let Shape::Polygon(polygon) = shape {
            let polygon: Polygon = Polygon::new(
                polygon
                    .ring(0)
                    .unwrap()
                    .points()
                    .into_iter()
                    .map(|p| Coord { x: p.x, y: p.y })
                    .collect_linestring(false),
                Vec::new(),
            );
            let geoid = if let FieldValue::Character(Some(geoid_str)) = record.get(id_col).ok_or(
                anyhow::Error::msg(format!("record id_col {} does not exist", id_col)),
            )? {
                Ok(geoid_str.parse::<GeoId>()?)
            } else {
                Err(anyhow::Error::msg("geoid column incorrect type"))
            }?;
            data.push((geoid, polygon, *pop_data.get(&geoid).unwrap()));
        }
    }

    let geoid_to_ind = BTreeMap::from_iter(
        data.iter()
            .enumerate()
            .map(|(i, (geoid, _, _))| (*geoid, i)),
    );

    let mut pop_list = vec![0_usize; pop_data.len()];
    for (k, v) in pop_data.into_iter() {
        pop_list[*geoid_to_ind.get(&k).unwrap()] = v;
    }

    let (_adj, adj_list) = get_adjacency(data.as_slice(), &geoid_to_ind, (5, 15));

    let objective = |assignment: &[usize]| {
        let mut districts = vec![
            MultiPolygon::<f64>::new(vec![Polygon::<f64>::new(
                LineString::<f64>::from(Vec::<(f64, f64)>::new()),
                vec![]
            )]);
            NUM_DISTRICTS
        ];

        for ((_, tract, _), &district) in zip(data.iter(), assignment.iter()) {
            districts[district] =
                districts[district].union(&MultiPolygon::new(vec![tract.clone()]));
            print!("e");
        }

        -districts
            .iter()
            .map(|district| district.convex_hull().unsigned_area() / district.unsigned_area())
            .sum::<f64>()
    };

    let mut annealer = Annealer::from_starting_state(
        init_precinct_with_threads(
            adj_list.clone(),
            pop_list.clone(),
            NUM_DISTRICTS,
            INIT_POP_THRESH,
            NUM_THREADS,
        ),
        adj_list,
        NUM_DISTRICTS,
        pop_list,
        ANNEAL_POP_THRESH,
        objective,
        |x| x * T0 as f64,
    );

    annealer.anneal(100, NUM_THREADS);

    Ok(())
}

type GeoId = usize;
fn load_pop_csv(path: &str, pop_col: &str, id_col: &str) -> Result<BTreeMap<GeoId, usize>> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let pop_header_idx = get_header_idx(&mut reader, pop_col)?;
    let id_header_idx = get_header_idx(&mut reader, id_col)?;
    let mut result = BTreeMap::new();
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        result.insert(
            parse_geoid_str(
                &record
                    .get(id_header_idx)
                    .ok_or(anyhow::Error::msg(format!(
                        "value missing, header: {}, row: {}",
                        id_col, i
                    )))?
                    .to_string(),
            )?,
            record
                .get(pop_header_idx)
                .ok_or(anyhow::Error::msg(format!(
                    "value missing, header: {}, row: {}",
                    pop_col, i
                )))?
                .parse::<usize>()?,
        );
    }
    Ok(result)
}

fn _geopoly_to_shppoly(p: &Polygon) -> shapefile::Polygon {
    shapefile::Polygon::new(shapefile::PolygonRing::Outer(
        p.exterior()
            .points()
            .map(|p| shapefile::Point::new(p.x(), p.y()))
            .collect_vec(),
    ))
}

fn parse_geoid_str(geoid: &str) -> Result<GeoId> {
    Ok(geoid
        .chars()
        .skip(geoid.rfind("US").unwrap() + "US".len())
        .collect::<String>()
        .parse::<GeoId>()?)
}

fn get_header_idx(reader: &mut csv::Reader<File>, header: &str) -> Result<usize> {
    Ok(reader
        .headers()?
        .into_iter()
        .enumerate()
        .find(|(_, s)| *s == header)
        .ok_or(anyhow::Error::msg(format!("header: {} not in csv", header)))?
        .0)
}

fn main() {
    get_data(
        "../data/tl/tl_2023_25_tract.shp",
        "GEOID",
        "../data/tl/DECENNIALPL2020.P3-Data.csv",
        "P3_001N",
        "GEO_ID",
    )
    .unwrap();
}
