use crate::vec3::Vector3;
use geo::{
    geometry::{Coord, LineString, Point, Polygon},
    CoordNum, Intersects,
};
use itertools::Itertools;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::iter;
use std::ops::Range;
use std::thread;

const ADJACENCY_PATH: &str = "../data/MA_adjacency.json";
const BUFFER_BY: f64 = 0.0004;
const NUM_THREADS: u8 = 8;

type GeoId = usize;

pub fn get_adjacency(
    data: &[(GeoId, Polygon, usize)],
    geoid_to_ind: &BTreeMap<GeoId, usize>,
    tiles_dim: (usize, usize),
) -> (BTreeMap<GeoId, Vec<GeoId>>, Vec<Vec<usize>>) {
    let mut content = String::new();
    let adj = File::open(ADJACENCY_PATH)
        .map(|mut f| {
            f.read_to_string(&mut content).unwrap();
            serde_json::from_str(&content).unwrap()
        })
        .unwrap_or_else(|_| {
            let result = get_adjacency_with_threads(data, tiles_dim);

            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(ADJACENCY_PATH)
                .unwrap()
                .write_all(serde_json::to_string(&result).unwrap().as_bytes())
                .unwrap();

            result
        });

    let mut adj_list = vec![Vec::new(); geoid_to_ind.len()];
    for (geoid, adj_to) in adj.iter() {
        adj_list[*geoid_to_ind.get(geoid).unwrap()] = adj_to
            .into_iter()
            .map(|adj_geoid| *geoid_to_ind.get(adj_geoid).unwrap())
            .collect_vec()
    }

    (adj, adj_list)
}

fn get_adjacency_with_threads(
    data: &[(GeoId, Polygon, usize)],
    tiles_dim: (usize, usize),
) -> BTreeMap<GeoId, Vec<GeoId>> {
    let ind_to_geoid = data.iter().map(|(geoid, _, _)| *geoid).collect_vec();
    let chunk_len = data.len().div_ceil(NUM_THREADS as usize);
    let chunks = (0..NUM_THREADS as usize).map(|i| {
        let upper = (chunk_len * (i + 1)).min(data.len());
        (chunk_len * i)..upper
    });

    let tiles = get_tiles(tiles_dim.0, tiles_dim.1, data);
    let tile_tracts = tiles
        .iter()
        .map(|tile| {
            data.iter()
                .enumerate()
                .filter(|(_, (_, tract, _))| tile.intersects(tract))
                .map(|(i, _)| i)
                .collect_vec()
        })
        .collect_vec();

    let mut thread_adjs = Vec::<Vec<(GeoId, Vec<GeoId>)>>::new();
    thread::scope(|s| {
        let mut handles = chunks
            .into_iter()
            .map(|chunk_range| {
                let ind_to_geoid = &ind_to_geoid;
                let tiles = &tiles;
                let tile_tracts = &tile_tracts;
                s.spawn(move || {
                    get_adjacency_single_thread(data, chunk_range, ind_to_geoid, tiles, tile_tracts)
                })
            })
            .collect_vec();
        while let Some(handle) = handles.pop() {
            thread_adjs.push(handle.join().unwrap())
        }
    });

    BTreeMap::from_iter(thread_adjs.into_iter().flatten())
}

fn get_adjacency_single_thread(
    data: &[(GeoId, Polygon, usize)],
    tracts_to_process: Range<usize>,
    ind_to_geoid: &Vec<GeoId>,
    tiles: &Vec<Polygon>,
    tile_tracts: &Vec<Vec<usize>>,
) -> Vec<(GeoId, Vec<GeoId>)> {
    let mut adj = Vec::new();
    for (geoid, cur_tract, _) in data[tracts_to_process].iter() {
        let buffered_tract = buffer_polygon(&cur_tract, BUFFER_BY);
        adj.push((
            *geoid,
            tiles
                .iter()
                .enumerate()
                .filter(|(_, tile)| tile.intersects(&buffered_tract))
                .flat_map(|(i, _)| tile_tracts[i].iter())
                .unique()
                .filter(|tract| buffered_tract.intersects(&data[**tract].1))
                .map(|x| ind_to_geoid[*x])
                .collect_vec(),
        ));
    }

    adj
}

fn get_tiles(width: usize, height: usize, geometries: &[(GeoId, Polygon, usize)]) -> Vec<Polygon> {
    let mut points = geometries
        .iter()
        .flat_map(|(_, geometry, _)| geometry.exterior().points());
    let first = points.next().unwrap();
    let (mut maxx, mut maxy, mut minx, mut miny) = (first.x(), first.y(), first.x(), first.y());
    for p in points {
        maxx = maxx.max(p.x());
        minx = minx.min(p.x());
        maxy = maxy.max(p.y());
        miny = miny.min(p.y());
    }
    let pixel_dim = ((maxx - minx) / width as f64, (maxy - miny) / height as f64);

    (0..width * height)
        .map(|i| {
            let row = (i / width) as f64;
            let col = (i % width) as f64;
            Polygon::new(
                vec![
                    (col * pixel_dim.0 + minx, row * pixel_dim.1 + miny),
                    ((col + 1.0) * pixel_dim.0 + minx, row * pixel_dim.1 + miny),
                    (
                        (col + 1.0) * pixel_dim.0 + minx,
                        (row + 1.0) * pixel_dim.1 + miny,
                    ),
                    (col * pixel_dim.0 + minx, (row + 1.0) * pixel_dim.1 + miny),
                ]
                .into_iter()
                .collect_linestring(true),
                Vec::new(),
            )
        })
        .collect_vec()
}

fn buffer_polygon(polygon: &Polygon, dist: f64) -> Polygon {
    // collect bcs circular_cuple_windows clones
    let points = polygon.exterior().points().collect_vec();
    Polygon::new(
        points
            .iter()
            .skip(1) // polygon is closed
            .circular_tuple_windows()
            .map(|(pi, pj, pk)| buffer_angle(&pi, &pj, &pk, dist))
            .collect_linestring(false),
        Vec::new(),
    )
}

fn buffer_angle(pi: &Point, pj: &Point, pk: &Point, dist: f64) -> Point {
    let dist = -dist;
    let vi = v3_from_point(pi);
    let vj = v3_from_point(pj);
    let vk = v3_from_point(pk);
    let ij = (&vj - &vi).normalize();
    let kj = (&vj - &vk).normalize();
    let jk = (&vk - &vj).normalize();
    // counter-clockwise winding on Polygons by default
    let direction = ij.cross(&jk).normalize().z;
    point_from_v3(&(&((&ij + &kj) * direction * dist) + &vj))
}

pub trait CollectLineString<T: CoordNum> {
    fn collect_linestring(self, close: bool) -> LineString<T>
    where
        Self: Sized;
}

impl<I, T> CollectLineString<T> for I
where
    T: CoordNum,
    I: Iterator + Sized,
    I::Item: Clone + Into<Coord<T>> + Debug,
{
    fn collect_linestring(self, close: bool) -> LineString<T>
    where
        Self: Sized,
    {
        if close {
            let mut peekable = self.peekable();
            let first = peekable.peek().unwrap().clone();
            LineString::from_iter(peekable.chain(iter::once(first)))
        } else {
            LineString::from_iter(self)
        }
    }
}

fn v3_from_point(p: &Point) -> Vector3 {
    Vector3 {
        x: p.x(),
        y: p.y(),
        z: 0.0,
    }
}

fn point_from_v3(v: &Vector3) -> Point {
    Point::new(v.x, v.y)
}
