use crate::csv::write_csv;
use rand::{rngs::ThreadRng, Rng};

use indicatif::ProgressBar;
use shapefile::dbase::FieldValue;
use shapefile::{Point, Polygon, PolygonRing};
use shapefile::{Reader, Shape};

pub struct Rasterizer {
    raster: Vec<Vec<f64>>,
    keys: Vec<String>,
    id: Vec<String>,
}

impl Rasterizer {
    pub fn from_shp(
        path: &str,
        height: usize,
        width: usize,
        sample_size: usize,
        id_key: &str,
    ) -> std::io::Result<Rasterizer> {
        let mut reader = match Reader::from_path(path) {
            Ok(x) => x,
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            }
        };
        let shape_count = reader.shape_count().unwrap();
        let progress = ProgressBar::new(shape_count as u64);
        let mut raster = Vec::<Vec<f64>>::new();
        let mut id = Vec::<String>::new();

        for result in reader.iter_shapes_and_records() {
            let (shape, record) = result.unwrap();
            id.push(match record.get(id_key).expect("id key not found") {
                FieldValue::Character(s) => s.clone().expect("id value missing"),
                _ => panic!("id must be string type"),
            });
            if let Shape::Polygon(polygon) = shape {
                let mut rasterizer = PolygonRasterizer::new(polygon, (width, height));
                raster.push(rasterizer.get_raster(sample_size));
                progress.inc(1);
            } else {
                panic!("Only polygons are supported");
            }
        }

        progress.finish_with_message("All done!");
        let mut keys = Vec::<String>::new();
        for col in 0..width {
            for row in 0..height {
                keys.push(format!("p{col}x{row}, "));
            }
        }

        Ok(Self { raster, keys, id })
    }

    pub fn write_csv(&self, path: &str) -> std::io::Result<()> {
        write_csv(path, &self.keys, &self.raster, &self.id)?;
        Ok(())
    }
}

struct PolygonRasterizer {
    polygon: Polygon,
    pixel_hits: Vec<Vec<usize>>,
    shape: (usize, usize),
    bb: Point,
    rng: ThreadRng,
    pixel_bounds: Vec<(Point, Point)>,
}

impl PolygonRasterizer {
    pub fn new(polygon: Polygon, shape: (usize, usize)) -> PolygonRasterizer {
        let mut result = Self {
            polygon,
            shape,
            pixel_hits: Vec::default(),
            pixel_bounds: Vec::default(),
            bb: Point::default(),
            rng: rand::thread_rng(),
        };
        result.reduce_polygon();
        result.init_pixel_bounds();
        result.init_pixel_hits();
        result
    }

    pub fn get_raster(&mut self, sample_size: usize) -> Vec<f64> {
        let result = (0..(self.shape.0 * self.shape.1))
            .map(|i| self.process_pixel(i, sample_size))
            .collect();
        result
    }

    fn process_pixel(&mut self, pixel: usize, sample_size: usize) -> f64 {
        let mut points_in: usize = 0;
        for _ in 0..sample_size {
            let point = self.random_point(pixel);
            if self.point_is_in(point, pixel) {
                points_in += 1;
            }
        }
        points_in as f64 / sample_size as f64
    }

    fn point_is_in(&self, point: Point, pixel: usize) -> bool {
        let mut num_hits = 0;
        let points = self
            .polygon
            .rings()
            .iter()
            .flat_map(|r| r.points().iter())
            .collect::<Vec<&Point>>();
        for p in self.pixel_hits[pixel].iter() {
            // draw vertical ray from p to +inf and test intersection
            let mut p0 = points[*p];
            let mut p1 = points[(p + 1) % points.len()];
            if p0.x > p1.x {
                std::mem::swap(&mut p0, &mut p1);
            }
            let y = p0.y.min(p1.y);
            if (p0.x..p1.x).contains(&point.x) && y > point.y {
                num_hits += 1;
            }
        }
        if num_hits % 2 == 1 {
            true
        } else {
            false
        }
    }

    fn random_point(&mut self, i: usize) -> Point {
        let bottom_left = self.pixel_bounds[i].0;
        let top_right = self.pixel_bounds[i].1;
        Point::new(
            self.rng.gen_range(bottom_left.x..top_right.x),
            self.rng.gen_range(bottom_left.y..top_right.y),
        )
    }

    fn init_pixel_bounds(&mut self) {
        self.pixel_bounds = vec![(Point::default(), Point::default()); self.shape.0 * self.shape.1];
        for col in 0..self.shape.0 {
            for row in 0..self.shape.1 {
                self.pixel_bounds[row * self.shape.0 + col] = (
                    Point::new(
                        col as f64 * self.bb.x / self.shape.0 as f64,
                        row as f64 * self.bb.y / self.shape.1 as f64,
                    ),
                    Point::new(
                        (col as f64 + 1.0) * self.bb.x / self.shape.0 as f64,
                        (row as f64 + 1.0) * self.bb.y / self.shape.1 as f64,
                    ),
                )
            }
        }
    }

    fn init_pixel_hits(&mut self) {
        self.pixel_hits = vec![Vec::new(); self.shape.0 * self.shape.1];
        let points = self
            .polygon
            .rings()
            .iter()
            .flat_map(|r| r.points().iter())
            .collect::<Vec<&Point>>();
        for col in 0..self.shape.0 {
            for row in 0..self.shape.1 {
                let index = row * self.shape.0 + col;
                let start_vertex = 0..points.len();
                // offset by 1 and cycle
                let end_vertex = (0..points.len()).cycle().skip(1).take(points.len());
                self.pixel_hits[index] = std::iter::zip(start_vertex, end_vertex)
                    .filter(|(start, end)| {
                        let side_x_range = points[*start].x..points[*end].x;
                        let pixel_hits_x_range = side_x_range
                            .contains(&self.pixel_bounds[index].0.x)
                            || side_x_range.contains(&self.pixel_bounds[index].1.x);
                        let pixel_hits_y_range =
                            points[*start].y.max(points[*end].y) > self.pixel_bounds[index].0.y;
                        pixel_hits_x_range && pixel_hits_y_range
                    })
                    .map(|(x, _)| x)
                    .collect()
            }
        }
    }

    fn reduce_polygon(&mut self) {
        let (bottom_left, top_right) = self.bounding_box();
        self.bb = Point::new(top_right.x - bottom_left.x, top_right.y - bottom_left.y);
        self.polygon = Polygon::with_rings(
            self.polygon
                .rings()
                .iter()
                .map(|x| Self::reduce_ring(x, bottom_left))
                .collect(),
        );
    }

    fn reduce_ring(ring: &PolygonRing<Point>, bottom_left: Point) -> PolygonRing<Point> {
        let constructor = match ring {
            PolygonRing::Outer(_) => PolygonRing::Outer,
            PolygonRing::Inner(_) => PolygonRing::Inner,
        };
        constructor(
            ring.points()
                .iter()
                .map(|Point { x, y }| Point::new(x - bottom_left.x, y - bottom_left.y))
                .collect(),
        )
    }

    fn bounding_box(&self) -> (Point, Point) {
        let mut points = self.polygon.rings().iter().flat_map(|x| x.points().iter());

        let Point { x, y } = points.next().unwrap();
        let mut bottom_left = Point::new(*x, *y);
        let mut top_right = Point::new(*x, *y);

        for Point { x, y } in points {
            bottom_left.x = bottom_left.x.min(*x);
            bottom_left.y = bottom_left.y.min(*y);
            top_right.x = top_right.x.max(*x);
            top_right.y = top_right.y.max(*y);
        }

        (bottom_left, top_right)
    }
}
