use std::env;

use indicatif::ProgressBar;
use rand::{rngs::ThreadRng, Rng};
use shapefile::{Point, Polygon, PolygonRing, Reader, Shape};

// for debuggigg
mod image;

struct Args {
    input: String,
    output: String,
    height: usize,
    width: usize,
    sample_size: usize,
}

impl Args {
    fn new(args: &Vec<String>) -> Args {
        Self {
            input: args
                .get(1)
                .expect("missing argument input file")
                .to_string(),
            output: args
                .get(2)
                .expect("missing argument output file")
                .to_string(),
            height: args
                .get(3)
                .expect("missing argument height")
                .parse::<usize>()
                .expect("invalid value for argument height"),
            width: args
                .get(4)
                .expect("missing argument width")
                .parse::<usize>()
                .expect("invalid value for argument width"),
            sample_size: args
                .get(5)
                .expect("missing argument sample size")
                .parse::<usize>()
                .expect("invalid value for argument sample size"),
        }
    }
}

fn main() {
    let Args {
        input,
        output,
        height,
        width,
        sample_size,
    } = Args::new(&env::args().collect());
    let mut reader =
        Reader::from_path(&input).expect(&format!("Could not find input file: {}", input));
    let mut rng = rand::thread_rng();
    let shape_count = reader.shape_count().unwrap();
    let progress = ProgressBar::new(shape_count as u64);

    for result in reader.iter_shapes_and_records() {
        let (shape, record) = result.unwrap();
        if let Shape::Polygon(polygon) = shape {
            process_polygon(&polygon, height, width, sample_size, &mut rng);
            progress.inc(1);
        } else {
            panic!("Only polygons are supported");
        }
    }

    progress.finish_with_message("All done!");
}

fn process_polygon(
    polygon: &Polygon,
    height: usize,
    width: usize,
    sample_size: usize,
    rng: &mut ThreadRng,
) -> Vec<f64> {
    let (polygon, bb) = reduce_polygon(polygon);
    let result = (0..(height * width))
        .map(|i| process_pixel(i, bb, height, width, sample_size, &polygon, rng))
        .collect();
    result
}

fn process_pixel(
    i: usize,
    bb: Point,
    height: usize,
    width: usize,
    sample_size: usize,
    polygon: &Polygon,
    rng: &mut ThreadRng,
) -> f64 {
    let (bottom_left, top_right) = get_pixel_bounds(i, bb, height, width);
    let mut points_in: usize = 0;
    for _ in 0..sample_size {
        let point = random_point(bottom_left, top_right, rng);
        if point_is_in(polygon, point) {
            points_in += 1;
        }
    }
    points_in as f64 / sample_size as f64
}

fn point_is_in(polygon: &Polygon, p: Point) -> bool {
    for ring in polygon.rings() {
        let mut hits_ring = 0;
        for i in 1..ring.points().len() {
            // draw vertical ray from p to +inf and test intersection
            let mut x0 = ring.points()[i - 1].x;
            let mut x1 = ring.points()[i].x;
            if x0 > x1 {
                std::mem::swap(&mut x0, &mut x1);
            }
            let y = ring.points()[i].y.min(ring.points()[i - 1].y);
            if (x0..x1).contains(&p.x) && y > p.y {
                hits_ring += 1;
            }
        }
        if hits_ring % 2 == 1 {
            return true;
        }
    }
    false
}

fn random_point(bottom_left: Point, top_right: Point, rng: &mut ThreadRng) -> Point {
    Point::new(
        rng.gen_range(bottom_left.x..top_right.x),
        rng.gen_range(bottom_left.y..top_right.y),
    )
}

fn get_pixel_bounds(i: usize, bb: Point, height: usize, width: usize) -> (Point, Point) {
    let x = (i % width) as f64;
    let y = (i / width) as f64;
    (
        Point::new(x * bb.x / width as f64, y * bb.y / height as f64),
        Point::new(
            (x + 1.0) * bb.x / width as f64,
            (y + 1.0) * bb.y / height as f64,
        ),
    )
}

fn reduce_polygon(polygon: &Polygon) -> (Polygon, Point) {
    let (bottom_left, top_right) = bounding_box(polygon);
    let reduced_top_right = Point::new(top_right.x - bottom_left.x, top_right.y - bottom_left.y);
    let reduced_polygon = Polygon::with_rings(
        polygon
            .rings()
            .iter()
            .map(|x| reduce_ring(x, bottom_left))
            .collect(),
    );
    (reduced_polygon, reduced_top_right)
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

fn bounding_box(polygon: &Polygon) -> (Point, Point) {
    let mut points = polygon.rings().iter().flat_map(|x| x.points().iter());

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
