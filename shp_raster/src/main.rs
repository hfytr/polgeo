use raster::Rasterizer;
use std::env;

// for debuggigg
mod csv;
mod image;
mod raster;

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
    let rasterizer = Rasterizer::from_shp(&input, height, width, sample_size, "district")
        .expect(&format!("Could not find input file: {}", input));
    rasterizer
        .write_csv(&output)
        .expect("failed to write to csv");
}
