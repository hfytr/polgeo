use std::{
    fs::File,
    io::BufWriter,
    ops::{Index, IndexMut},
};

use shapefile::{Point, Polygon};

#[derive(Debug)]
pub struct Image {
    pixels: Vec<Vec<f64>>,
    bb: Point,
    shape: (usize, usize),
}

impl Image {
    pub fn new(polygon: &Polygon, bb: Point, shape: (usize, usize)) -> Image {
        let mut img = Image {
            pixels: vec![vec![0.0, 0.0, 0.0]; shape.0 * shape.1],
            bb,
            shape,
        };
        for point in polygon.rings().iter().flat_map(|r| r.points().iter()) {
            let x_coord = ((point.x / img.bb.x * shape.0 as f64) as usize).min(shape.0 - 1);
            let y_coord = ((point.y / img.bb.x * shape.1 as f64) as usize).min(shape.1 - 1);
            img[(x_coord, y_coord)] = vec![255.0, 255.0, 255.0];
        }
        img
    }

    pub fn write_to_png(&self, path: &str) -> Result<(), std::io::Error> {
        let file = File::options()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        let ref mut w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, self.shape.0 as u32, self.shape.1 as u32);

        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));
        let source_chromaticities = png::SourceChromaticities::new(
            (0.31270, 0.32900),
            (0.64000, 0.33000),
            (0.30000, 0.60000),
            (0.15000, 0.06000),
        );
        encoder.set_source_chromaticities(source_chromaticities);
        let mut writer = encoder.write_header()?;

        writer.write_image_data(&self.as_flattened_u8())?;

        Ok(())
    }

    pub fn set_white(&mut self, p: &Point) {
        let x_coord = ((p.x / self.bb.x * self.shape.0 as f64) as usize).min(self.shape.0 - 1);
        let y_coord = ((p.y / self.bb.x * self.shape.1 as f64) as usize).min(self.shape.1 - 1);
        self[(x_coord, y_coord)] = vec![255.0, 255.0, 255.0];
    }

    fn as_flattened_u8(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .flat_map(|v| v.iter())
            .map(|x| *x as u8)
            .collect()
    }
}

impl Index<(usize, usize)> for Image {
    type Output = Vec<f64>;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.pixels[index.0 + index.1 * self.shape.0]
    }
}

impl IndexMut<(usize, usize)> for Image {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.pixels[index.0 + index.1 * self.shape.0]
    }
}
