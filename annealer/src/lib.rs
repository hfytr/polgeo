pub mod adjacency;
mod anneal;
mod init;
mod rand;
mod vec3;
pub use anneal::Annealer;
pub use init::init_precinct_with_threads;

use adjacency::CollectLineString;
use itertools::Itertools;

use geo::{
    algorithm::{Area, ConvexHull},
    BooleanOps, LineString, MultiPolygon, Polygon,
};
use pyo3::{
    exceptions::PyException,
    pyclass, pyfunction, pymethods, pymodule,
    types::{PyAnyMethods, PyModule},
    wrap_pyfunction, Bound, FromPyObject, PyAny, PyErr, PyResult,
};

#[pyclass]
struct AnnealerService {
    annealer: Annealer,
}

#[derive(Clone)]
struct WrappedPolygon {
    exterior: LineString,
    interior: Vec<LineString>,
}

impl Into<Polygon> for WrappedPolygon {
    fn into(self) -> Polygon {
        Polygon::new(self.exterior, self.interior)
    }
}

impl<'py> FromPyObject<'py> for WrappedPolygon {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (exterior_vec, interior_vec) =
            ob.extract::<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)>()?;
        let interior = interior_vec
            .into_iter()
            .map(|x| x.into_iter().collect_linestring(true))
            .collect_vec();
        let exterior = exterior_vec.into_iter().collect_linestring(true);
        Ok(Self { exterior, interior })
    }
}

#[pymethods]
impl AnnealerService {
    #[new]
    fn new(
        precinct_in: Vec<usize>,
        adj: Vec<Vec<usize>>,
        geometries: Vec<WrappedPolygon>,
        num_districts: usize,
        population: Vec<usize>,
        pop_constant: f32,
        pop_constraint: bool, // otherwise pop in objective
        t0: f64,
    ) -> PyResult<Self> {
        if adj.len() != precinct_in.len() {
            return PyResult::Err(PyErr::new::<PyException, _>(PyException::new_err(
                "adj.len() != precinct_in.len",
            )));
        }
        let temperature = move |x: f64| (x * t0 as f64);

        let objective = move |assignment: &[usize]| {
            let mut districts = vec![
                MultiPolygon::<f64>::new(vec![Polygon::<f64>::new(
                    LineString::<f64>::from(Vec::<(f64, f64)>::new()),
                    vec![]
                )]);
                num_districts
            ];

            for (geometry, &district) in geometries.iter().zip(assignment.iter()) {
                districts[district] =
                    districts[district].union(&MultiPolygon::new(vec![(geometry.clone()).into()]));
            }

            districts
                .iter()
                .map(|district| district.convex_hull().unsigned_area() / district.unsigned_area())
                .sum::<f64>()
        };

        let objective: Box<dyn Send + Sync + Fn(&[usize]) -> f64> = if pop_constraint {
            Box::new(objective)
        } else {
            let cloned_population = population.clone();
            Box::new(move |assignment: &[usize]| {
                let num_districts = num_districts;
                let mut district_pops = vec![0.0; num_districts];
                for (node, district) in assignment.iter().enumerate() {
                    district_pops[*district] += cloned_population[node] as f32;
                }
                let p_avg = district_pops.iter().sum::<f32>() / num_districts as f32;
                objective(assignment)
                    + (pop_constant * district_pops.iter().map(|p| p - p_avg).sum::<f32>()) as f64
            })
        };

        Ok(Self {
            annealer: anneal::Annealer::from_starting_state(
                precinct_in,
                adj,
                num_districts,
                population,
                pop_constant,
                pop_constraint,
                objective,
                Box::new(temperature),
            ),
        })
    }

    fn anneal(
        &mut self,
        starting_state: Vec<usize>,
        num_steps: usize,
        num_threads: u8,
    ) -> PyResult<(Vec<usize>, Vec<(usize, usize, f64)>)> {
        self.annealer.set_state(starting_state);
        Ok(self.annealer.anneal(num_steps, num_threads))
    }
}

#[pyfunction]
fn init_precinct(
    adj: Vec<Vec<usize>>,
    population: Vec<usize>,
    num_districts: usize,
    pop_thresh: f32,
    num_threads: u8,
) -> Vec<usize> {
    init::init_precinct_with_threads(adj, population, num_districts, pop_thresh, num_threads)
}

#[pymodule]
fn annealer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnealerService>()?;
    m.add_function(wrap_pyfunction!(init_precinct, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{adjacency::*, *};
    use geo::{
        algorithm::{bool_ops::BooleanOps, Area, ConvexHull},
        geometry::{LineString, MultiPolygon, Polygon},
    };
    use itertools::Itertools;
    use std::fmt::Debug;

    const ANNEAL_POP_THRESH: f32 = 0.50;
    const T0: f64 = 0.1;
    const NUM_THREADS: u8 = 8;

    #[test]
    fn example() {
        const SIDE_LEN: usize = 10;
        test_grid((SIDE_LEN, SIDE_LEN), vec![1; SIDE_LEN * SIDE_LEN], 2);
    }

    fn test_grid(dim: (usize, usize), pop: Vec<usize>, num_districts: usize) {
        let cells = (0..dim.0 * dim.1)
            .map(|i| {
                let row = (i / dim.0) as f64;
                let col = (i % dim.0) as f64;
                Polygon::new(
                    vec![
                        (col, row),
                        (col + 1.0, row),
                        (col + 1.0, row + 1.0),
                        (col, row + 1.0),
                    ]
                    .into_iter()
                    .collect_linestring(true),
                    Vec::new(),
                )
            })
            .collect_vec();

        let adj = (0..dim.0 * dim.1)
            .map(|i| {
                let row = (i / dim.0) as i32;
                let col = (i % dim.0) as i32;
                let mut result = Vec::new();
                for offset in vec![(0, 1), (0, -1), (1, 0), (-1, 0)] {
                    if (0..dim.0).contains(&((row + offset.0) as usize))
                        && (0..dim.1).contains(&((col + offset.1) as usize))
                    {
                        result.push((row + offset.0) as usize * dim.0 + (col + offset.1) as usize);
                    }
                }
                result
            })
            .collect_vec();

        let objective = move |assignment: &[usize]| {
            let mut districts = vec![
                MultiPolygon::<f64>::new(vec![Polygon::<f64>::new(
                    LineString::<f64>::from(Vec::<(f64, f64)>::new()),
                    vec![]
                )]);
                num_districts
            ];

            for (cell, &district) in cells.iter().zip(assignment.iter()) {
                districts[district] =
                    districts[district].union(&MultiPolygon::new(vec![cell.clone()]));
            }

            -districts
                .iter()
                .map(|district| district.convex_hull().unsigned_area() / district.unsigned_area())
                .sum::<f64>()
        };

        let mut annealer = Annealer::from_starting_state(
            init_precinct_with_threads(
                adj.clone(),
                pop.clone(),
                num_districts,
                ANNEAL_POP_THRESH,
                NUM_THREADS,
            ),
            adj,
            num_districts,
            pop,
            ANNEAL_POP_THRESH,
            true,
            Box::new(objective),
            Box::new(|x| (x * T0).max(0.1)),
        );

        let (assignment, hist) = annealer.anneal(10, NUM_THREADS);
        dbg!(hist);
        print_grid(&assignment, dim.0);
    }

    fn print_grid<T: Debug>(a: &Vec<T>, width: usize) {
        println!("");
        for (i, x) in a.iter().enumerate() {
            print!("{:?} ", x);
            if i % width == width - 1 {
                print!("\n");
            }
        }
    }
}
