#![feature(let_chains)]

pub mod adjacency;
mod anneal;
mod anneal_step;
mod init;
mod rand;
mod vec3;
pub use anneal::Annealer;
pub use init::init_precinct_with_threads;

use adjacency::CollectLineString;
use itertools::Itertools;

use anneal_step::{RecomStrategy, SingleNodeStrategy};
use geo::{
    algorithm::{Area, ConvexHull},
    BooleanOps, LineString, MultiPolygon, Polygon,
};
use pyo3::{
    exceptions::PyException,
    pyfunction, pymodule,
    types::{PyAnyMethods, PyModule},
    wrap_pyfunction, Bound, FromPyObject, PyAny, PyErr, PyResult,
};
use rand::UniformDist;
use std::fmt::Debug;
use std::sync::Mutex;

static RANDOM: Mutex<UniformDist> =
    Mutex::new(UniformDist::new([0xe125793adf7617c2, 0x241d1623a7a207c7]));

fn print_indexer<F: Fn(usize) -> usize>(indexer: F, width: usize, height: usize) {
    print_grid(&(0..width * height).map(indexer).collect_vec(), width)
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

#[pyfunction]
fn anneal_districts(
    starting_state: Vec<usize>,
    adj: Vec<Vec<usize>>,
    geometries: Vec<WrappedPolygon>,
    num_districts: usize,
    population: Vec<usize>,
    num_steps: usize,
    num_threads: u8,
    single_step: bool,
    pop_constr: bool,
    pop_constant: f64,
    t0: f64,
) -> PyResult<(Vec<usize>, Vec<(f64, Vec<usize>)>)> {
    if adj.len() != starting_state.len() {
        return PyResult::Err(PyErr::new::<PyException, _>(PyException::new_err(
            "adj.len() != precinct_in.len",
        )));
    }
    let temperature = move |x: f64| (x * t0 as f64);

    let base_objective = move |assignment: &[usize]| {
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

    let cloned_population = population.clone();
    let boxed_objective: Box<dyn Send + Sync + Fn(&[usize]) -> f64> = if pop_constr {
        Box::new(base_objective)
    } else {
        Box::new(move |assignment: &[usize]| {
            let num_districts = num_districts;
            let mut district_pops = vec![0.0; num_districts];
            for (node, district) in assignment.iter().enumerate() {
                district_pops[*district] += cloned_population[node] as f64;
            }
            let p_avg = district_pops.iter().sum::<f64>() / num_districts as f64;
            base_objective(assignment)
                + (pop_constant * district_pops.iter().map(|p| p - p_avg).sum::<f64>())
        })
    };

    if single_step {
        let mut annealer = anneal::Annealer::<SingleNodeStrategy>::from_starting_state(
            starting_state,
            adj,
            num_districts,
            population,
            pop_constant,
            pop_constr,
            boxed_objective,
            Box::new(temperature),
        );
        Ok(annealer.anneal(num_steps, num_threads))
    } else {
        let mut annealer = anneal::Annealer::<RecomStrategy>::from_starting_state(
            starting_state,
            adj,
            num_districts,
            population,
            pop_constant,
            pop_constr,
            boxed_objective,
            Box::new(temperature),
        );
        Ok(annealer.anneal(num_steps, num_threads))
    }
}

#[pyfunction]
fn init_precinct(
    adj: Vec<Vec<usize>>,
    population: Vec<usize>,
    num_districts: usize,
    pop_thresh: f64,
    num_threads: u8,
) -> Vec<usize> {
    init::init_precinct_with_threads(adj, population, num_districts, pop_thresh, num_threads)
}

#[pymodule]
fn annealer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_precinct, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_districts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::anneal_step::StepStrategy;
    use super::{adjacency::*, *};
    use geo::{
        algorithm::{bool_ops::BooleanOps, Area, ConvexHull},
        geometry::{LineString, MultiPolygon, Polygon},
    };
    use itertools::Itertools;

    const ANNEAL_POP_THRESH: f64 = 0.85;
    const T0: f64 = 0.1;
    const NUM_THREADS: u8 = 8;

    #[test]
    fn single_node_mini() {
        const SIDE_LEN: usize = 10;
        test_grid::<SingleNodeStrategy>(
            (SIDE_LEN, SIDE_LEN),
            2,
            true,
            vec![1; SIDE_LEN * SIDE_LEN],
            None,
        );
    }

    #[test]
    fn recom_mini() {
        const SIDE_LEN: usize = 4;
        test_grid::<RecomStrategy>(
            (SIDE_LEN, SIDE_LEN),
            2,
            false,
            vec![1; SIDE_LEN * SIDE_LEN],
            None,
        );
    }

    #[test]
    fn recom_large() {
        test_grid::<RecomStrategy>(
            (5, 5),
            3,
            false,
            vec![
                5, 125, 5, 298, 224, 5, 93, 64, 68, 228, 8, 223, 147, 124, 130, 70, 193, 163, 118,
                81, 96, 13, 115, 5, 160,
            ],
            Some(vec![
                0, 2, 2, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
            ]),
        )
    }

    fn test_grid<S: StepStrategy>(
        dim: (usize, usize),
        num_districts: usize,
        pop_constraint: bool,
        pop: Vec<usize>,
        starting_state: Option<Vec<usize>>,
    ) {
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

            for (geometry, &district) in cells.iter().zip(assignment.iter()) {
                districts[district] =
                    districts[district].union(&MultiPolygon::new(vec![(geometry.clone()).into()]));
            }

            let result = districts
                .iter()
                .map(|district| district.convex_hull().unsigned_area() / district.unsigned_area())
                .sum::<f64>();

            result
        };

        let pop_constant = if pop_constraint {
            ANNEAL_POP_THRESH
        } else {
            num_districts as f64 / pop.iter().sum::<usize>() as f64
        };
        let boxed_objective: Box<dyn Send + Sync + Fn(&[usize]) -> f64> = if pop_constraint {
            Box::new(objective)
        } else {
            let cloned_population = pop.clone();
            Box::new(move |assignment: &[usize]| {
                let num_districts = num_districts;
                let mut district_pops = vec![0.0; num_districts];
                for (node, district) in assignment.iter().enumerate() {
                    district_pops[*district] += cloned_population[node] as f64;
                }
                let p_avg = district_pops.iter().sum::<f64>() / num_districts as f64;
                objective(assignment)
                    + (pop_constant * district_pops.iter().map(|p| (p - p_avg).abs()).sum::<f64>())
                        as f64
            })
        };

        let starting_state = starting_state.unwrap_or_else(|| {
            init_precinct_with_threads(
                adj.clone(),
                pop.clone(),
                num_districts,
                ANNEAL_POP_THRESH,
                NUM_THREADS,
            )
        });

        let mut annealer = Annealer::<S>::from_starting_state(
            starting_state,
            adj,
            num_districts,
            pop,
            pop_constant,
            pop_constraint,
            boxed_objective,
            Box::new(|x| (x * T0).max(0.1)),
        );

        let (assignment, hist) = annealer.anneal(100, 1);
        dbg!(hist);
        print_grid(&assignment, dim.0);
    }
}
