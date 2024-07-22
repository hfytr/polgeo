use pyo3::{
    exceptions::PyException,
    pyfunction, pymodule,
    types::{PyAnyMethods, PyModule},
    wrap_pyfunction, Bound, PyAny, PyErr, PyResult,
};
mod anneal;

///@param objective callable objective function
///@param temperature callable temperature function
///@param precinct_in: a vector of all precincts, and which pixel they are in; pixels are row *
///width + col
///@param adj a 2d vector adjacency list for the precincts
///@param num_districts the number of districts to be planned
///@param population the population of each precinct (indices should match precinct_in, adj)
#[pyfunction]
fn optimize_func(
    objective_raw: Bound<'_, PyAny>,
    temperature_raw: Bound<'_, PyAny>,
    precinct_in: Vec<usize>,
    adj: Vec<Vec<usize>>,
    num_districts: usize,
    population: Vec<usize>,
    num_steps: usize,
) -> PyResult<(f64, Vec<usize>)> {
    if adj.len() != precinct_in.len() {
        return PyResult::Err(PyErr::new::<PyException, _>(PyException::new_err(
            "adj.len() != precinct_in.len",
        )));
    }
    if !objective_raw.is_callable() {
        return PyResult::Err(PyErr::new::<PyException, _>(PyException::new_err(
            "objective argument must be callable",
        )));
    }
    let objective = |v: &[Vec<bool>]| {
        objective_raw
            .call1((v.to_vec(),))
            .unwrap()
            .extract()
            .expect("objective returned wrong type, must retur f64")
    };
    if !temperature_raw.is_callable() {
        return PyResult::Err(PyErr::new::<PyException, _>(PyException::new_err(
            "temperature argument must be callable",
        )));
    }
    let temperature = |x: f64| {
        objective_raw
            .call1((x,))
            .unwrap()
            .extract()
            .expect("temperature returned wrong type, must retur f64")
    };
    let mut annealer = anneal::Annealer::new(
        precinct_in,
        adj,
        objective,
        temperature,
        num_districts,
        population,
    );
    annealer.anneal(num_steps)
}

#[pymodule]
fn annealer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_func, m)?)?;
    Ok(())
}
