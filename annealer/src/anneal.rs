use crate::anneal_step::StepStrategy;
use crate::rand::MAX;
use crate::RANDOM;
use itertools::Itertools;
use std::fmt::Debug;
use std::thread;

pub struct Annealer<S: StepStrategy> {
    // need to be accessed by StepStrategy
    pub adj: Vec<Vec<usize>>,
    pub num_nodes: usize,
    pub cur_state: (f64, Vec<usize>),
    pub best: (f64, Vec<usize>),
    pub objective: Box<dyn Send + Sync + Fn(&[usize]) -> f64>,
    pub temperature: Box<dyn Send + Sync + Fn(f64) -> f64>,
    pub population: Vec<usize>,
    pub num_districts: usize,
    pub pop_const: f32,
    pub pop_constraint: bool,

    hist: Vec<f64>,
    stepper: S,
}

fn _print_grid<T: Debug>(a: &Vec<T>, width: usize) {
    println!("");
    for (i, x) in a.iter().enumerate() {
        print!("{:?} ", x);
        if i % width == width - 1 {
            print!("\n");
        }
    }
}

impl<S: StepStrategy> Annealer<S> {
    pub fn from_starting_state(
        starting_assignment: Vec<usize>,
        adj: Vec<Vec<usize>>,
        num_districts: usize,
        population: Vec<usize>,
        pop_thresh: f32,
        pop_constraint: bool,
        objective: Box<dyn Send + Sync + Fn(&[usize]) -> f64>,
        temperature: Box<dyn Send + Sync + Fn(f64) -> f64>,
    ) -> Annealer<S> {
        let mut annealer = Annealer {
            adj,
            num_districts,
            num_nodes: starting_assignment.len(),
            cur_state: (0.0, starting_assignment),
            best: (0.0, Vec::new()),
            population,
            stepper: S::default(),
            hist: Vec::new(),
            pop_const: pop_thresh,
            objective,
            temperature,
            pop_constraint,
        };
        annealer.best = annealer.cur_state.clone();
        annealer.stepper = S::init(&annealer);
        annealer
    }

    pub fn set_state(&mut self, new_state: Vec<usize>) {
        self.cur_state.1 = new_state;
        self.cur_state.0 = (self.objective)(&self.cur_state.1);
        self.stepper = S::init(&self);
    }

    pub fn anneal(&mut self, num_steps: usize, num_threads: u8) -> (Vec<usize>, Vec<f64>) {
        for step in 0..num_steps {
            let temp = (self.temperature)(step as f64 / num_steps as f64);
            let feasible_moves = self.stepper.next_states(&self);

            let num_feasible = feasible_moves.len();
            let chunks = feasible_moves
                .into_iter()
                .chunks((num_feasible + num_threads as usize - 1) / num_threads as usize)
                .into_iter()
                .map(|x| x.collect_vec())
                .collect_vec();

            let mut scores = Vec::new();
            thread::scope(|s| {
                let handles = chunks
                    .into_iter()
                    .map(|chunk| {
                        s.spawn(|| {
                            Self::get_scores(
                                &self.cur_state.1,
                                self.num_nodes,
                                &self.objective,
                                &self.stepper,
                                chunk,
                            )
                        })
                    })
                    .collect_vec();

                scores = handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect_vec();
            });

            let adjusted = scores
                .iter()
                .flatten()
                .map(|(x, _)| ((self.cur_state.0 - x) / temp).exp())
                .collect_vec();

            let sum: f64 = adjusted.iter().fold(0.0, |acc, elem| acc + elem);

            let probabilities = scores
                .into_iter()
                .flatten()
                .zip(adjusted.into_iter())
                .map(|((score, step), adjusted)| (score, step, adjusted / sum));

            let rand = RANDOM.lock().unwrap().next() as f64 / MAX;
            let mut accumulated_probability = 0.0;
            for (score, step, probability) in probabilities.clone() {
                accumulated_probability += probability;
                if accumulated_probability > rand {
                    self.hist.push(score);
                    self.stepper.update(
                        &mut self.cur_state.1,
                        &self.adj,
                        &self.population,
                        self.num_districts,
                        &step,
                    );
                    break;
                }
            }
        }

        (self.cur_state.1.clone(), self.hist.clone())
    }

    fn get_scores(
        cur_assignment: &Vec<usize>,
        num_nodes: usize,
        objective: &Box<dyn Send + Sync + Fn(&[usize]) -> f64>,
        stepper: &S,
        chunk: Vec<S::Step>,
    ) -> Vec<(f64, S::Step)> {
        chunk
            .into_iter()
            .map(|step| {
                (
                    objective(
                        &(0..num_nodes)
                            .map(|i| stepper.index(&cur_assignment, &step, i))
                            .collect_vec(),
                    ),
                    step,
                )
            })
            .collect()
    }
}
