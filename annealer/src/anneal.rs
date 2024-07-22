use itertools::Itertools;
use pyo3::PyResult;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::num::Wrapping;

const MAX: f64 = std::u64::MAX as f64;
const MAX_POP_DIFF: f64 = 0.05;

#[inline]
fn rotl(x: u64, k: i32) -> u64 {
    (x << k) | (x >> (64 - k))
}

#[derive(Debug)]
pub struct UniformDist([u64; 2]);
impl UniformDist {
    fn next(&mut self) -> u64 {
        let s0 = Wrapping(self.0[0]);
        let mut s1 = Wrapping(self.0[1]);
        let result = s0 + s1;
        s1 ^= s0;
        self.0[0] = (Wrapping(rotl(s0.0, 55)) ^ s1 ^ (s1 << 14)).0;
        self.0[1] = rotl(s1.0, 36);
        result.0
    }

    fn new(seed: [u64; 2]) -> UniformDist {
        Self([seed[0], seed[1]])
    }
}

pub struct Annealer<F: Fn(&[Vec<bool>]) -> f64, G: Fn(f64) -> f64> {
    objective: F,
    temperature: G,
    precinct_in: Vec<usize>,
    adj: Vec<Vec<usize>>,
    cur_state: (f64, Vec<usize>),
    best: (f64, Vec<usize>),
    num_districts: usize,
    population: Vec<usize>,
    border_nodes: BTreeMap<usize, usize>,
}

impl<F: Fn(&[Vec<bool>]) -> f64, G: Fn(f64) -> f64> Debug for Annealer<F, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("precinct_in: {:?}", self.precinct_in))?;
        f.write_fmt(format_args!("adj: {:?}", self.adj))?;
        f.write_fmt(format_args!("cur_state: {:?}", self.cur_state))?;
        f.write_fmt(format_args!("best_state: {:?}", self.best))?;
        f.write_fmt(format_args!("num_districts: {}", self.num_districts))?;
        f.write_fmt(format_args!("population: {:?}", self.population))?;
        f.write_fmt(format_args!("border_nodes: {:?}", self.border_nodes))?;
        Ok(())
    }
}

impl<F: Fn(&[Vec<bool>]) -> f64, G: Fn(f64) -> f64> Annealer<F, G> {
    pub fn new(
        precinct_in: Vec<usize>,
        adj: Vec<Vec<usize>>,
        objective: F,
        temperature: G,
        num_districts: usize,
        population: Vec<usize>,
    ) -> Annealer<F, G> {
        let state = vec![0_usize; precinct_in.len()];
        let mut annealer = Annealer {
            objective,
            temperature,
            adj,
            precinct_in,
            num_districts,
            cur_state: (0.0, state),
            best: (0.0, Vec::new()),
            population,
            border_nodes: BTreeMap::new(),
        };
        annealer.cur_state.0 =
            (annealer.objective)(&annealer.state_as_input::<fn(usize) -> usize>(None));
        annealer.best = annealer.cur_state.clone();
        annealer.border_nodes = annealer.get_border_nodes();
        dbg!(&annealer);
        annealer
    }

    fn get_border_nodes(&self) -> BTreeMap<usize, usize> {
        BTreeMap::from_iter((0..self.precinct_in.iter().len()).map(|x| {
            (
                x,
                self.adj[x].iter().fold(0, |acc, elem| {
                    acc + (self.cur_state.1[*elem] != self.cur_state.1[x]) as usize
                }),
            )
        }))
    }

    // assignment_indexer is used for score_change, to make it &self instead of &mut self
    fn state_as_input<H: Fn(usize) -> usize>(
        &self,
        assignment_indexer: Option<H>,
    ) -> Vec<Vec<bool>> {
        let default_indexer = |x| self.cur_state.1[x];
        let assignment_indexer: &dyn Fn(usize) -> usize = match &assignment_indexer {
            Some(x) => x,
            None => &default_indexer,
        };
        let mut result = vec![vec![false; self.precinct_in.len()]; self.num_districts];
        for precinct in self.precinct_in.iter() {
            result[assignment_indexer(*precinct)][*precinct] = true;
        }
        result
    }

    pub fn anneal(&mut self, num_steps: usize) -> PyResult<(f64, Vec<usize>)> {
        let mut rand_state = UniformDist::new([0xfda52833df686ae6, 0x7919f78c90a9362c]);
        for step in 0..num_steps {
            let temp = (self.temperature)(step as f64 / num_steps as f64);
            let scores: Vec<(usize, usize, f64)> = self
                .border_nodes
                .iter()
                .flat_map(|(node, _)| {
                    self.adj[*node]
                        .iter()
                        .filter_map(|neighbor| {
                            if self.cur_state.1[*neighbor] == self.cur_state.1[*node]
                                || !self.feasible(*node, self.cur_state.1[*neighbor])
                            {
                                None
                            } else {
                                Some(self.cur_state.1[*neighbor])
                            }
                        })
                        .unique()
                        .map(|district_num| {
                            (
                                *node,
                                district_num,
                                (self.score_change(*node, district_num) / temp).exp(),
                            )
                        })
                })
                .collect();
            let sum: f64 = scores.iter().fold(0.0, |acc, (_, _, elem)| acc + elem);
            let next_probabilities = scores.iter().map(|(_, _, x)| x / sum);
            let rand = rand_state.next() as f64 / MAX;
            let mut accumulated_probability = 0.0;
            for (i, probability) in next_probabilities.enumerate() {
                accumulated_probability += probability;
                if accumulated_probability > rand {
                    self.update_borders(scores[i].0, scores[i].1);
                    self.cur_state.0 =
                        (self.objective)(&self.state_as_input::<fn(usize) -> usize>(None));
                    if self.cur_state.0 > self.best.0 {
                        self.best = self.cur_state.clone();
                    }
                    break;
                }
            }
        }
        Ok(self.cur_state.clone())
    }

    fn score_change(&self, node: usize, district_num: usize) -> f64 {
        let score = (self.objective)(&self.state_as_input(Some(|i| {
            if i == node {
                district_num
            } else {
                self.cur_state.1[i]
            }
        })));
        score
    }

    fn update_borders(&mut self, node: usize, district: usize) {
        for neighbor in self.adj[node].iter() {
            if district == self.cur_state.1[*neighbor]
                && self.cur_state.1[node] != self.cur_state.1[*neighbor]
            {
                *self.border_nodes.get_mut(neighbor).unwrap() -= 1;
            } else if self.cur_state.1[node] == self.cur_state.1[*neighbor] {
                match self.border_nodes.get_mut(neighbor) {
                    Some(v) => *v += 1,
                    None => _ = self.border_nodes.insert(*neighbor, 1),
                }
            }
        }
    }

    fn feasible(&self, changed_node: usize, district: usize) -> bool {
        let populations = self.population.iter().enumerate().fold(
            vec![0_usize; self.num_districts],
            |mut acc, (i, elem)| {
                acc[self.precinct_in[i]] += elem;
                acc
            },
        );
        let mean = populations.iter().sum::<usize>() as f64 / populations.len() as f64;
        for p in populations {
            if !((1.0 - MAX_POP_DIFF) * mean..(1.0 + MAX_POP_DIFF) * mean).contains(&(p as f64)) {
                return false;
            }
        }
        let mut flood_filled = vec![false; self.num_districts];
        let mut vis = vec![self.num_districts; self.adj.len()];
        for node in 0..self.precinct_in.len() {
            if vis[node] != self.num_districts {
                continue;
            }
            let node_district = if node == changed_node {
                self.cur_state.1[node]
            } else {
                district
            };
            if !flood_filled[node_district] {
                self.feasible_helper(node, &mut vis, node, district);
                flood_filled[node_district] = true;
            } else if vis[node] == self.num_districts {
                return false;
            }
        }
        true
    }

    fn feasible_helper(
        &self,
        node: usize,
        vis: &mut Vec<usize>,
        changed_node: usize,
        district: usize,
    ) {
        vis[node] = if node == changed_node {
            self.cur_state.1[node]
        } else {
            district
        };
        for next in self.adj[node]
            .iter()
            .filter(|n| vis[**n] == vis[node])
            .collect::<Vec<&usize>>()
        {
            self.feasible_helper(*next, vis, changed_node, district);
        }
    }
}
