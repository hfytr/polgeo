use crate::rand::{UniformDist, MAX};
use itertools::Itertools;
use std::fmt::Debug;
use std::iter::zip;
use std::thread;

pub struct Annealer<F: Send + Sync + Fn(&[usize]) -> f64 + Clone, G: Send + Sync + Fn(f64) -> f64> {
    adj: Vec<Vec<usize>>,
    cur_state: (f64, Vec<usize>),
    best: (f64, Vec<usize>),
    num_districts: usize,
    population: Vec<usize>,
    border_nodes: Vec<usize>,
    hist: Vec<(usize, usize, f64)>,
    pop_thresh: f32,
    num_nodes: usize,
    objective: F,
    temperature: G,
}

impl<F: Send + Sync + Fn(&[usize]) -> f64 + Clone, G: Send + Sync + Fn(f64) -> f64> Debug
    for Annealer<F, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("adj: {:?}", self.adj))?;
        f.write_fmt(format_args!("cur_state: {:?}", self.cur_state))?;
        f.write_fmt(format_args!("best_state: {:?}", self.best))?;
        f.write_fmt(format_args!("num_districts: {}", self.num_districts))?;
        f.write_fmt(format_args!("population: {:?}", self.population))?;
        f.write_fmt(format_args!("border_nodes: {:?}", self.border_nodes))?;
        Ok(())
    }
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

impl<F: Send + Sync + Fn(&[usize]) -> f64 + Clone, G: Send + Sync + Fn(f64) -> f64> Annealer<F, G> {
    pub fn from_starting_state(
        starting_assignment: Vec<usize>,
        adj: Vec<Vec<usize>>,
        num_districts: usize,
        population: Vec<usize>,
        pop_thresh: f32,
        objective: F,
        temperature: G,
    ) -> Annealer<F, G> {
        let mut annealer = Annealer {
            adj,
            num_districts,
            num_nodes: starting_assignment.len(),
            cur_state: (0.0, starting_assignment),
            best: (0.0, Vec::new()),
            population,
            border_nodes: Vec::new(),
            hist: Vec::new(),
            pop_thresh,
            objective,
            temperature,
        };
        annealer.best = annealer.cur_state.clone();
        annealer.border_nodes = annealer.get_border_nodes();
        if !annealer.feasible(None) {
            panic!("infeasible starting");
        } else {
            println!("starting feasible");
        }
        annealer
    }

    fn get_border_nodes(&self) -> Vec<usize> {
        (0..self.num_nodes)
            .map(|x| {
                self.adj[x].iter().fold(0, |acc, elem| {
                    acc + (self.cur_state.1[*elem] != self.cur_state.1[x]) as usize
                })
            })
            .collect()
    }

    pub fn anneal(
        mut self,
        num_steps: usize,
        num_threads: u8,
    ) -> (f64, Vec<usize>, Vec<(usize, usize, f64)>) {
        let mut rand_state = UniformDist::new([0xfda52833df686ae6, 0x7919f78c90a9362c]);
        for step in 0..num_steps {
            let temp = (self.temperature)(step as f64 / num_steps as f64);
            let immut_self = &self;
            let feasible_moves = (0..self.num_nodes)
                .flat_map(|node| {
                    self.adj[node]
                        .iter()
                        .filter_map(move |neighbor| {
                            if immut_self.cur_state.1[*neighbor] == immut_self.cur_state.1[node]
                                || !immut_self
                                    .feasible(Some((node, immut_self.cur_state.1[*neighbor])))
                            {
                                None
                            } else {
                                Some((node, immut_self.cur_state.1[*neighbor]))
                            }
                        })
                        .unique()
                })
                .collect_vec();

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
                    .map(|chunk| s.spawn(|| immut_self.get_scores(chunk)))
                    .collect_vec();

                scores = handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect_vec();
            });

            let adjusted = scores
                .iter()
                .flatten()
                .map(|(_, _, x)| (x / temp).exp())
                .collect_vec();

            let sum: f64 = adjusted.iter().fold(0.0, |acc, elem| acc + elem);

            let probabilities = zip(scores.into_iter().flatten(), adjusted.into_iter())
                .map(|((node, district, score), adjusted)| (node, district, score, adjusted / sum));

            let rand = rand_state.next() as f64 / MAX;
            let mut accumulated_probability = 0.0;
            for (node, district, score, probability) in probabilities {
                accumulated_probability += probability;
                if accumulated_probability > rand {
                    println!("{}", score);
                    self.hist.push((node, district, score));
                    self.update_state(node, district);
                    break;
                }
            }
        }
        (self.cur_state.0, self.cur_state.1, self.hist)
    }

    fn get_scores(&self, chunk: Vec<(usize, usize)>) -> Vec<(usize, usize, f64)> {
        chunk
            .into_iter()
            .map(|(node, district_num)| {
                (
                    node,
                    district_num,
                    (self.objective)(
                        &self
                            .cur_state
                            .1
                            .iter()
                            .enumerate()
                            .map(|(i, x)| if i == node { district_num } else { *x })
                            .collect_vec(),
                    ),
                )
            })
            .collect()
    }

    fn update_state(&mut self, node: usize, district: usize) {
        for neighbor in self.adj[node].iter() {
            if self.cur_state.1[*neighbor] == district {
                if self.cur_state.1[node] != self.cur_state.1[*neighbor] {
                    self.border_nodes[*neighbor] -= 1;
                }
                self.border_nodes[node] -= 1;
            } else if self.cur_state.1[node] == self.cur_state.1[*neighbor] {
                self.border_nodes[*neighbor] += 1;
                self.border_nodes[node] += 1;
            }
        }
        self.cur_state.1[node] = district;
        self.cur_state.0 = (self.objective)(self.cur_state.1.as_slice());
        if self.cur_state.0 > self.best.0 {
            self.best = self.cur_state.clone();
        }
    }

    fn feasible(&self, changes: Option<(usize, usize)>) -> bool {
        let (changed_node, district) = changes.unwrap_or((0, self.cur_state.1[0]));

        let populations = self.population.iter().enumerate().fold(
            vec![0_usize; self.num_districts],
            |mut acc, (i, elem)| {
                let node_district = if i == changed_node {
                    district
                } else {
                    self.cur_state.1[i]
                };
                acc[node_district] += elem;
                acc
            },
        );
        if *populations.iter().max().unwrap() as f32 / *populations.iter().min().unwrap() as f32
            > 1.0 + self.pop_thresh
        {
            return false;
        }
        // whether we have flood filled a given district
        let mut flood_filled = vec![false; self.num_districts];
        // node district if unvis, else num districts
        let mut vis = self.cur_state.1.clone();
        vis[changed_node] = district;
        // only check nodes in districts adjacent to changed node
        let relevant_districts = self.adj[changed_node]
            .iter()
            .map(|x| self.cur_state.1[*x])
            .unique()
            .collect_vec();
        for node in
            (0..self.num_nodes).filter(|x| relevant_districts.contains(&self.cur_state.1[*x]))
        {
            if vis[node] == self.num_districts {
                continue;
            }
            let node_district = if node == changed_node {
                district
            } else {
                self.cur_state.1[node]
            };
            if !flood_filled[node_district] {
                self.feasible_helper(node, &mut vis, node, district);
                flood_filled[node_district] = true;
            }
            // if have flood filled node district, but haven't visited, then node is disconnected
            else if vis[node] != self.num_districts {
                return false;
            }
        }
        true
    }

    fn feasible_helper(
        &self,
        node: usize,
        vis: &mut Vec<usize>,
        _changed_node: usize,
        _district: usize,
    ) {
        // getting some bs if i don't write the stack
        let mut stack = vec![node];
        while let Some(node) = stack.pop() {
            let children = self.adj[node]
                .iter()
                .filter(|n| vis[**n] != self.num_districts && vis[**n] == vis[node])
                .collect_vec();
            vis[node] = self.num_districts;
            for child in children {
                stack.push(*child);
            }
        }
    }
}
