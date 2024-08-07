use itertools::Itertools;
use log::info;
use pyo3::PyResult;
use std::collections::HashSet;
use std::fmt::Debug;
use std::num::Wrapping;
use std::ops::Range;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;

const MAX: f64 = std::u64::MAX as f64;

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

pub fn init_precinct_with_threads(
    adj: Vec<Vec<usize>>,
    population: Vec<usize>,
    num_districts: usize,
    pop_thresh: f32,
    num_threads: u8,
) -> Vec<usize> {
    let stop_token = Arc::new(AtomicBool::new(false));
    let adj_arc = Arc::new(adj);
    let pop_arc = Arc::new(population);
    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let adj_cloned = adj_arc.clone();
        let pop_cloned = pop_arc.clone();
        let stop_cloned = stop_token.clone();
        handles.push(thread::spawn(move || {
            init_precinct(
                adj_cloned,
                pop_cloned,
                num_districts,
                pop_thresh,
                stop_cloned,
            )
        }));
    }
    while !stop_token.load(Ordering::Relaxed) {}
    for handle in handles {
        if let Some(x) = handle.join().unwrap() {
            return x;
        }
    }
    panic!("stop token was modified, but no thread returned Some");
}

fn init_precinct(
    adj: Arc<Vec<Vec<usize>>>,
    population: Arc<Vec<usize>>,
    num_districts: usize,
    pop_thresh: f32,
    stop_token: Arc<AtomicBool>,
) -> Option<Vec<usize>> {
    let total_pop = population.iter().sum::<usize>();
    let num_nodes = population.len();
    let max_pop = total_pop as f32 / (num_districts as f32 - 1.0 + pop_thresh);
    let min_pop = pop_thresh * max_pop;
    let mut rand_state = UniformDist::new([0xfda52833df686ae6, 0x7919f78c90a9362c]);
    let mut result = Vec::new();
    let mut sol_feasible = false;
    let mut vis;

    while !sol_feasible {
        vis = vec![false; num_nodes];
        result = vec![num_districts; num_nodes];
        let mut nodes_added = num_districts;
        let mut frontier: Vec<(usize, HashSet<usize>)> = Vec::new();
        let mut district_pops = vec![0; num_districts];
        let mut starter_nodes = Vec::new();
        for d in 0..num_districts {
            let mut starter_node = num_nodes;
            while starter_node == num_nodes || vis[starter_node] {
                starter_node = rand_state.next() as usize % num_nodes;
            }
            starter_nodes.push(starter_node);
            vis[starter_node] = true;
            result[starter_node] = d;
            district_pops[d] += population[starter_node];
            frontier.push((d, HashSet::from_iter(adj[starter_node].clone())))
        }
        for node in starter_nodes {
            for district in frontier.iter_mut() {
                district.1.remove(&node);
            }
        }

        sol_feasible = true;
        while sol_feasible && nodes_added < num_nodes {
            let not_dead_end = frontier
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .fold(0, |acc, _| acc + 1);

            let index = frontier
                .iter()
                .enumerate()
                .filter(|(_, (_, v))| !v.is_empty())
                .nth(rand_state.next() as usize % not_dead_end)
                .unwrap()
                .0;

            let added_node = *frontier[index]
                .1
                .iter()
                .nth(rand_state.next() as usize % frontier[index].1.len())
                .unwrap();

            district_pops[index] += population[added_node];
            vis[added_node] = true;
            nodes_added += 1;
            if district_pops[index] > max_pop as usize
                || (frontier[index].1.len() == 0 && district_pops[index] < min_pop as usize)
            {
                sol_feasible = false;
            }
            result[added_node] = index;
            for next in adj[added_node].iter().filter(|n| !vis[**n]) {
                frontier[index].1.insert(*next);
            }
            for district in 0..num_districts {
                frontier[district].1.remove(&added_node);
            }

            if stop_token.load(Ordering::Relaxed) {
                return None;
            }
        }
    }
    stop_token.store(true, Ordering::Relaxed);
    Some(result)
}

pub struct Annealer<F: Sync + Fn(&[Vec<bool>]) -> f64, G: Sync + Fn(f64) -> f64> {
    objective: F,
    temperature: G,
    adj: Vec<Vec<usize>>,
    cur_state: (f64, Vec<usize>),
    best: (f64, Vec<usize>),
    num_districts: usize,
    population: Vec<usize>,
    border_nodes: Vec<usize>,
    pop_thresh: f32,
    num_nodes: usize,
}

impl<F: Sync + Fn(&[Vec<bool>]) -> f64, G: Sync + Fn(f64) -> f64> Debug for Annealer<F, G> {
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

impl<F: Sync + Fn(&[Vec<bool>]) -> f64, G: Sync + Fn(f64) -> f64> Annealer<F, G> {
    pub fn from_starting_state(
        starting_assignment: Vec<usize>,
        adj: Vec<Vec<usize>>,
        objective: F,
        temperature: G,
        num_districts: usize,
        population: Vec<usize>,
        pop_thresh: f32,
    ) -> Annealer<F, G> {
        info!("{:?}", starting_assignment);
        let mut annealer = Annealer {
            objective,
            temperature,
            adj,
            num_districts,
            num_nodes: starting_assignment.len(),
            cur_state: (0.0, starting_assignment),
            best: (0.0, Vec::new()),
            population,
            border_nodes: Vec::new(),
            pop_thresh,
        };
        annealer.cur_state.0 = (annealer.objective)(&annealer.state_as_input(None, None));
        annealer.best = annealer.cur_state.clone();
        annealer.border_nodes = annealer.get_border_nodes();
        if !annealer.feasible(None, None) {
            panic!("infeasible starting");
        } else {
            info!("starting feasible");
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

    // assignment_indexer is used for score_change, to make it &self instead of &mut self
    fn state_as_input(
        &self,
        changed_node: Option<usize>,
        district: Option<usize>,
    ) -> Vec<Vec<bool>> {
        assert_eq!(changed_node.is_some(), district.is_some());
        // nothing changes if None
        let changed_node = changed_node.unwrap_or(0);
        let district = district.unwrap_or(self.cur_state.1[0]);

        let mut result = vec![vec![false; self.num_nodes]; self.num_districts];
        for node in 0..self.num_nodes {
            result[if node == changed_node {
                district
            } else {
                self.cur_state.1[node]
            }][node] = true;
        }
        result
    }

    pub fn anneal(&mut self, num_steps: usize, num_threads: u8) -> PyResult<(f64, Vec<usize>)> {
        info!("{:?}", self.border_nodes);
        let mut rand_state = UniformDist::new([0xfda52833df686ae6, 0x7919f78c90a9362c]);
        for step in 0..num_steps {
            let temp = (self.temperature)(step as f64 / num_steps as f64);
            let immut_self = &*self;
            // let handles = (0..num_threads).map(|i| {
            //     thread::spawn(move || {
            //         let start = (i as usize) * self.num_nodes / num_threads as usize;
            //         let end = self
            //             .num_nodes
            //             .min((i as usize + 1) * self.num_nodes / num_threads as usize)
            //             - 1;
            //         immut_self.get_scores(start..end, temp)
            //     })
            // });
            let scores: Vec<(usize, usize, f64)> = (0..self.num_nodes)
                .flat_map(|node| {
                    self.adj[node]
                        .iter()
                        .filter_map(move |neighbor| {
                            if immut_self.cur_state.1[*neighbor] == immut_self.cur_state.1[node]
                                || !immut_self
                                    .feasible(Some(node), Some(immut_self.cur_state.1[*neighbor]))
                            {
                                None
                            } else {
                                Some(immut_self.cur_state.1[*neighbor])
                            }
                        })
                        .unique()
                        .map(move |district_num| {
                            (
                                node,
                                district_num,
                                (immut_self.score_change(node, district_num) / temp).exp(),
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
                    self.cur_state.0 = (self.objective)(&self.state_as_input(None, None));
                    if self.cur_state.0 > self.best.0 {
                        self.best = self.cur_state.clone();
                    }
                    break;
                }
            }
        }
        Ok(self.cur_state.clone())
    }

    fn get_scores(&self, range: Range<usize>, temp: f64) -> Vec<(usize, usize, f64)> {
        range
            .flat_map(|node| {
                self.adj[node]
                    .iter()
                    .filter_map(move |neighbor| {
                        if self.cur_state.1[*neighbor] == self.cur_state.1[node]
                            || !self.feasible(Some(node), Some(self.cur_state.1[*neighbor]))
                        {
                            None
                        } else {
                            Some(self.cur_state.1[*neighbor])
                        }
                    })
                    .unique()
                    .map(move |district_num| {
                        (
                            node,
                            district_num,
                            (self.score_change(node, district_num) / temp).exp(),
                        )
                    })
            })
            .collect()
    }

    fn score_change(&self, node: usize, district_num: usize) -> f64 {
        info!("starting score");
        let start = std::time::SystemTime::now();
        let score = (self.objective)(&self.state_as_input(Some(node), Some(district_num)));
        info!("ending score: {}", start.elapsed().unwrap().as_millis());
        score
    }

    fn update_borders(&mut self, node: usize, district: usize) {
        for neighbor in self.adj[node].iter() {
            if district == self.cur_state.1[*neighbor]
                && self.cur_state.1[node] != self.cur_state.1[*neighbor]
            {
                self.border_nodes[*neighbor] -= 1;
            } else if self.cur_state.1[node] == self.cur_state.1[*neighbor] {
                match self.border_nodes.get_mut(*neighbor) {
                    Some(v) => *v += 1,
                    None => _ = self.border_nodes.insert(*neighbor, 1),
                }
            }
        }
    }

    fn feasible(&self, changed_node: Option<usize>, district: Option<usize>) -> bool {
        // nothing changes
        assert_eq!(changed_node.is_some(), district.is_some());
        // info!("starting feas");
        let changed_node = changed_node.unwrap_or(0);
        let district = district.unwrap_or(self.cur_state.1[0]);

        let populations = self.population.iter().enumerate().fold(
            vec![0_usize; self.num_districts],
            |mut acc, (i, elem)| {
                acc[self.cur_state.1[i]] += elem;
                acc
            },
        );
        if *populations.iter().max().unwrap() as f32 * self.pop_thresh
            > *populations.iter().min().unwrap() as f32
        {
            // info!("ending feas");
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
            .collect::<Vec<_>>();
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
                // info!("ending feas");
                return false;
            }
        }
        // info!("ending feas");
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
                .collect::<Vec<&usize>>();
            vis[node] = self.num_districts;
            for child in children {
                stack.push(*child);
            }
        }
    }
}
