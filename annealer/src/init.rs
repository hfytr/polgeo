use crate::RANDOM;
use std::collections::HashSet;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
pub fn init_precinct_with_threads(
    adj: Vec<Vec<usize>>,
    population: Vec<usize>,
    num_districts: usize,
    pop_thresh: f64,
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
    let mut result = None;
    for handle in handles {
        if let Some(x) = handle.join().unwrap() {
            result = Some(x);
        }
    }
    result.expect("stop token was modified, but no thread returned Some")
}

fn init_precinct(
    adj: Arc<Vec<Vec<usize>>>,
    population: Arc<Vec<usize>>,
    num_districts: usize,
    pop_thresh: f64,
    stop_token: Arc<AtomicBool>,
) -> Option<Vec<usize>> {
    let total_pop = population.iter().sum::<usize>();
    let num_nodes = population.len();
    let max_pop = total_pop as f64 / (1.0 + (num_districts as f64 - 1.0) / (1.0 + pop_thresh));
    let min_pop = max_pop / (1.0 + pop_thresh);
    let mut result = Vec::new();
    let mut sol_feasible = false;
    let mut vis;
    let mut random = RANDOM.lock().unwrap();

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
                starter_node = random.next() as usize % num_nodes;
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
        let mut d = Vec::new();
        while sol_feasible && nodes_added < num_nodes {
            let not_dead_end = frontier
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .fold(0, |acc, _| acc + 1);

            d.push(frontier.clone());

            let index = frontier
                .iter()
                .enumerate()
                .filter(|(_, (_, v))| !v.is_empty())
                .nth(random.next() as usize % not_dead_end)
                .unwrap()
                .0;

            let added_node = *frontier[index]
                .1
                .iter()
                .nth(random.next() as usize % frontier[index].1.len())
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
                if frontier[district].1.len() == 0
                    && !(min_pop..max_pop).contains(&(district_pops[district] as f64))
                {
                    sol_feasible = false;
                }
            }

            if stop_token.load(Ordering::Relaxed) {
                return None;
            }
        }
    }
    stop_token.store(true, Ordering::Relaxed);
    Some(result)
}
