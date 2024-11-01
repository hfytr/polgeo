use crate::Annealer;
use crate::RANDOM;
use itertools::Itertools;
use std::fmt::Debug;

pub trait StepStrategy: Default + Send + Sync + Debug {
    type Step: Send + Clone + Debug;
    fn init(annealer: &Annealer<Self>) -> Self;
    fn next_states(&self, annealer: &Annealer<Self>) -> Vec<Self::Step>;
    /// this is called many times
    fn update(
        &mut self,
        cur_state: &mut Vec<usize>,
        adj: &Vec<Vec<usize>>,
        populations: &Vec<usize>,
        num_districts: usize,
        step: &Self::Step,
    );
    fn index(&self, cur_assignment: &Vec<usize>, step: &Self::Step, i: usize) -> usize;
}

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct SingleNodeStep(usize, usize);

#[derive(Default, Debug)]
pub struct SingleNodeStrategy;

impl SingleNodeStrategy {
    fn population_feasible(
        &self,
        annealer: &Annealer<Self>,
        step: &<Self as StepStrategy>::Step,
    ) -> bool {
        let populations = annealer.population.iter().enumerate().fold(
            vec![0_usize; annealer.num_districts],
            |mut acc, (i, elem)| {
                acc[self.index(&annealer.cur_state.1, step, i)] += elem;
                acc
            },
        );
        return (*populations.iter().max().unwrap() as f32
            / *populations.iter().min().unwrap() as f32)
            < 1.0 + annealer.pop_const;
    }

    fn feasible_helper(&self, annealer: &Annealer<Self>, node: usize, vis: &mut Vec<usize>) {
        // getting some bs if i don't write the stack
        let mut stack = vec![node];
        while let Some(node) = stack.pop() {
            let children = annealer.adj[node]
                .iter()
                .filter(|n| vis[**n] != annealer.num_districts && vis[**n] == vis[node])
                .collect_vec();
            vis[node] = annealer.num_districts;
            for child in children {
                stack.push(*child);
            }
        }
    }

    fn feasible(&self, annealer: &Annealer<Self>, step: &<Self as StepStrategy>::Step) -> bool {
        if annealer.pop_constraint && !self.population_feasible(annealer, step) {
            return false;
        }

        if !annealer.pop_constraint {}
        // whether we have flood filled a given district
        let mut flood_filled = vec![false; annealer.num_districts];
        // node district if unvis, else num districts
        let mut vis = (0..annealer.num_nodes)
            .map(|i| self.index(&annealer.cur_state.1, step, i))
            .collect_vec();
        // only check nodes in districts adjacent to changed node
        let relevant_districts = annealer.adj[step.0]
            .iter()
            .map(|x| annealer.cur_state.1[*x])
            .unique()
            .collect_vec();
        for node in (0..annealer.num_nodes)
            .filter(|x| relevant_districts.contains(&annealer.cur_state.1[*x]))
        {
            if vis[node] == annealer.num_districts {
                continue;
            }
            let node_district = self.index(&annealer.cur_state.1, step, node);
            if !flood_filled[node_district] {
                self.feasible_helper(annealer, node, &mut vis);
                flood_filled[node_district] = true;
            }
            // if have flood filled node district, but haven't visited, then node is disconnected
            else if vis[node] != annealer.num_districts {
                return false;
            }
        }
        true
    }
}

impl StepStrategy for SingleNodeStrategy {
    type Step = SingleNodeStep;

    fn init(_annealer: &Annealer<Self>) -> Self {
        SingleNodeStrategy {}
    }

    fn next_states(&self, annealer: &Annealer<Self>) -> Vec<Self::Step> {
        (0..annealer.num_nodes)
            .flat_map(|node| {
                annealer.adj[node]
                    .iter()
                    .filter_map(move |neighbor| {
                        if annealer.cur_state.1[*neighbor] == annealer.cur_state.1[node] {
                            None
                        } else {
                            Some(SingleNodeStep(node, annealer.cur_state.1[*neighbor]))
                        }
                    })
                    .unique()
            })
            .filter(|step| self.feasible(annealer, step))
            .collect_vec()
    }

    fn update(
        &mut self,
        cur_state: &mut Vec<usize>,
        _: &Vec<Vec<usize>>,
        _: &Vec<usize>,
        _: usize,
        step: &Self::Step,
    ) {
        let SingleNodeStep(node, district) = *step;
        cur_state[node] = district;
    }

    fn index(
        &self,
        cur_assignment: &Vec<usize>,
        SingleNodeStep(node, district): &Self::Step,
        i: usize,
    ) -> usize {
        if i == *node {
            *district
        } else {
            cur_assignment[i]
        }
    }
}

#[derive(Default, Debug)]
struct SegTreeNode {
    children: Vec<usize>,
    population: usize,
    index: (usize, usize),
    parent: Option<usize>,
}

#[derive(Default, Debug)]
struct SegTree {
    root: usize,
    nodes: Vec<SegTreeNode>,
    mapping: (Vec<usize>, Vec<Option<usize>>),
}

impl SegTree {
    fn wilson(adj: &Vec<Vec<usize>>, populations: &Vec<usize>, mask: Vec<bool>) -> SegTree {
        let mut idx = -1;
        let mapping = (
            mask.iter()
                .enumerate()
                .filter(|(_, x)| **x)
                .map(|(i, _)| i)
                .collect_vec(),
            mask.iter()
                .map(|x| {
                    x.then(|| {
                        idx += 1;
                        idx as usize
                    })
                })
                .collect_vec(),
        );
        let num_nodes = mapping.0.len();

        let mut random = RANDOM.lock().unwrap();
        let root = random.next() as usize % num_nodes;
        let mut nodes = (0..num_nodes).map(|_| SegTreeNode::default()).collect_vec();
        let mut vis = vec![false; num_nodes];
        vis[root] = true;
        let mut nodes_left = num_nodes - 1;

        while nodes_left > 0 {
            let mut walk_vis = vec![None; num_nodes];
            let walk_start = random.next() as usize % nodes_left;
            let mut unvis = 0;
            let mut frontier = vis.len();
            for (i, vis_i) in vis.iter().enumerate() {
                if !vis_i {
                    unvis += 1;
                }
                if unvis > walk_start {
                    frontier = i;
                    break;
                }
            }

            loop {
                if vis[frontier] {
                    let mut cur = frontier;
                    while let Some(loop_prev) = walk_vis[cur] {
                        nodes[cur].children.push(loop_prev);
                        nodes[loop_prev].parent = Some(cur);
                        vis[loop_prev] = true;
                        nodes_left -= 1;
                        cur = loop_prev;
                    }
                    break;
                } else if walk_vis[frontier].is_some() {
                    let mut cur = frontier;
                    while let Some(loop_prev) = walk_vis[cur]
                        && loop_prev != frontier
                    {
                        walk_vis[loop_prev] = None;
                        cur = loop_prev;
                    }
                }
                let next = *random.choice(&adj[frontier].iter().filter(|x| mask[**x]).collect());
                walk_vis[next] = Some(frontier);
                frontier = next;
            }
        }

        Self::node_populations(&mut nodes, populations, &mapping.0, root);
        Self::node_index(&mut nodes, root);

        SegTree {
            root,
            nodes,
            mapping,
        }
    }

    fn node_populations(
        nodes: &mut Vec<SegTreeNode>,
        population: &Vec<usize>,
        mapping: &Vec<usize>,
        node: usize,
    ) -> usize {
        nodes[node].population = nodes[node]
            .children
            .clone()
            .iter()
            .map(|&child| Self::node_populations(nodes, population, mapping, child))
            .sum::<usize>()
            + population[mapping[node]];
        nodes[node].population
    }

    fn node_index(nodes: &mut Vec<SegTreeNode>, root: usize) {
        fn helper(nodes: &mut Vec<SegTreeNode>, node: usize, index: &mut usize) {
            nodes[node].index.0 = *index;
            for child_ind in 0..nodes[node].children.len() {
                helper(nodes, nodes[node].children[child_ind], index);
            }
            nodes[node].index.1 = *index;
            *index += 1;
        }
        let mut index = 0;
        helper(nodes, root, &mut index);
    }
}

#[derive(Default, Debug)]
pub struct RecomStrategy {
    seg_tree: SegTree,
    districts: (usize, usize),
    pop_const: f32,
}

type RecomStep = usize;

impl RecomStrategy {
    fn random_districts(
        cur_state: &Vec<usize>,
        adj: &Vec<Vec<usize>>,
        num_districts: usize,
    ) -> (usize, usize) {
        let mut random = RANDOM.lock().unwrap();
        let district1 = random.next() as usize % num_districts;
        let district2 = random.choice(
            &cur_state
                .iter()
                .enumerate()
                .filter(|(_, x)| **x == district1)
                .flat_map(|(i, _)| adj[i].iter().map(|&j| cur_state[j]))
                .unique()
                .filter(|x| *x != district1)
                .collect_vec(),
        );
        (district1, district2)
    }
}

impl StepStrategy for RecomStrategy {
    type Step = RecomStep;

    fn init(annealer: &Annealer<Self>) -> Self {
        let districts =
            Self::random_districts(&annealer.cur_state.1, &annealer.adj, annealer.num_districts);
        RecomStrategy {
            seg_tree: SegTree::wilson(
                &annealer.adj,
                &annealer.population,
                annealer
                    .cur_state
                    .1
                    .iter()
                    .map(|x| *x == districts.0 || *x == districts.1)
                    .collect(),
            ),
            districts,
            pop_const: annealer.pop_const,
        }
    }

    fn index(&self, cur_assignment: &Vec<usize>, step: &Self::Step, i: usize) -> usize {
        if let Some(im) = self.seg_tree.mapping.1[i] {
            if (self.seg_tree.nodes[*step].index.0..=self.seg_tree.nodes[*step].index.1)
                .contains(&self.seg_tree.nodes[im].index.1)
            {
                self.districts.0
            } else {
                self.districts.1
            }
        } else {
            cur_assignment[i]
        }
    }

    fn update(
        &mut self,
        cur_state: &mut Vec<usize>,
        adj: &Vec<Vec<usize>>,
        populations: &Vec<usize>,
        num_districts: usize,
        _: &Self::Step,
    ) {
        self.districts = Self::random_districts(cur_state, adj, num_districts);
        self.seg_tree = SegTree::wilson(
            adj,
            populations,
            cur_state
                .iter()
                .map(|x| *x == self.districts.0 || *x == self.districts.1)
                .collect(),
        );
    }

    fn next_states(&self, annealer: &Annealer<Self>) -> Vec<Self::Step> {
        if !annealer.pop_constraint {
            // step doesn't need to be mapped
            return (0..(self.seg_tree.root - 1)).collect_vec();
        }

        let mut result = Vec::new();
        // for 2 districts all districts are involved in the recombination
        let mut global_min_pop = None;
        let mut global_max_pop = None;

        if annealer.num_districts > 2 {
            let populations = annealer
                .population
                .iter()
                .enumerate()
                .filter(|(i, _)| self.seg_tree.mapping.1[*i].is_none())
                .fold(
                    vec![0_usize; annealer.num_districts],
                    |mut acc, (i, elem)| {
                        acc[annealer.cur_state.1[i]] += elem;
                        acc
                    },
                );
            for pop in populations {
                if pop != 0 {
                    global_min_pop = Some(global_min_pop.unwrap_or(usize::MAX).min(pop));
                }
                global_max_pop = Some(global_max_pop.unwrap_or(0).max(pop))
            }
        }

        for (i, node) in self
            .seg_tree
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.seg_tree.root)
        {
            let min_pop = global_min_pop
                .unwrap_or(node.population)
                .min(node.population)
                .min(self.seg_tree.nodes[self.seg_tree.root].population - node.population);
            let max_pop = global_max_pop
                .unwrap_or(node.population)
                .max(node.population)
                .max(self.seg_tree.nodes[self.seg_tree.root].population - node.population);

            dbg!(max_pop);
            dbg!(min_pop);
            dbg!(node.population);

            if (max_pop as f32) * self.pop_const <= min_pop as f32 {
                result.push(i);
            }
        }

        if result.len() == 0 {
            dbg!(&self.seg_tree);
        }
        result
    }
}
