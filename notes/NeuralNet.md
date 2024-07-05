---
id: NeuralNet
aliases: []
tags: []
---

goal: given a net N (deep ff ReLU), and polyhedron P(x) representing inputs to N: estimate a range denoted (l, p) for each output of the net N, that takes all possible outputs and is within a given epsilon
encodes N to MILP; adds P(x) to said MILP. max / min the MILP to get range. find local minima using gradient approach. once stuck, use MILP solver to try and find better solution (or prove absence).
useful in robotics context (machines have physical limits)

let z = output of each hidden layer, y overall output, t a binary variable used in representing ReLU. for each hidden layer i 1..k
    z_{i+1} >= 0
    z_{i+1} >= W_i * z_i + b_i
    z_{i+1} <= M(1-t_{i+1})
    z_{i+1} <= W_i * z_i + b_i + M * t_{i+1}
where M > maximum possible output of any node. M is estimated using inf. norm of W, and bounding box of polyhedra
for other activations, approximate using multiple linear pieces (3 for tanh etc)

pseudo code:
while not feas:
    (x, u) = local_search()           // via gradient descent / ascent
    u += delta
    (x', u', feas) = solve_milp(l, u) // solves until u is beaten
    if feas:
        (x, u) = (x', u')

for a given input x, and piecewise linear activation, the locally active region around x is the set of all x' which are in the same linear segment as x, for all nodes. notice: gradient stays constant over locally active regions.
then the local maximum is simply max p_i^T x s.t. x in L(x_i) union P, where p_i is the gradient of F at x_i
