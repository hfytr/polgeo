import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os

def parse_var(var):
    if var == 'z':
        return None, None
    if 'Y' in var:
        return None, None
    temp = var.split('[')[1].split(']')[0].split(',')
    
    col_idx = int(temp[1])
    row_idx = int(temp[0])
    return row_idx, col_idx



if __name__ == '__main__':
    model =   model = gp.Model('simple_lp')

    A = np.load('./data/A.npy')
    A = A.astype(bool)
    c = np.load('./data.npy')
    n = A.shape[0]
    m = 8
    A_t = np.triu(A)


    # Define your graph using an adjacency matrix
    nodes = np.arange(n)

    # Create a new model
    model = gp.Model("graph_partitioning")

    # Add variables
    edges = model.addVars(nodes, nodes, m, vtype=GRB.BINARY, name="edges")  # Edge in subgraph k
    nodes = model.addVars(nodes, m, vtype=GRB.BINARY, name="nodes")  # Node in subgraph k
    flow = model.addVars(nodes, nodes, m, vtype=GRB.CONTINUOUS, name="flow")  # Flow on edge in subgraph k
    z = model.addVar(name='z', vtype=GRB.CONTINUOUS)
    # Objective function: minimize the number of inter-subgraph edges
    model.setObjective(z, GRB.MINIMIZE)

    # Constraints
    # Each node is in exactly one subgraph
    model.addConstrs((nodes.sum(i, '*') == 1 for i in nodes), name="node_assignment")

    # Connectivity via flow
    source_nodes = np.random.choice(nodes, size=(m,))  # Arbitrarily chosen source nodes for each subgraph


    for k in range(m):
        print(m)
        model.addConstr(-z <= c @ nodes[:, k])
        model.addConstr(z >= c @ nodes[:, k])

    for k in range(m):
        print(k)
        source = source_nodes[k]
        # Flow conservation for non-source nodes
        model.addConstrs((gp.quicksum(flow[j, i, k] for j in nodes if A[i, j]) ==
                        gp.quicksum(flow[i, j, k] for j in nodes if A[i, j])
                        for i in nodes if i != source), name=f"flow_conservation_{k}")

        # Source node outflow equals to number of nodes in subgraph - 1
        model.addConstr(gp.quicksum(flow[source, j, k] for j in nodes if A[i, j]) ==
                        gp.quicksum(nodes[i, k] for i in nodes) - 1, name=f"source_outflow_{k}")

    # Edge inclusion and flow linkage constraints
    for i in nodes:
        print(i)
        for j in nodes:
            if A[i, j]:
                for k in range(m):
                    model.addConstr(flow[i, j, k] <= edges[i, j, k] * 1000)  # Assumed large constant
                    model.addConstr(flow[i, j, k] <= nodes[i, k] * 1000)
                    model.addConstr(flow[i, j, k] <= nodes[j, k] * 1000)

    # Solve model
    model.optimize()

    # Output solution
    if model.status == GRB.OPTIMAL:
        print('Optimal solution found:')
        for k in range(m):
            print(f"\nSubgraph {k + 1}:")
            for i in nodes:
                for j in nodes:
                    if x[i, j, k].X > 0.5:
                        print(f"Edge {i}-{j} is in subgraph {k + 1}")
                    if y[i, k].X > 0.5:
                        print(f"Node {i} is in subgraph {k + 1}")
    else:
        print('No optimal solution found.')
