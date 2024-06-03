---
id: Sampling
aliases: []
tags: []
---
Sampling generates a large number of possibly feasible districts with looser criteria.

Methods:
    single flood fill - choose rand start and using rook/queen moves randomly create district with balanced pop. repeat for all districst.
    multi flood fill - same as above but create all districts simultaneously. avoids sections of state being boxed off.
    MCMC (markov chain monte carlo) - start from existing plan and take random steps to new plans. possible steps:
        swapping district assignment of border vertices.
        changing district assignment of border vertex.
        splitting/recombining districts. create spanning tree for merged adjacent districts and choose random cut vertex
