---
id: Compactness
aliases: []
tags: []
---
notes from this paper: https://gking.harvard.edu/sites/scholar.harvard.edu/files/gking/files/ajps.12603.pdf

compactness is a function only of geometry, not geography, population, or even other districts.
some common measures:
    - Length-Width Ratio: of the minimum bounding rectangle
    - Convex-Hull:    A district / A convex hull
    - Reock:          A district / A circumscribed circle
    - Polsby-Popper:  A district / A circle with same area
    - Boyce-Clark:    mean standard deviation in distance from centroid
contrasting judges, these measures are rotationally invariant. these measures try to address this:
    - X-symmetry:     A overlap with horizontal reflection / A original
people often only consider a couple salient features of an image:
    - Significant Corners: foud via CNN / edge detection algorithm

when surveying humans for compactness, we can use either paired comparison, or full ranking methods
paired comparison is bad:
    - humans are goot at seeing the full picture, paired comparisons doesn't leverage this
    - paired comparisons is easy / boring, respondents are less engaged
    - the goal is a single measure of compactness, but in paired comparisons, different measures may be used at different times. i.e. the relative importance of attributes is unconscious
to evaluate paired comparisons: evaluate (intra / inter) coder reliability of actual vs. implied comparisons in the data

--procedure
collect 6 sets of rankings of 100 districts as data
extract first principle component
calculate top metrics for each district
trained AdaBoosted decision trees, SVM's, and random forests

validated by mixing group size, mode of administration, type of respondent (groduade, undergrad, amzn mturk)
created 6 groups from above types of people, trained on 5, predicted last group. no large variation when using different groups as test set

hypothesis: all reasonable humans grade districts similarly
took 96 public officials, 102 other, had them rank 20 districts.
found no statistically significant correlations among different groups of respondents
found measure favors: square, with less arms, islands, pockets, or jagged edges
