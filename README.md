# Unsupervised learning of Goal Spaces for Intrinsically Motivated Exploration

This repository hosts the code to reproduce the results presented in the paper [Unsupervised Learning of Goal Spaces for Intrinsically Motivated Goal Exploration](https://openreview.net/forum?id=S1DWPP1A-). In this paper, we propose a novel exploration algorithmic architecture that uses a goal space learned using representation learning algorithms. Experiments are performed on two simple tasks in which multi-joint arm must handle and gather an oobject in a 2D space.

## Running the experiments 

To run a single experiment, you can run one of the three following python scripts:

+ `rpe.py` to perform a Random Parameterization Exploration
+ `rge_efr.py` to perform a Random Goal Exploration using Engineered Features Representation
+ `rge_rep.py` to perform a Random Goal Exploration using a learned Representation

You can also run a full campaign batch by executing `campaign.sh`.

Finally, to generate the different figures out of the raw results, you can use the two notebooks:

+ `Experiment_Visualization.ipynb` to visualize the data of a single run
+ `Performance_Comparison.ipynb`  to visualize the compared performance of multiple runs