Hierarchical Latent Dirichlet Allocation
----------------------------------------

Hierarchical Latent Dirichlet Allocation (hLDA) addresses the problem of learning topic hierarchies from data. The model relies on a non-parametric prior called the nested Chinese restaurant process, which allows for arbitrarily large branching factors and readily accommodates growing
data collections. The hLDA model combines this prior with a likelihood that is based on a hierarchical variant of latent Dirichlet allocation.

[Hierarchical Topic Models and the Nested Chinese Restaurant Process](http://www.cs.columbia.edu/~blei/papers/BleiGriffithsJordanTenenbaum2003.pdf)

[The Nested Chinese Restaurant Process and Bayesian Nonparametric Inference of Topic Hierarchies](http://cocosci.berkeley.edu/tom/papers/ncrp.pdf)

Implementation
--------------

- [hlda/sampler.py](hlda/sampler.py) is the Gibbs sampler for hLDA inference, based on the implementation from [Mallet](http://mallet.cs.umass.edu/topics.php) having a fixed depth on the nCRP tree.


Installation
------------

- Simply use `pip install hlda` to install the package.
- An example notebook that infers the hierarchical topics on the BBC Insight corpus can be found in [notebooks/bbc_test.ipynb](notebooks/bbc_test.ipynb).

ZW: Additional Conda Setup
--------------------------

From a basic anaconda environment (2020.02), a number of additional packages need to be installed for the notebook to work out of the box:
- `conda update -n base -c defaults conda`
- `conda install jupyter matplotlib nltk wordcloud`

Black:
- `conda install black`
- https://neuralcoder.science/Black-Jupyter/

Tomotopy:
- `conda install py-cpuinfo`
- `pip install tomotopy`

Also make sure the necessary NLTK packages are downloaded and accessible.  