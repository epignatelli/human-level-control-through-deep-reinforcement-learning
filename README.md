[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# Human-level-control-through-deep-reinforcement-learning
A jax/stax implementation of the Nature paper: _Human-level control through deep reinforcement learning_ [[1]](https://www.nature.com/articles/nature14236)

- [x] Experience replay
- [x] Target network
- [x] Bellmand error clipping
- [ ] Exploration annealing
- [ ] Consecutive no-ops prevention

 
## Installation
To run the algorithm on a GPU, I suggest to [install](https://github.com/google/jax#pip-installation) the gpu version of `jax`.
You can then install this repo using [Anaconda python](https://www.anaconda.com/products/individual) and [pip](https://pip.pypa.io/en/stable/installing/).
```sh
conda env create -n dqn
conda activate dqn
pip install git+https://github.com/epignatelli/human-level-control-through-deep-reinforcement-learning
```

## References
[[1]](https://www.nature.com/articles/nature14236) _Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533._
[[2]](http://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf) _Lin, L.-J. Reinforcement learning for robots using neural networks. Technical Report, DTIC Document (1993)_
