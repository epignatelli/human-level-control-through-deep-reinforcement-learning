![Build](https://github.com/epignatelli/human-level-control-through-deep-reinforcement-learning/workflows/build/badge.svg)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# Human-level-control-through-deep-reinforcement-learning
A jax/stax implementation of the Nature paper: _Human-level control through deep reinforcement learning_ [[1]](https://www.nature.com/articles/nature14236)

The agent at `qdn.agent.py` implements the `bsuite.baseline.base.Agent` interface.
The `dqn//train.py` interfaces with a `dm_env.Environment`.
We wrap the [gym-atari](https://github.com/openai/gym) suite using the `bsuite.utils.gym_wrapper.DMEnvFromGym` adapter into a `dqn.AtariEnv` to implement historical observations and actions repeat.

Implementation status of some of the techniques used in the paper:
- [x] Experience replay [[2]](http://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf)
- [x] Target network [[1]](https://www.nature.com/articles/nature14236)
- [x] Reward clipping [[1]](https://www.nature.com/articles/nature14236)
- [x] Linear ε annealing [[6]](http://incompleteideas.net/book/RLbook2020.pdf)
- [x] Frame skipping [[5]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.261.274&rep=rep1&type=pdf)
- [x] Bellman error clipping [[1]](https://www.nature.com/articles/nature14236)
- [ ] Consecutive no-ops prevention [[1]](https://www.nature.com/articles/nature14236)

## Installation
To run the algorithm on a GPU, I suggest to [install](https://github.com/google/jax#pip-installation) the gpu version of `jax` [[4]](https://github.com/google/jax). You can then install this repo using [Anaconda python](https://www.anaconda.com/products/individual) and [pip](https://pip.pypa.io/en/stable/installing/).
```sh
conda env create -n dqn
conda activate dqn
pip install git+https://github.com/epignatelli/human-level-control-through-deep-reinforcement-learning
```

## References
[1] [_Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533._](https://www.nature.com/articles/nature14236)


[2] [_Lin, L.-J. Reinforcement learning for robots using neural networks. Technical Report, DTIC Document (1993)_](http://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf)


[3] [Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.](https://arxiv.org/pdf/1312.5602.pdf)


[4] [_Bradbury, J., Frostig, R., Hawkins, P., Johnson, M.J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., Zhang, Q. JAX: composable transformations of Python+NumPy programs. 2018_](https://github.com/google/jax)


[5] [_Bellemare, M. G., Veness, J. & Bowling, M. Investigating contingency awareness using Atari 2600 games. Proc. Conf. AAAI. Artif. Intell. 864–871 (2012)_](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.261.274&rep=rep1&type=pdf)

[6] [_Sutton, R.S. and Barto, A.G., 1998. Introduction to reinforcement learning (Vol. 135). Cambridge: MIT press._](http://incompleteideas.net/book/RLbook2020.pdf)
