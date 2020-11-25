#!/usr/bin/env python

from distutils.core import setup

setup(
    name="dqn-jax",
    version="0.0.1",
    description="A jax implementation of the dqn original algorithm",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignateli/human-level-control-through-deep-reinforcement-learning",
    packages=["dqn"],
    install_requires=["bsuite", "jaxlib", "jax"],
)
