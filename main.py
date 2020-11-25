from absl import app
from absl import flags

from dqn import DQN, HParams, train, env_from_name


# experiment:
flags.DEFINE_string("env", "pong", "Environment name")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("num_episodes", 10, "Number of episodes to train on")
# hparams:
flags.DEFINE_integer(
    "batch_size",
    32,
    "Number of training cases over which each stochastic gradient descent (SGD) update is computed",
)
flags.DEFINE_integer(
    "replay_memory_size",
    1000000,
    "SGD updates are sampled from this number of most recent frames",
)
flags.DEFINE_integer(
    "agent_history_length",
    4,
    "The number of most recent frames experienced by the agent that are given as  input to the Q network",
)
flags.DEFINE_integer(
    "target_network_update_frequency",
    10000,
    "The frequency (measured in the number of parameters update) with which the target network is updated (this corresponds to the parameter C from Algorithm 1)",
)
flags.DEFINE_float(
    "discount", 0.99, "Discount factor gamma used in the Q-learning update"
)
flags.DEFINE_integer(
    "action_repeat",
    4,
    "Repeat each action selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4th input frame",
)
flags.DEFINE_integer(
    "update_frequency",
    4,
    "The number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates.",
)
flags.DEFINE_float("learning_rate", 0.00025, "The learning rate used by the RMSProp")
flags.DEFINE_float("gradient_momentum", 0.95, "Gradient momentum used by the RMSProp")
flags.DEFINE_float(
    "squared_gradient_momentum",
    0.95,
    "Squared gradient (denominator) momentum used by the RMSProp",
)
flags.DEFINE_float(
    "min_squared_gradient",
    0.01,
    "Constant added to the squared gradient in the denominator of the RMSProp update",
)
flags.DEFINE_float(
    "initial_exploration", 1.0, "Initial value of ɛ in ɛ-greedy exploration"
)
flags.DEFINE_float(
    "final_exploration", 0.01, "Final value of ɛ in ɛ-greedy exploration"
)
flags.DEFINE_integer(
    "final_exploration_frame",
    1000000,
    "The number of frames over which the initial value of ɛ is linearly annealed to its final value",
)
flags.DEFINE_integer(
    "replay_start",
    50000,
    "A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory",
)
flags.DEFINE_integer(
    "no_op_max",
    30,
    'Maximum numer of "do nothing" actions to be performed by the agent at the start of an episode',
)

FLAGS = flags.FLAGS


def main(argv):
    env = env_from_name(FLAGS.env)
    in_shape = (4, 84, 84)
    hparams = HParams(
        batch_size=FLAGS.batch_size,
        replay_memory_size=FLAGS.replay_memory_size,
        agent_history_len=FLAGS.agent_history_length,
        target_network_update_frequency=FLAGS.target_network_update_frequency,
        discount=FLAGS.discount,
        action_repeat=FLAGS.action_repeat,
        update_frequency=FLAGS.update_frequency,
        learning_rate=FLAGS.learning_rate,
        gradient_momentum=FLAGS.gradient_momentum,
        squared_gradient_momentum=FLAGS.squared_gradient_momentum,
        min_squared_gradient=FLAGS.min_squared_gradient,
        initial_exploration=FLAGS.initial_exploration,
        final_exploration=FLAGS.final_exploration,
        final_exploration_frame=FLAGS.final_exploration_frame,
        replay_start=FLAGS.replay_start,
        no_op_max=FLAGS.no_op_max,
    )
    agent = DQN(env.action_space.n, in_shape, hparams, FLAGS.seed)
    return train(agent, env, FLAGS.num_episodes)


if __name__ == "__main__":
    app.run(main)
