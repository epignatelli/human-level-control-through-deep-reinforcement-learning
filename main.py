from absl import app
from absl import flags

from dqn import DQN, HParams, train, env_from_name


# experiment:
flags.DEFINE_string("env", "pong", "Environment name")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_integer("num_episodes", 10, "Number of episodes to train on")
# hparams:
flags.DEFINE_integer('batch_size', 32, "")
flags.DEFINE_integer('replay_memory_size', 1000000, "")
flags.DEFINE_integer('agent_history_len', 4, "")
flags.DEFINE_integer('target_network_update_frequency', 10000, "")
flags.DEFINE_float('discount', 0.99, "")
flags.DEFINE_integer('action_repeat', 4, "")
flags.DEFINE_integer('update_frequency', 4, "")
flags.DEFINE_float('learning_rate', 0.00025, "")
flags.DEFINE_float('gradient_momentum', 0.95, "")
flags.DEFINE_float('squared_gradient_momentum', 0.95, "")
flags.DEFINE_float('min_squared_gradient', 0.01, "")
flags.DEFINE_float('initial_exploration', 1.0, "")
flags.DEFINE_float('final_exploration', 0.01, "")
flags.DEFINE_integer('final_exploration_frame', 1000000, "")
flags.DEFINE_integer('replay_start', 50000, "")
flags.DEFINE_integer('no_op_max', 30, "")

FLAGS = flags.FLAGS


def main(argv):
    env = env_from_name(FLAGS.env)
    in_shape = (4, 84, 84)
    hparams = HParams(
        batch_size=FLAGS.batch_size,
        replay_memory_size=FLAGS.replay_memory_size,
        agent_history_len=FLAGS.agent_history_len,
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

