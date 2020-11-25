from typing import NamedTuple


class HParams(NamedTuple):
    batch_size: Shape = 32
    replay_memory_size: int = 1000000
    agent_history_len: int = 4
    target_network_update_frequency: int = 10000
    discount: float = 0.99
    action_repeat: int = 4
    update_frequency: int = 4
    learning_rate: float = 0.00025
    gradient_momentum: float = 0.95
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    final_exploration_frame: int = 1000000
    replay_start: int = 50000
    no_op_max: int = 30
