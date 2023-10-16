from thefool.DQN.dqn_agent import DQN
from thefool.common.env.make import make_atari_game

train_env, test_env = make_atari_game("PongNoFrameskip-v4", 8, obs_shape=[84, 84], stack_frame=4)


model = DQN(train_env, test_env, use_dueling=True, use_double=True, use_per=True, n_step=3)
model.learn(total_steps=100000000)
