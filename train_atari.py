from thefool.DQN.dqn_agent import DQN
from thefool.PPO.ppo_agent import PPO
from thefool.SAC.sac_agent import SAC
from thefool.common.env.make import make_atari_game

train_env, test_env = make_atari_game("BreakoutNoFrameskip-v4", 8, obs_shape=[84, 84], stack_frame=4)

model = DQN(train_env, test_env, use_double=False, use_dueling=False, use_per=True, use_average_is=False, n_step=1)
model.learn(total_steps=100000000)
