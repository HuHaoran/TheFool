import numpy as np
import gymnasium as gym
from thefool.common.env.env_wrapper import EnvWrapper, BatchEnvWrapper
from thefool.common.env.atari_wrapper import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, atari_preprocess

def make_raw_atari_game(env_name, num_envs):

    state_preprocess = lambda s: s / 255.0
    reward_preprocess = lambda r: np.clip(r, -1.0, 1.0)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        # raw 用不到
        env = NoopResetEnv(env, noop_max=15)
        # env = MaxAndSkipEnv(env)
        # env = EpisodicLifeEnv(env)
        wrapped_env = EnvWrapper(
            env,
            r_preprocess=reward_preprocess,
            s_preprocess=state_preprocess
        )
        envs.append(wrapped_env)
    batch_env = BatchEnvWrapper(envs)

    return batch_env


def make_atari_game(env_name, num_envs, obs_shape, stack_frame=4, test_num_envs=8):
    def state_preprocess(state):
        state = atari_preprocess(state, obs_shape)
        state = np.array(state, dtype=np.float32)
        return state / 255.0

    reward_preprocess = lambda r: np.clip(r, -1.0, 1.0)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        env = NoopResetEnv(env, noop_max=30)
        # env = FireResetEnv(env)
        env = MaxAndSkipEnv(env)
        env = EpisodicLifeEnv(env)
        wrapped_env = EnvWrapper(
            env,
            obs_shape,
            stack_frame,
            r_preprocess=reward_preprocess,
            s_preprocess=state_preprocess
        )
        envs.append(wrapped_env)

    # 测试环境
    test_envs = []
    for i in range(test_num_envs):
        test_env = gym.make(env_name)
        test_env = NoopResetEnv(test_env, noop_max=30)
        test_env = FireResetEnv(test_env)
        test_env = MaxAndSkipEnv(test_env)
        wrapped_env = EnvWrapper(
            test_env,
            obs_shape,
            stack_frame,
            r_preprocess=None,
            s_preprocess=state_preprocess
        )
        test_envs.append(wrapped_env)

    batch_train_env = BatchEnvWrapper(envs)
    batch_test_env = BatchEnvWrapper(test_envs)

    return batch_train_env, batch_test_env
