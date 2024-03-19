
from statistics import mean, median

import gymnasium as gym

from .ai.DQN import DQN
from .config import (HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
                     ORDINARY_TETRIS_MINOS)


def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))


def start():
    env = gym.make("tetris-v1", height=20, width=10, minos=ORDINARY_TETRIS_MINOS)
    env.reset()
    done = False

    while not done:
        print(env.render())
        command = input("Enter action:")
        # command が action に無い場合は無視
        if command not in HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP:
            continue
        action = HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP[command].id
        _, _, done, _, _ = env.step(action)
    # GameOver
    print(env.render())


def train():
    env = gym.make("tetris-v1", height=20, width=10, minos=ORDINARY_TETRIS_MINOS, action_mode=1)
    env.reset()

    # env から input と output の次元を取得
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.spaces[0].n * env.action_space.spaces[1].n # action_mode = 1 の場合
    agent = DQN(input_size, output_size)
    # agent.load() # 途中経過をロード
    
    running = True
    total_games = 0
    total_steps = 0
    while running:
        steps, rewards = agent.train(env, episodes=25)
        total_games += len(rewards)
        total_steps += steps
        agent.save() # 途中経過を保存
        print(env.render())
        print("==================")
        print("* Total Games: ", total_games, "       ")
        print("* Total Steps: ", total_steps, "       ")
        print("* Epsilon: ", agent.epsilon, "       ")
        print("*")
        print("* Average: ", sum(rewards) / len(rewards), "       ")
        print("* Median: ", median(rewards), "       ")
        print("* Mean: ", mean(rewards), "       ")
        print("* Min: ", min(rewards), "       ")
        print("* Max: ", max(rewards), "       ")
        print("==================")

def simulate():
    env = gym.make("tetris-v1", height=20, width=10, minos=ORDINARY_TETRIS_MINOS, action_mode=1)
    obs, _ = env.reset()

    # env から input と output の次元を取得
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.spaces[0].n * env.action_space.spaces[1].n # action_mode = 1 の場合
    agent = DQN(input_size, output_size, epsilon=0.0, epsilon_decay=0.0, epsilon_min=0.0)
    agent.load() # 途中経過をロード

    running = True
    while running:
        actions = env.get_possible_actions()
        print(actions)
        action = agent.act(obs, actions)
        _, _, done, _, _ = env.step(action)
        print(overwrite_print(env.render(), 0))

        if done:
            obs, _ = env.reset()
            input("Press Enter to continue...")

if __name__ == "__main__":
    train()
