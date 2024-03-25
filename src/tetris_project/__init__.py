from statistics import mean, median

import gymnasium as gym

# from .ai import NN, NNPlayerController, NNTrainerController, WEIGHT_OUT_PATH
from .dqn import DQN, DQNTrainerController, WEIGHT_OUT_PATH
from .config import (
    ALL_HARDDROP_ACTIONS,
    HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    ORDINARY_TETRIS_ACTIONS,
    ORDINARY_TETRIS_MINOS,
    TETRIS_HEIGHT,
    TETRIS_WIDTH,
)
from .controller import HumanController


def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))


def start():
    env = gym.make(
        "tetris-v1",
        height=TETRIS_HEIGHT,
        width=TETRIS_WIDTH,
        minos=ORDINARY_TETRIS_MINOS,
        action_mode=0,
    )
    env.reset()
    done = False
    controller = HumanController(
        ORDINARY_TETRIS_ACTIONS,
        HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    )
    while not done:
        overwrite_print(env.render(), 0)
        action = controller.get_action(env)
        _, _, done, _, _ = env.step(action)
    # GameOver
    print(env.render())


def train_cuda():
    train("cuda")


def train_mps():
    train("mps")


def train(device="cpu"):
    print("Device: ", device)

    env = gym.make(
        "tetris-v1",
        height=TETRIS_HEIGHT,
        width=TETRIS_WIDTH,
        minos=ORDINARY_TETRIS_MINOS,
        action_mode=1,
        env_mode="natural",
    )
    env.reset()

    model = DQN(
        (TETRIS_HEIGHT, TETRIS_WIDTH),
        len(ORDINARY_TETRIS_MINOS),
        3,
        len(ALL_HARDDROP_ACTIONS),
    ).to(device)
    controller = DQNTrainerController(
        ALL_HARDDROP_ACTIONS,
        model,
        discount=1.00,
        epsilon=1.00,
        epsilon_min=0.001,
        epsilon_decay=0.999,
        device=device,
        mino_kinds=len(ORDINARY_TETRIS_MINOS),
    )

    # 既存の parametor を load する場合はファイル名指定
    # model.load("param/NN4.weights.h5")

    running = True
    total_games = 0
    total_steps = 0
    while running:
        steps, rewards = controller.train(env, episodes=20)
        total_games += len(rewards)
        total_steps += steps
        model.save()  # 途中経過を保存

        print(env.render())
        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", controller.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards))
        print("* Median: ", median(rewards))
        print("* Mean: ", mean(rewards))
        print("* Min: ", min(rewards))
        print("* Max: ", max(rewards))
        print("==================")


def simulate():
    env = gym.make(
        "tetris-v1",
        height=TETRIS_HEIGHT,
        width=TETRIS_WIDTH,
        minos=ORDINARY_TETRIS_MINOS,
        action_mode=1,
    )
    env.reset()

    # input: 状態特徴量
    # output: 今後の報酬の期待値
    input_size = env.observation_space.shape[0]
    output_size = 1
    # model = NN(input_size, output_size)
    # controller = NNPlayerController(ALL_HARDDROP_ACTIONS, model)
    model = DQN(
        (TETRIS_HEIGHT, TETRIS_WIDTH),
        len(ORDINARY_TETRIS_MINOS),
        3,
        len(ALL_HARDDROP_ACTIONS),
    )
    controller = DQNTrainerController(
        ALL_HARDDROP_ACTIONS,
        model,
        discount=1.00,
        epsilon=0.00,
        epsilon_min=0.00,
        epsilon_decay=1.00,
    )

    # 既存の parametor を load する場合は param 配下のファイル名指定
    model.load(WEIGHT_OUT_PATH)

    running = True
    while running:
        action = controller.get_action(env)
        _, _, done, _, _ = env.step(action)
        print(overwrite_print(env.render(), 0))
        if done:
            _, _ = env.reset()
            input("Press Enter to continue...")


if __name__ == "__main__":
    train()
