
import gymnasium as gym

import tetris_gym

from .config import (HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
                     ORDINARY_TETRIS_ACTIONS, ORDINARY_TETRIS_MINOS)

# from .controller import DQN, DQNTrainerController, HumanController


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
    epoch = 5000
    game = Tetris(20, 10, ORDINARY_TETRIS_MINOS, ORDINARY_TETRIS_ACTIONS, 1)
    state = game.observe()
    model = DQN(state.size, len(ORDINARY_TETRIS_ACTIONS))
    controller = DQNTrainerController(ORDINARY_TETRIS_ACTIONS, model, 0.1)
    rewards = []
    # 累積報酬の割引率
    gamma = 0.9
    print("Training...")
    for i in range(epoch):
        print(f"Epoch {i+1}/{epoch}")
        prev_rewards = 0
        while game.game_over is False:
            action = controller.get_action(state)
            game.step(action.id)
            next_state = game.observe()
            reward = controller.evaluate(next_state)
            reward = reward + gamma * prev_rewards
            controller.train(state, action, next_state, reward)
            state = next_state
            prev_rewards = reward
        print(game.render(), end="\r")
        print("Reward:", reward)
        rewards.append(reward)
        print("Current epsilon:", controller.epsilon)
        print("Epoch finished!")
        game.reset()
    print("Done!")
    with open("rewards.csv", "w") as f:
        f.write("epoch,reward\n")
        for i, reward in enumerate(rewards):
            f.write(f"{i},{reward}\n")
    print("Saving model...")
    model.save("model.pth")
    print("Done!")
    print("Training finished!")
    model.save("model.pth")
    print("Done!")
    print("Training finished!")


if __name__ == "__main__":
    train()
