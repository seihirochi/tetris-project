from tetris_gym import Tetris

from .config import (
    HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    ORDINARY_TETRIS_ACTIONS,
    ORDINARY_TETRIS_MINOS,
    ALL_HARDDROP_ACTIONS,
)
from .controller import DQN, DQNTrainerController, HumanController


def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))


def start():
    game = Tetris(20, 10, ORDINARY_TETRIS_MINOS, ORDINARY_TETRIS_ACTIONS)
    controller = HumanController(
        ORDINARY_TETRIS_ACTIONS,
        HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    )
    while game.game_over is False:
        overwrite_print(game.render(), 0)
        action = controller.get_action()
        game.step(action.id)

    # Game Over
    overwrite_print(game.render(), 0)


def train():
    epoch = 5000
    game = Tetris(20, 10, ORDINARY_TETRIS_MINOS, ALL_HARDDROP_ACTIONS)
    state = game.observe()
    model = DQN(state.size, len(ALL_HARDDROP_ACTIONS))
    controller = DQNTrainerController(ALL_HARDDROP_ACTIONS, model, 0.1)
    rewards = []
    # 累積報酬の割引率
    gamma = 0.9
    print("Training...")
    for i in range(epoch):
        print(f"Epoch {i+1}/{epoch}")
        prev_rewards = 0
        while game.game_over is False:
            action = controller.get_action(state)
            game.hard_drop_step(action.id)
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
