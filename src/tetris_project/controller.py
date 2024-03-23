import copy
from abc import ABC, abstractmethod
from typing import List, Mapping, Tuple

import numpy as np
from gymnasium import Env

from tetris_gym import Action


class Controller(ABC):
    def __init__(self, actions: set[Action]) -> None:
        self.actions = actions
        self.action_map = {action.id: action for action in actions}

    def get_possible_actions(self, env: Env) -> List[Tuple[Action, np.ndarray]]:
        # Env の情報を見て可能な行動を return (行動決定を controller に一任する)
        actions = []
        if env.action_mode == 0:
            # actions = [0, 1, 2, 3, 4, 5, 6]
            # ※ 現時点では機械学習には用いていない
            pass
        elif env.action_mode == 1:
            for action in self.actions:
                y, rotate, hold = action.convert_to_tuple(env.tetris.board.width)
                if hold:
                    tetris_copy = copy.deepcopy(env.unwrapped.tetris)
                    if tetris_copy.hold():
                        actions.append((
                            action,
                            tetris_copy.observe()
                        ))
                    continue
                tetris_copy = copy.deepcopy(env.unwrapped.tetris)
                flag = tetris_copy.move_and_rotate_and_drop(y, rotate)
                if flag:
                    actions.append((
                        action,
                        tetris_copy.observe()
                    ))
        return actions

    @abstractmethod
    def get_action(self, env: Env) -> Action:
        pass


class HumanController(Controller):
    def __init__(self, actions: set[Action], input_map: Mapping[str, Action]) -> None:
        super().__init__(actions)
        self.input_map = input_map

    def get_action(self, _: Env) -> Action:
        while True:
            try:
                action_input = input("Enter action: ")
                action = self.input_map[action_input]
                return self.action_map[action.id]
            except KeyError:
                print("Invalid action")


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim


# class DQN(nn.Module):
#     def __init__(self, input_size: int, output_size: int) -> None:
#         super().__init__()
#         # 3層のニューラルネットワーク, 最後にsoftmax関数を適用
#         # Linear(input_size, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, output_size) -> Softmax
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_size)
#         self.softmax = nn.Softmax(dim=0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)

#     def save(self, path: str) -> None:
#         torch.save(self.state_dict(), path)

#     def load(self, path: str) -> None:
#         self.load_state_dict(torch.load(path))


# class DQNTrainerController(Controller):
#     def __init__(self, actions: set[Action], model: nn.Module, epsilon: float) -> None:
#         super().__init__(actions)
#         self.model = model
#         self.epsilon = epsilon

#     def get_action(self, state: TetrisState) -> Action:
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(list(self.actions))
#         else:
#             with torch.no_grad():
#                 state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32)
#                 q_values = self.model(state_tensor)
#                 return self.action_map[torch.argmax(q_values).item()]

#     def train(
#         self, state: TetrisState, action: Action, next_state: TetrisState, reward: float
#     ) -> None:
#         state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32)
#         next_state_tensor = torch.tensor(next_state.to_tensor(), dtype=torch.float32)
#         q_values = self.model(state_tensor)
#         next_q_values = self.model(next_state_tensor)
#         target_q_values = q_values.clone()
#         target_q_values[action.id] = reward + torch.max(next_q_values)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         loss = criterion(q_values, target_q_values)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     def evaluate(self, state: TetrisState) -> float:
#         return state.score + state.turn


# class DQNPlayerController(Controller):
#     def __init__(self, actions: set[Action], model: nn.Module) -> None:
#         super().__init__(actions)
#         self.model = model

#     def get_action(self, state: TetrisState) -> Action:
#         with torch.no_grad():
#             state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32)
#             q_values = self.model(state_tensor)
#             return self.actions[torch.argmax(q_values).item()]
