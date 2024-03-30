import os
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gymnasium import Env

import torch
import torch.nn as nn

from tetris_gym import Action
from tetris_gym.tetris import LINE_CLEAR_SCORE, Mino
from tetris_project.controller import Controller
from tetris_project.config import get_ordinary_tetris_mino_one_hot
import copy

WEIGHT_OUT_PATH = os.path.join(os.path.dirname(__file__), "out.pth")


class Experience:
    def __init__(
        self,
        observe: Tuple[np.ndarray, int, np.ndarray, int],
        action: Action,
        reward: float,
        next_observe: Tuple[np.ndarray, int, np.ndarray, int],
        done: bool,
    ):
        self.observe = observe
        self.action = action
        self.reward = reward
        self.next_observe = next_observe
        self.done = done


class ExperienceBuffer:
    def __init__(self, buffer_size=20000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, size: int) -> List[Experience]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]

    def len(self) -> int:
        return len(self.buffer)


# DQNの設計について
# このDQNには入力として
# 1. 現在の盤面の状態(10x20x2 (0: void, 1: block))
# 2. 現在のミノID(0~6)
# 3. 現在のHoldミノID(0~6)
# 4. 現在のNextミノID(0~6)x3
# が与えられる
# 出力としては4つの回転方向と11つの横移動の組み合わせ(4x11=44)を返す
# 盤面情報を2D情報として扱うことでより高度な予測が可能になると考えられる
# そのため、盤面情報にはCNNを適用する。
# さらに、ミノID, HoldミノID, NextミノIDはそれぞれEmbedding層を通して結合する
# IDではそれぞれの値が離散的であるため、One-hot表現を通してEmbedding層に入力する
# これによって、盤面情報とそれ以外の情報を結合することができる
class DQN(nn.Module):
    def __init__(
        self,
        board_size: Tuple[int, int],
        mino_kinds: int,
        next_minos_size: int,
        output_size: int,
    ) -> None:
        super(DQN, self).__init__()
        self.board_size = board_size
        self.mino_kinds = mino_kinds
        self.next_minos_size = next_minos_size
        self.output_size = output_size

        # 盤面情報を扱うCNN, 3層で盤面情報は(20, 10)の2D情報, 0: void, 1: blockの1チャンネル
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.mino_embedding = nn.Embedding(mino_kinds, 1)
        self.hold_mino_embedding = nn.Embedding(mino_kinds, 1)
        self.next_minos_embedding = nn.Embedding(mino_kinds * next_minos_size, 1)
        self.layer5 = nn.Sequential(
            nn.Linear(64 * 2 * 1 + 7 * (2 + next_minos_size), 128),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.layer7 = nn.Linear(64, output_size)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        board, mino_id, next_mino_ids, hold_mino_id = x
        x = self.layer1(board)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 64 * 2 * 1)

        mino_id = self.mino_embedding(mino_id)
        hold_mino_id = self.hold_mino_embedding(hold_mino_id)
        next_mino_ids = self.next_minos_embedding(next_mino_ids)
        mino_id = mino_id.view(-1, self.mino_kinds)
        hold_mino_id = hold_mino_id.view(-1, self.mino_kinds)
        next_mino_ids = next_mino_ids.view(-1, self.mino_kinds * self.next_minos_size)

        x = torch.cat([x, mino_id, hold_mino_id, next_mino_ids], dim=1)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

    def save(self) -> None:
        torch.save(self.state_dict(), WEIGHT_OUT_PATH)

    def load(self, path: str) -> None:
        path = os.path.join(os.path.dirname(__file__), path)
        if Path(path).is_file():
            self.load_state_dict(torch.load(path))


class DQNTrainerController(Controller):
    def __init__(
        self,
        actions: set[Action],
        model: nn.Module,
        discount=0.95,
        epsilon=0.50,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        device="cpu",
        mino_kinds=7,
    ) -> None:
        super().__init__(actions)
        self.action_size = len(actions)
        self.model = model
        self.discount = discount  # 割引率
        self.epsilon = epsilon  # ε-greedy法 の ε
        self.epsilon_min = epsilon_min  # ε-greedy法 の ε の最小値
        self.epsilon_decay = epsilon_decay  # ε-greedy法 の ε の減衰率

        # Experience Replay Buffer (上下 2 つ)
        self.experience_buffer = ExperienceBuffer()

        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.SmoothL1Loss()
        self.mino_kinds = mino_kinds
        self.losses = []

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

    def train(self, env: Env, episodes: int) -> Tuple[int, List[float]]:
        steps = 0
        rewards = []
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state, env)
                next_state, reward, done, _, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)

                if reward >= LINE_CLEAR_SCORE[4]:
                    print("★★★★★★★★★★ 4 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[3]:
                    print("★★★★★★★★★★ 3 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[2]:
                    print("★★★★★★★★★★ 2 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[1]:
                    print("★★★★★★★★★★ 1 Line Clear! ★★★★★★★★★★")

                total_reward += reward
                steps += 1
            self.learn()
            if episode % 10 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            rewards.append(total_reward)
        return steps, rewards

    def get_action(self, state, env) -> Action:
        if random.random() < self.epsilon:
            possible_states = self.get_possible_actions(env)
            possible_action_ids = [action.id for action, _ in possible_states]
            action_id = random.choice(possible_action_ids)
            res = self.action_map[action_id]
            return res
        else:
            board, mino_id, next_mino_ids, hold_mino_id = state
            board = (
                torch.tensor(board, dtype=torch.float32).to(self.device).unsqueeze(0)
            )
            mino_id = torch.tensor(
                get_ordinary_tetris_mino_one_hot(mino_id), dtype=torch.long
            ).to(self.device)
            next_mino_ids = torch.tensor(
                [
                    get_ordinary_tetris_mino_one_hot(next_mino_id)
                    for next_mino_id in next_mino_ids
                ],
                dtype=torch.long,
            ).to(self.device)
            hold_mino_id = torch.tensor(
                get_ordinary_tetris_mino_one_hot(hold_mino_id), dtype=torch.long
            ).to(self.device)
            possible_states = self.get_possible_actions(env)
            possible_action_ids = [action.id for action, _ in possible_states]
            zeros = torch.zeros(self.action_size, dtype=torch.float32).to(self.device)
            zeros[possible_action_ids] = 1
            q_values = self.model((board, mino_id, next_mino_ids, hold_mino_id))
            min_q_values = torch.min(q_values)
            q_values = q_values - min_q_values
            q_values = q_values * zeros
            action_id = torch.argmax(q_values).item()
            res = self.action_map[action_id]
            return res

    def remember(
        self,
        state: Tuple[np.ndarray, int, np.ndarray, int],
        action: Action,
        reward: float,
        next_state: Tuple[np.ndarray, int, np.ndarray, int],
        done: bool,
    ) -> None:
        self.experience_buffer.add(Experience(state, action, reward, next_state, done))

    def learn(self) -> None:
        if self.experience_buffer.len() < 32:
            return

        experiences = self.experience_buffer.sample(32)

        # board情報は(10, 20)の2D情報
        # TorchはNCHW形式であるため、(バッチサイズ, チャンネル数, 高さ, 幅)の形式に変換する
        board = (
            torch.tensor([exp.observe[0] for exp in experiences], dtype=torch.float32)
            .to(self.device)
            .unsqueeze(1)
        )  # テンソルの2階目に[0,1]の次元を追加
        mino_id = torch.tensor(
            [get_ordinary_tetris_mino_one_hot(exp.observe[1]) for exp in experiences],
            dtype=torch.long,
        ).to(self.device)
        # next_mino_ids (32, 3, 7) -> (32, 21) に変換
        next_mino_ids = []
        for exp in experiences:
            next_mino_ids_each = []
            for nmid in exp.observe[2]:
                for value in get_ordinary_tetris_mino_one_hot(nmid):
                    next_mino_ids_each.append(value)
            next_mino_ids.append(next_mino_ids_each)
        next_mino_ids = torch.tensor(
            next_mino_ids,
            dtype=torch.long,
        ).to(self.device)
        hold_mino_id = torch.tensor(
            [get_ordinary_tetris_mino_one_hot(exp.observe[3]) for exp in experiences],
            dtype=torch.long,
        ).to(self.device)

        next_board = (
            torch.tensor(
                [exp.next_observe[0] for exp in experiences], dtype=torch.float32
            )
            .to(self.device)
            .unsqueeze(1)
        )
        next_mino_id = torch.tensor(
            [
                get_ordinary_tetris_mino_one_hot(exp.next_observe[1])
                for exp in experiences
            ],
            dtype=torch.long,
        ).to(self.device)
        next_next_mino_ids = []
        for exp in experiences:
            next_next_mino_ids_each = []
            for nmid in exp.next_observe[2]:
                for value in get_ordinary_tetris_mino_one_hot(nmid):
                    next_next_mino_ids_each.append(value)
            next_next_mino_ids.append(next_next_mino_ids_each)
        next_next_mino_ids = torch.tensor(
            next_next_mino_ids,
            dtype=torch.long,
        ).to(self.device)
        next_hold_mino_id = torch.tensor(
            [
                get_ordinary_tetris_mino_one_hot(exp.next_observe[3])
                for exp in experiences
            ],
            dtype=torch.long,
        ).to(self.device)

        self.model.train()
        q_values = self.model((board, mino_id, next_mino_ids, hold_mino_id))
        next_q_values = self.target_model(
            (next_board, next_mino_id, next_next_mino_ids, next_hold_mino_id)
        )
        next_q_values_max = torch.max(next_q_values, dim=1).values
        target_values = []
        for i, exp in enumerate(experiences):
            if exp.done:
                target_values.append(exp.reward)
            else:
                target_values.append(
                    exp.reward + self.discount * next_q_values_max[i].item()
                )
        target_values = torch.tensor(target_values, dtype=torch.float32).to(self.device)
        zeros = torch.zeros(self.action_size, dtype=torch.float32).to(self.device)
        zeros[torch.tensor([exp.action.id for exp in experiences])] = 1
        q_values = q_values - torch.min(q_values)
        q_values = q_values * zeros
        q_values = torch.sum(q_values, dim=1)
        loss = self.loss(q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        self.update_epsilon()

    def update_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
