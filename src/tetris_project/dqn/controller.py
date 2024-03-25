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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 20 * 10, 512)

        # ミノID, HoldミノID, NextミノIDsを扱うEmbedding層
        self.mino_embedding = nn.Embedding(mino_kinds, 1)
        self.hold_mino_embedding = nn.Embedding(mino_kinds, 1)
        self.next_minos_embedding = nn.Embedding(mino_kinds * next_minos_size, 1)

        # 結合層
        self.fc2 = nn.Linear(512 + 7 * (2 + next_minos_size), 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        board, mino_id, next_mino_ids, hold_mino_id = x

        # 盤面情報をCNNに通す
        x = torch.relu(self.conv1(board))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 20 * 10)
        x = torch.relu(self.fc1(x))

        # ミノID, HoldミノID, NextミノIDsをEmbedding層に通す
        mino_id = self.mino_embedding(mino_id)
        hold_mino_id = self.hold_mino_embedding(hold_mino_id)
        # print(next_mino_ids.shape)
        next_mino_ids = self.next_minos_embedding(next_mino_ids)
        # print(x.shape, mino_id.shape, hold_mino_id.shape, next_mino_ids.shape)
        # (32, 512), (32, 7, 1), (32, 7, 1), (32, 21, 3)
        mino_id = mino_id.view(-1, self.mino_kinds)
        hold_mino_id = hold_mino_id.view(-1, self.mino_kinds)
        next_mino_ids = next_mino_ids.view(-1, self.mino_kinds * self.next_minos_size)
        # print(x.shape, mino_id.shape, hold_mino_id.shape, next_mino_ids.shape)
        # Embedding層の出力を結合
        x = torch.cat([x, mino_id, hold_mino_id, next_mino_ids], dim=1)

        # 結合層に通す
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
        self.mino_kinds = mino_kinds

    def train(self, env: Env, episodes: int) -> Tuple[int, List[float]]:
        steps = 0
        rewards = []
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()

                if reward >= LINE_CLEAR_SCORE[4]:  # Line Clear 時
                    print("★★★★★★★★★★ 4 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[3]:
                    print("★★★★★★★★★★ 3 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[2]:
                    print("★★★★★★★★★★ 2 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[1]:
                    print("★★★★★★★★★★ 1 Line Clear! ★★★★★★★★★★")

                total_reward += reward
                steps += 1
            rewards.append(total_reward)
        return steps, rewards

    def get_action(self, state: Tuple[np.ndarray, int, List[int], int]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(list(self.actions))
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
            q_values = self.model((board, mino_id, next_mino_ids, hold_mino_id))
            action_id = torch.argmax(q_values).item()
            return self.action_map[action_id]

    def remember(
        self,
        state: Tuple[np.ndarray, int, np.ndarray, int],
        action: Action,
        reward: float,
        next_state: Tuple[np.ndarray, int, np.ndarray, int],
        done: bool,
    ) -> None:
        self.experience_buffer.add(Experience(state, action, reward, next_state, done))

    def replay(self) -> None:
        if self.experience_buffer.len() < 32:
            return

        experiences = self.experience_buffer.sample(32)

        self.optimizer.zero_grad()

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

        q_values = self.model((board, mino_id, next_mino_ids, hold_mino_id))
        q_values = torch.max(q_values, dim=1).values

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

        next_q_values = self.model(
            (next_board, next_mino_id, next_next_mino_ids, next_hold_mino_id)
        )

        targets = []
        for i, exp in enumerate(experiences):
            if exp.done:
                targets.append(exp.reward)
            else:
                targets.append(
                    exp.reward + self.discount * torch.max(next_q_values[i]).item()
                )

        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
        loss = nn.MSELoss()(q_values, targets)
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
