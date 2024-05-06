import os
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env

from tetris_gym import Action
from tetris_gym.tetris import LINE_CLEAR_SCORE
from tetris_project.controller import Controller

WEIGHT_OUT_PATH = os.path.join(os.path.dirname(__file__), "out.pth")

def lines_cleared(score):
    if score >= 800:
        return 4
    elif score >= 500:
        return 3
    elif score >= 300:
        return 2
    elif score >= 100:
        return 1
    else:
        return 0

class ExperienceBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
        self.data_line_cnt = [0, 0, 0, 0, 0]

    def add(self, experience):
        # Replay Buffer には (observe, action, reward, next_observe, done, clear_num) を追加
        while len(self.buffer) >= self.buffer.maxlen:
            # 先頭を取り出す
            # _, _, reward, _, _, clear_num = self.buffer.popleft()
            observe, action, reward, next_observe, done, clear_num = self.buffer.popleft()
            self.data_line_cnt[lines_cleared(reward)] -= 1

            if clear_num > 0:
                self.buffer.append((observe, action, reward, next_observe, done, clear_num-1))
                self.data_line_cnt[lines_cleared(reward)] += 1
        self.buffer.append(experience)
        self.data_line_cnt[lines_cleared(experience[2])] += 1

    def sample(
        self, size: int
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]

    def len(self) -> int:
        return len(self.buffer)


class NN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def save(self) -> None:
        torch.save(self.state_dict(), WEIGHT_OUT_PATH)

    def load(self, path: str) -> None:
        path = os.path.join(os.path.dirname(__file__), path)
        if Path(path).is_file():
            self.load_state_dict(torch.load(path))


class NNTrainerController(Controller):
    def __init__(
        self,
        actions: set[Action],
        model: nn.Module,
        discount=0.95,
        epsilon=0.50,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        device="cpu",
    ) -> None:
        super().__init__(actions)
        self.model = model
        self.discount = discount  # 割引率
        self.epsilon = epsilon  # ε-greedy法 の ε
        self.epsilon_min = epsilon_min  # ε-greedy法 の ε の最小値
        self.epsilon_decay = epsilon_decay  # ε-greedy法 の ε の減衰率

        # Experience Replay Buffer (上下 2 つ)
        self.lower_experience_buffer = ExperienceBuffer()
        self.upper_experience_buffer = ExperienceBuffer()

        self.device = device

    def get_action(self, env: Env) -> Action:
        possible_states = self.get_possible_actions(env)
        if random.random() < self.epsilon:  # ε-greedy法
            return random.choice(possible_states)[0]
        else:  # 最適行動
            states = [state for _, state in possible_states]
            states_tensor = torch.tensor(np.array(states)).float().to(self.device)
            rating = self.model(states_tensor)
            action = possible_states[rating.argmax().item()][0]
            return action

    def train(self, env: Env, episodes=1):
        # 統計情報
        rewards = []
        steps = 0

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(env)  # 行動を選択 (ε-greedy法)
                next_state, reward, done, _, info = env.step(action)  # 行動を実行
                if info["is_lower"]:
                    self.lower_experience_buffer.add(
                        (state, action, reward, next_state, done, lines_cleared(reward))
                    )
                else:
                    self.upper_experience_buffer.add(
                        (state, action, reward, next_state, done, lines_cleared(reward))
                    )

                if reward >= LINE_CLEAR_SCORE[4]:  # Line Clear 時
                    print("★★★★★★★★★★ 4 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[3]:
                    print("★★★★★★★★★★ 3 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[2]:
                    print("★★★★★★★★★★ 2 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[1]:
                    print("★★★★★★★★★★ 1 Line Clear! ★★★★★★★★★★")

                state = next_state
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            self.learn()

        return [steps, rewards]

    def learn(self, batch_size=128, epochs=8):
        # 上下合わせて batch_size 個のデータを取得
        if (
            self.lower_experience_buffer.len() < batch_size // 2
            or self.upper_experience_buffer.len() < batch_size - batch_size // 2
        ):
            print("lower experience buffer size: ", self.lower_experience_buffer.len())
            print(
                "upper experience buffer size: ",
                self.upper_experience_buffer.len(),
                "\n",
            )
            return

        # 訓練データ
        lower_batch = self.lower_experience_buffer.sample(batch_size // 2)
        upper_batch = self.upper_experience_buffer.sample(batch_size - batch_size // 2)
        all_batch = lower_batch + upper_batch

        # 現在と次の状態の Q(s, a) を纏めてバッチ処理して効率化
        states = np.array([sample[0] for sample in all_batch])
        next_states = np.array([sample[3] for sample in all_batch])
        cancat_states_tensor = (
            torch.tensor(np.concatenate([states, next_states])).float().to(self.device)
        )
        all_targets = self.model(cancat_states_tensor)

        targets = all_targets[:batch_size]
        next_targets = all_targets[batch_size:]

        # batch 内で最も高い報酬の期待値 Q(s, a) と即時報酬 r を表示
        # idx: 最も高い報酬の期待値のインデックス
        idx = np.argmax([sample[2] for sample in all_batch])
        print(f"Immediate max reward in batch: {all_batch[idx][2]}")
        print(
            f"Action max value for the first sample in batch: {targets[idx].item()}"
        )

        # Q(s, a) の更新
        for i, (_, _, reward, _, done, _) in enumerate(all_batch):
            targets[i] = reward
            if not done:
                targets[i] += self.discount * next_targets[i]

        targets_tensor = torch.tensor(targets).float().to(self.device)

        # 学習
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            states_tensor = torch.tensor(states).float().to(self.device)
            outputs = self.model(states_tensor)
            loss = criterion(outputs, targets_tensor)

            loss.backward()
            optimizer.step()

        # 学習後に再度 batch 内で最も高い報酬の期待値 Q(s, a) を表示 (確認用)
        targets = self.model(torch.tensor(states).float().to(self.device))
        print(
            f"Action max value for the first sample in batch after learning: {targets[idx].item()}"
        )
        # Buffer 内部のデータ内訳
        print("Data line count (lower): ", self.lower_experience_buffer.data_line_cnt)
        print("Data line count (upper): ", self.upper_experience_buffer.data_line_cnt)
        print("\n")

        # 学習させる度に ε を減衰
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class NNPlayerController(Controller):
    def __init__(self, actions: set[Action], model) -> None:
        super().__init__(actions)
        self.model = model

    def get_action(self, env: Env) -> Action:
        possible_states = self.get_possible_actions(env)
        # 状態から最適行動を選択
        states = [state for _, state in possible_states]
        rating = self.model(torch.tensor(np.array(states)).float())
        action = possible_states[rating.argmax().item()][0]
        return action
