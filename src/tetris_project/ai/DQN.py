import os
import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env

WEIGHT_PATH = os.path.join(os.path.dirname(__file__), 'tetris_DQN.pth')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

class ExperienceBuffer:
    def __init__(self, buffer_size=20000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        # Replay Buffer には (observe, reward, next_observe, done) を追加
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, size: int) -> List[Tuple[np.ndarray, float, np.ndarray, bool]]:
        return random.sample(self.buffer, size)

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, discount=0.99, epsilon=0.5, epsilon_min=0.0001, epsilon_decay=0.9995) -> None:
        super().__init__()
        self.discount = discount # 割引率
        self.epsilon = epsilon # ε-greedy法 の ε
        self.epsilon_min = epsilon_min # ε-greedy法 の ε の最小値
        self.epsilon_decay = epsilon_decay # ε-greedy法 の ε の減衰率
        self.loss_fn = nn.MSELoss() # 損失関数
        self.experience_buffer = ExperienceBuffer()
        self.input_size = input_size
        self.output_size = output_size

        self._create_model()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01) # 最適化手法

    def _create_model(self) -> nn.Module:
        # 3層のニューラルネットワーク
        # Linear(input_size, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, output_size) -> Softmax
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def act(self, state: np.ndarray) -> int:
        # 状態から最適な行動を選択
        # action_mode = 0 : 0, 1, 2, 3, 4, 5, 6
        # action_mode = 1 : (y, rotate) => 
        # ※ 厳密にやるなら action_mode によって分岐させるべき(?)

        if random.random() < self.epsilon: # ε-greedy法
            return random.choice(range(self.output_size))
        
        max_rating = None
        best_action = None

        rating = self._predict_rating(state)

        for action in range(self.output_size):
            if max_rating is None or rating[action] > max_rating:
                max_rating = rating[action]
                best_action = action
        return best_action

    def _predict_rating(self, state: np.ndarray) -> np.ndarray:
        # 状態 -> 行動価値
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            return self.forward(state_tensor).numpy()
        
    def train(self, env: Env, episodes=1):

        # 統計情報として、エピソードごとの報酬とステップ数を保存
        rewards = []
        steps = 0

        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.act(state) # 行動を選択 (ε-greedy法)
                next_state, reward, done, _, _ = env.step(action) # 行動を実行
                self.experience_buffer.add((state, reward, next_state, done))
                state = next_state

                total_reward += reward
                steps += 1

            rewards.append(total_reward)

            # ε-greedy によって採取された経験を使って学習
            self.learn()
        return [steps, rewards]
    
    def learn(self, batch_size=512, epochs=1):
        if len(self.experience_buffer.buffer) < batch_size:
            return # バッチサイズよりも経験が少ない場合は学習しない
        
        for _ in range(epochs):
            # 訓練データ
            train_x = []
            train_y = []
            # リプレイバッファから random sampling
            batch = self.experience_buffer.sample(batch_size)

            # 各 sample から行動価値を計算
            for i, (state, reward, next_state, done) in enumerate(batch):
                q = reward
                rating = self._predict_rating(next_state)
                if not done:
                    q = reward + self.discount * np.max(rating)
                train_x.append(state)
                train_y.append(q)

            # 学習
            train_x = torch.tensor(train_x, dtype=torch.float32)
            train_y = torch.tensor(train_y, dtype=torch.float32)
            self.optimizer.zero_grad()  # 勾配をゼロにリセット
            predictions = self.forward(train_x)  # 予測を行う
            loss = self.loss_fn(predictions, train_y.unsqueeze(1))  # 損失を計算
            loss.backward()  # 逆伝播 (nn.Modules のおかげで自動で勾配計算してくれる)
            self.optimizer.step()  # パラメータの更新
        
        # 学習させる度に ε を減衰
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self) -> None:
        if Path(WEIGHT_PATH).is_file():
            torch.save(self.state_dict(), WEIGHT_PATH)

    def load(self) -> None:
        if Path(WEIGHT_PATH).is_file():
            self.load_state_dict(torch.load(WEIGHT_PATH))
