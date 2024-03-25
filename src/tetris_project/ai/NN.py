import os
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gymnasium import Env
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam

from tetris_gym import Action
from tetris_gym.tetris import LINE_CLEAR_SCORE
from tetris_project.controller import Controller

WEIGHT_OUT_PATH = os.path.join(os.path.dirname(__file__), "out.weights.h5")


class ExperienceBuffer:
    def __init__(self, buffer_size=20000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        # Replay Buffer には (observe, action, reward, next_observe, done) を追加
        self.buffer.append(experience)

    def sample(
        self, size: int
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]

    def len(self) -> int:
        return len(self.buffer)


class NN:
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        # 4 層の Neural Network
        self.model = Sequential(
            [
                Input(shape=(input_size,)),  # 入力層の定義
                Dense(64, activation="relu"),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(output_size, activation="linear"),
            ]
        )
        self.optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss="mse", optimizer="adam", metrics=["mean_squared_error"])

    def save(self) -> None:
        if Path(WEIGHT_OUT_PATH).is_file():
            self.model.save_weights(WEIGHT_OUT_PATH)

    def load(self, path: str) -> None:
        path = os.path.join(os.path.dirname(__file__), path)
        if Path(path).is_file():
            self.model.load_weights(path)


class NNTrainerController(Controller):
    def __init__(
        self,
        actions: set[Action],
        model,
        discount=0.95,
        epsilon=0.50,
        epsilon_min=0.01,
        epsilon_decay=0.995,
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

    def get_action(self, env: Env) -> Action:
        possible_states = self.get_possible_actions(env)
        if random.random() < self.epsilon:  # ε-greedy法
            return random.choice(possible_states)[0]
        else:  # 最適行動
            states = [state for _, state in possible_states]
            rating = self.model.predict(np.array(states), verbose=0)
            action = possible_states[np.argmax(rating)][0]
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
                        (state, action, reward, next_state, done)
                    )
                else:
                    self.upper_experience_buffer.add(
                        (state, action, reward, next_state, done)
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
        all_targets = self.model.predict(
            np.concatenate([states, next_states]), batch_size=(batch_size * 2)
        )

        targets = all_targets[:batch_size]
        next_targets = all_targets[batch_size:]

        # batch 内で最も高い報酬の期待値 Q(s, a) と即時報酬 r を表示
        # idx: 最も高い報酬の期待値のインデックス
        idx = np.argmax([sample[2] for sample in all_batch])
        print(f"Immediate max reward in batch: {all_batch[idx][2]}")
        print(f"Action max value for the first sample in batch: {targets[idx]}")

        # Q(s, a) の更新
        for i, (_, _, reward, _, done) in enumerate(all_batch):
            targets[i] = reward
            if not done:
                targets[i] += self.discount * next_targets[i]

        # 学習
        self.model.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=0)

        # 学習後に再度 batch 内で最も高い報酬の期待値 Q(s, a) を表示 (確認用)
        targets = self.model.predict(states, batch_size=batch_size)
        print(
            f"Action max value for the first sample in batch after learning: {targets[idx]}\n"
        )

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
        rating = self.model.predict(np.array(states), verbose=0)
        action = possible_states[np.argmax(rating)][0]
        return action
