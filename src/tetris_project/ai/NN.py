import os
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from gymnasium import Env
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from tetris_gym import LINE_CLEAR_SCORE, Action, Tetris
from tetris_project.controller import Controller

WEIGHT_OUT_PATH = os.path.join(os.path.dirname(__file__), 'out.weights.h5')

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = tf.abs(err) < 1.0
    L2 = 0.5 * tf.square(err)
    L1 = tf.abs(err) - 0.5
    loss = tf.where(cond, L2, L1)
    return tf.reduce_mean(loss)

class ExperienceBuffer:
    def __init__(self, buffer_size=20000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        # Replay Buffer には (observe, action, reward, next_observe, done) を追加
        self.buffer.append(experience)

    def sample(self, size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def len(self) -> int:
        return len(self.buffer)

class NN:
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        
        # 3層のニューラルネットワーク
        self.model = Sequential([
            Dense(128, input_shape=(input_size,), activation='relu'),
            Dense(64, activation='relu'),
            Dense(output_size, activation='linear')
        ])
        self.optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)    

    def save(self) -> None:
        if Path(WEIGHT_OUT_PATH).is_file():
            self.model.save_weights(WEIGHT_OUT_PATH)

    def load(self, path: str) -> None:
        path = os.path.join(os.path.dirname(__file__), "param", path)
        if Path(path).is_file():
            self.model.load_weights(path)

class NNTrainerController(Controller):
    def __init__(self,
                 actions: set[Action],
                 model,
                 discount=0.95,
                 epsilon=0.50,
                 epsilon_min=0.0001,
                 epsilon_decay=0.999
        ) -> None:
        super().__init__(actions)
        self.model = model
        self.discount = discount # 割引率
        self.epsilon = epsilon # ε-greedy法 の ε
        self.epsilon_min = epsilon_min # ε-greedy法 の ε の最小値
        self.epsilon_decay = epsilon_decay # ε-greedy法 の ε の減衰率
        self.experience_buffer = ExperienceBuffer() # Experience Replay Buffer

    def get_action(self, env: Env) -> Action:
        possible_states = env.unwrapped.get_possible_states()
        # 状態から最適な行動を選択
        if random.random() < self.epsilon: # ε-greedy法
            return random.choice(possible_states)[0]
        else:
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
                possible_states = env.unwrapped.get_possible_states()
                action = self.get_action(env) # 行動を選択 (ε-greedy法)
                next_state, reward, done, _, _ = env.step(action) # 行動を実行
                self.experience_buffer.add((state, action, reward, next_state, done))

                if reward >= LINE_CLEAR_SCORE[4]: # Line Clear 時
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
    
    def learn(self, batch_size=128, epochs=16):
        if len(self.experience_buffer.buffer) < batch_size:
            return

        # 訓練データ
        batch = self.experience_buffer.sample(batch_size)

        # バッチ内の状態に対する予測を一括して計算
        states = np.array([sample[0] for sample in batch])
        targets = self.model.predict(states, batch_size=batch_size)
        next_states = np.array([sample[3] for sample in batch])
        next_targets = self.model.predict(next_states)

        # batch 内で最も高い報酬の期待値 Q(s, a) と即時報酬 r を表示
        # idx: 最も高い報酬の期待値のインデックス
        idx = np.argmax([sample[2] for sample in batch])  # 3番目の要素の中で最大値のインデックスを取得
        print(f"Immediate max reward: {batch[idx][2]}")
        print(f"Action max value for the first sample: {targets[idx]}")

        for i, (_, _, reward, _, done) in enumerate(batch):
            if done:
                targets[i] = reward
            else:
                targets[i] = reward + self.discount * next_targets[i]

        # 学習
        self.model.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=0)

        # 学習後に再度 batch 内で最も高い報酬の期待値 Q(s, a) を表示
        targets = self.model.predict(states, batch_size=batch_size, verbose=0)
        print(f"Action max value for the first sample after learning: {targets[idx]}\n")

        # 学習させる度に ε を減衰
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

class NNPlayerController(Controller):
    def __init__(self, actions: set[Action], model) -> None:
        super().__init__(actions)
        self.model = model

    def get_action(self, env: Env) -> Action:
        possible_states = env.unwrapped.get_possible_states()
        # 状態から最適行動を選択
        states = [state for _, state in possible_states]
        rating = self.model.predict(np.array(states), verbose=0)
        action = possible_states[np.argmax(rating)][0]
        return action
