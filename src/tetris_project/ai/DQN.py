import os
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from gymnasium import Env
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam

WEIGHT_PATH = os.path.join(os.path.dirname(__file__), 'tetris_DQN.h5')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = tf.abs(err) < 1.0
    L2 = 0.5 * tf.square(err)
    L1 = tf.abs(err) - 0.5
    loss = tf.where(cond, L2, L1)
    return tf.reduce_mean(loss)

class ExperienceBuffer:
    def __init__(self, buffer_size=10000):
        # ※ deque は最大長を超えた場合に自動で捨ててくれる
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        # Replay Buffer には (observe, action, reward, next_observe, done) を追加
        self.buffer.append(experience)

    def sample(self, size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def len(self) -> int:
        return len(self.buffer)

class DQN:
    def __init__(self, input_size: int, output_size: int, discount=0.90, epsilon=1.0, epsilon_min=0.0001, epsilon_decay=0.9995) -> None:
        super().__init__()
        self.discount = discount # 割引率
        self.epsilon = epsilon # ε-greedy法 の ε
        self.epsilon_min = epsilon_min # ε-greedy法 の ε の最小値
        self.epsilon_decay = epsilon_decay # ε-greedy法 の ε の減衰率

        self.experience_buffer = ExperienceBuffer()

        # 3層のニューラルネットワーク
        self.model = Sequential([
            Dense(128, input_shape=(input_size,), activation='relu'),
            Dense(64, activation='relu'),
            Dense(output_size, activation='linear')
        ])
        self.optimizer = Adam(learning_rate=0.00001)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
    
    def act(self, state: np.ndarray, possible_actions: list) -> Union[int, Tuple[int, int]]:
        # 状態から最適な行動を選択
        # action_mode = 0 : 0, 1, 2, 3, 4, 5, 6
        # action_mode = 1 : (y, rotate) => 
        # ※ 厳密にやるなら action_mode によって分岐させるべき(?)

        if random.random() < self.epsilon: # ε-greedy法
            return random.choice(possible_actions)
        else:
            rating = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            action = np.argmax(rating)
            return (action % 9, action // 9)
        
    def train(self, env: Env, episodes=1):
        # 統計情報として、エピソードごとの報酬とステップ数を保存
        rewards = []
        steps = 0

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                possible_actions = env.get_possible_actions()
                action = self.act(state, possible_actions) # 行動を選択 (ε-greedy法)

                next_state, reward, done, _, _ = env.step(action) # 行動を実行
                self.experience_buffer.add((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
                steps += 1

            rewards.append(total_reward)

            # ε-greedy によって採取された経験を使って学習
            self.learn()
        return [steps, rewards]
    
    def learn(self, batch_size=32, epochs=1):
        if len(self.experience_buffer.buffer) < batch_size:
            return 

        # 訓練データ
        batch = self.experience_buffer.sample(batch_size)

        # バッチ内の状態に対する予測を一括して計算
        states = np.array([sample[0] for sample in batch])
        targets = self.model.predict(states)
        next_states = np.array([sample[3] for sample in batch])
        next_targets = self.model.predict(next_states)
        print("Buffer length:", self.experience_buffer.len())

        # 最初のデータの argmax とその行動価値関数 Q(s, a) を表示
        action_idx = np.argmax(targets[0])
        action_value = targets[0][action_idx]
        print(f"Action value for the first sample: Action index = {action_idx}, Value = {action_value}")

        for i, (_, action, reward, _, done) in enumerate(batch):
            action = action[0] + action[1] * 9
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.discount * np.max(next_targets[i])

        # 学習
        self.model.fit(states, targets, batch_size=batch_size, epochs=epochs, verbose=0)
        # 学習後に今の推測値を表示
        # targets = self.model.predict(states)
        # action_idx = np.argmax(targets[0])
        # action_value = targets[0][action_idx]
        # print(f"Action value for the first sample after learning: Action index = {action_idx}, Value = {action_value}")

        # 学習させる度に ε を減衰
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self) -> None:
        if Path(WEIGHT_PATH).is_file():
            self.model.save(WEIGHT_PATH)

    def load(self) -> None:
        if Path(WEIGHT_PATH).is_file():
            self.model = load_model(WEIGHT_PATH)
