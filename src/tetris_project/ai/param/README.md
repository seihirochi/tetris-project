# parametor 一覧

## NN0

- Hold 無し
- 初めて成功した時の良い parametor。
- $Dense(128) \rightarrow Dense(64) \rightarrow Dense(output\_size)$
- 損失関数 : Huber Loss
- $\epsilon\_{start} = 1.0, \ discount = 0.90, \ \epsilon\_{min} = 0.00001, \ \epsilon_{decay} = 0.995$
- Experience Buffer を利用。エージェントの過去の行動結果を蓄積しておき、それを利用して学習することで NN の学習を方策 off で行う。経験の再利用によるデータの効率的利用・学習の安定性向上 (収束時間短縮)・時系列データにおけるノイズの排除が期待出来る。
- input は以下

```python
def observe(self) -> np.ndarray:
    return np.concatenate([
        [
            self.line_total_count,
            self.get_hole_count(),
            self.get_latest_clear_mino_heght(),
            self.get_row_transitions(),
            self.get_column_transitions(),
            self.get_bumpiness(),
            self.get_eroded_piece_cells(),
            self.get_cumulative_wells(),
            self.get_aggregate_height(),
        ],
        self.current_mino_state.mino.to_tensor().flatten(),
        np.concatenate(
            [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
        ),
    ])
```

## NN1

- Hold 機能追加。
- mino の動きのバグを修正した時の parametor。( 移動候補を全列挙出来ていないバグ )
- Model, parametor は NN0 と同じ
- input は以下

```python
def observe(self) -> np.ndarray:
    return np.concatenate([
        [
            self.line_total_count,
            self.get_hole_count(),
            self.get_latest_clear_mino_heght(),
            self.get_row_transitions(),
            self.get_column_transitions(),
            self.get_bumpiness(),
            self.get_eroded_piece_cells(),
            self.get_cumulative_wells(),
            self.get_aggregate_height(),
        ],
        self.current_mino_state.mino.to_tensor().flatten(),
        np.concatenate(
            [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
        ),
        self.hold_mino.mino.to_tensor().flatten(),
    ])
```

## NN2

- mino の動きのバグを修正した時の parametor。( 移動候補を全列挙出来ていないバグ )
- $Dense(64) \rightarrow Dense(64) \rightarrow Dense(32) \rightarrow Dense(output\_size)$
- 損失関数 : Mean Squared Error
- $\epsilon\_{start} = 1.0, \ discount = 1.00, \ \epsilon\_{min} = 0.001, \ \epsilon_{decay} = 0.995$
- parametor の数は NN1 から減っているが、層を深くすることでより非線形な表現を可能にした。
- input NN1 と同じ

## NN3

- 盤面特徴量のバグ修正 & 新特徴量の追加
- Model, parametor は NN2 と同じ
- input は以下

```python
def observe(self) -> np.ndarray:
    return np.concatenate([
        [
            self.get_hole_count(),
            self.get_above_block_squared_sum(),
            self.get_latest_clear_mino_heght(),
            self.get_row_transitions(),
            self.get_column_transitions(),
            self.get_bumpiness(),
            self.get_eroded_piece_cells(),
            self.get_cumulative_wells(),
            self.get_aggregate_height(),
        ],
        self.current_mino_state.mino.to_tensor().flatten(),
        np.concatenate(
            [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
        ),
        self.hold_mino.mino.to_tensor().flatten(),
    ])
```

## NN4

- NN3 だとボード上部だけで完結しようとする $\rightarrow$ Experience Buffer を前半・後半で 2 つ用意して偏りを減らす
- Model, parametor, input は NN2 と同じ
