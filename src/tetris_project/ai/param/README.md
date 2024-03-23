# parametor 一覧

## NN0

- Hold 無し
- 初めて成功した時の良い parametor。
- $Dense(128) \rightarrow Dense(64) \rightarrow Dense(output\_size)$
- 損失関数 : Huber Loss
- $\epsilon\_{start} = 1.0, \ discount = 0.90, \ \epsilon\_{min} = 0.00001, \ \epsilon_{decay} = 0.995$
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
- $Dense(128) \rightarrow Dense(64) \rightarrow Dense(output\_size)$
- 損失関数 : Huber Loss
- $\epsilon\_{start} = 1.0, \ discount = 0.90, \ \epsilon\_{min} = 0.00001, \ \epsilon_{decay} = 0.995$
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
