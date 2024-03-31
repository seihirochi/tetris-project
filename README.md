# tetris-project

深層強化学習を用いた​テトリスBotの作成の試行​

## Setup

```bash
rye sync
```

## Run

```bash
rye run train # cpu
rye run train-cuda # cuda (gpu)
rye run train-mps # metal (gpu)
```

## Simulate

```bash
# if you want to change the model, please edit `WEIGHT_OUT_PATH` in `src/tetris_project/ai/NN.py`
rye run simulate
```

## Test

```bash
rye run test
```
