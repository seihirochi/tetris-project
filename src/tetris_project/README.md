# 作成した AI 一覧

## Neural Network

Neural Network を利用してテトリスボードの特徴量を入力情報として、今後の報酬の期待値を推定。

**Input**: Dellacherie's Algorithm

**model**: Dense(128) -> Relu -> Dense(64) -> Relu -> Dense(1)

**optimizer**: Adam

### Experience Buffer

エージェントの過去の行動結果を蓄積しておき、それを利用して学習することで NN の学習を方策 off で行う。

経験の再利用によるデータの効率的利用・学習の安定性向上 (収束時間短縮)・時系列データにおけるノイズの排除 が期待出来る。
