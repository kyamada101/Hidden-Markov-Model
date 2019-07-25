# Hidden-Markov-Model
隠れマルコフモデル用コード
モデルは以下
![モデルの図](https://github.com/kyamada101/Hidden-Markov-Model/)

## viterbi.py
viterbiアルゴリズムによる最適パスの計算
観測列が長すぎると確率の値が小さくなりすぎてアンダーフロー

## log_viterbi.py
対数変換版viterbiアルゴリズム

## forward.py
前向きアルゴリズムによる同時確率の計算
※観測列が長すぎると確率の値が小さくなりすぎるのでアンダーフロー

## log_forward.py
対数変換版前向きアルゴリズム
forward.pyを対数版にしただけ。アンダーフローしづらい

## backward.py
後ろ向きアルゴリズムによる同時確率の計算
※観測列が長すぎると確率の値が小さくなりすぎるのでアンダーフロー

## Baum-Welch.py
正解モデルによって生成された観測列から、Baum-Welchアルゴリズムによってパラメータを推定。さらに対数変換版viterbiアルゴリズムで最適パスを計算。
結果を正解率で比較し、テキストファイルに結果をすべて書き込むまでのコード
