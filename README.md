# Few-shot Human Activity Recognition using Smooth Factorized KAN

## 概要
本研究では、スマートフォンの加速度・角速度センサから得られる特徴量を用いて、歩行・階段昇降・着座などの人間行動を分類する Human Activity Recognition（HAR）を対象に、少ない学習データでも高精度に推定できるモデル設計を検討しました。

特に、Kolmogorov-Arnold Networks（KAN）の「各入力特徴に対して1次元の非線形関数を学習し、それらを線形に混合して表現を作る」構造に着目し、特徴ごとの滑らかな関数学習と、特徴間の混合を分離した因子分解構造を導入しました。

## 背景
スマートフォンセンサを用いたHARでは、被験者ごとの動作の違いやセンサの持ち方の違いにより、学習データとテストデータの分布が変化しやすいという課題があります。また、実環境では十分なラベル付きデータを集めることが難しいため、少数データでも汎化しやすいモデルが求められます。

## 提案手法
本研究では、KANを基盤として以下の工夫を取り入れました。

- 特徴ごとに滑らかな1次元非線形関数を学習
- 特徴ごとの変換と特徴間の混合を分離した因子分解構造
- 少数データ下での過学習を抑えるモデル設計

## 実験設定
UCI HARデータセットを用いて評価を行いました。

- データセット：UCI Human Activity Recognition Dataset
- 入力：スマートフォンの加速度・角速度センサに基づく特徴量
- タスク：歩行、階段昇降、着座などの行動分類
- 評価条件：
  - LOSO（Leave-One-Subject-Out）
  - micro-shot / few-shot setting
- 比較対象：
  - MLP（多層パーセプトロン）

## 結果
被験者間の分布差を考慮したLOSO条件、および各クラスごく少数サンプルのみで学習するmicro-shot条件において、一般的なMLPと比較して性能改善の傾向を確認しました。

この結果から、提案したSmooth Factorized KANは、低データかつ分布シフトが存在する条件下で、HARモデルの汎化性能向上に有効である可能性が示されました。

## 使用技術
- Python
- PyTorch
- Machine Learning
- Human Activity Recognition
- Kolmogorov-Arnold Networks
- Few-shot Learning

## ソースコード
```python
# %% [colab] UCI HAR — LOSO + MICRO-SHOT: MLP vs improved KAN
import os, zipfile, urllib.request, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

# =========================================================
# Config
# =========================================================
SEED = 1337
TEST_SUBJECT = 1
K_PER_CLASS = 1
```
