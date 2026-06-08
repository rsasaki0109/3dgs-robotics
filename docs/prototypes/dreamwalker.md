# Dynamic Map Viewer プロトタイプメモ

## コンセプト

夢の中で崩れた美しい Gaussian Splat 世界を歩き回り、
歪みの欠片を集めながら、断片化した記憶をつなぐ一人称探索ゲーム。

## 現在の方向性

- Unity 版は高機能プロトタイプの土台として維持する
- 初期の公開体験はブラウザ版 `Dynamic Map Viewer` を優先する
- 写真モードと VTuber 配信空間としての利用を最初から織り込む
- 詳細は `docs/prototypes/dreamwalker-live.md` を参照

## 最初のプレイアブル目標

以下がつながれば、Dynamic Map Viewer の最初の縦切りとして十分です。

1. Marble 由来の 1 シーンを正常表示できる
2. プレイヤーが歩く、走る、跳ぶ、低重力で漂う
3. 3 個の `DistortionShard` を拾える
4. 3 個集めると出口の `DreamGate` が開く
5. 出口に触れると次の dream fragment に遷移する

## 優先実装順

### Milestone 1: 空間を歩ける

- GaussianSplatRenderer の表示確認
- Marble collider mesh による歩行可能面の確立
- プレイヤー開始位置の調整
- 低重力浮遊のチューニング

### Milestone 2: 調べる・拾う

- 画面中央 raycast ベースの `Interact` システム
- `IInteractable` インターフェース
- `DistortionShard` の取得演出
- 簡易 HUD

### Milestone 3: 進行ループ

- `DreamStateManager`
- 収集数カウント
- `DreamGate` の解放条件
- シーン遷移または同シーン内ワープ

### Milestone 4: Dynamic Map Viewer らしさ

- 視界歪みポストプロセス
- 環境音の位置トリガー
- 近づくと世界が変形するホットスポット
- テキスト断片や声の残響
- `DreamGate` のフェード遷移
- `Echo Note` による断片的な物語提示

## 技術方針

- splat は見た目専用
- 物理判定は collider mesh / proxy collider を使用
- interaction 判定もまずは collider mesh ベースで進める
- 透明オブジェクトとの深度相性は悪いので、重要ギミックは不透明寄りに設計する

## 次に作る具体物

- `SplatInteractProbe`
- `IInteractable`
- `DistortionShard`
- `DreamGate`
- `DynamicMapViewerHUD`
- `DreamScreenFader`
- `DreamEchoNote`
- `DreamStateManager`
