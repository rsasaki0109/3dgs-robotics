# Dynamic Map Viewer

## コンセプト

Dynamic Map Viewer は、Gaussian Splat の夢景色を
「探索ゲーム」ではなく「住める配信空間」として扱うブラウザ版プロトタイプ。

目標は次の 3 つを 1 つの体験としてつなぐことです。

1. ブラウザで即入れる splat 世界
2. プロフィール画像や告知画像を撮れる写真モード
3. VTuber が「ここに住んでいる体」で配信できる常設ステージ

## なぜブラウザ優先か

- Marble 自体が web sharing を前提にしている
- PlayCanvas / SuperSplat 系はブラウザ配信と相性がよい
- Aras の UnityGaussianSplatting は D3D12 / Metal / Vulkan 前提で、
  Web を主戦場にするには向かない
- Dynamic Map Viewer は重い物理より、空気感・見栄え・回遊性の方が価値が高い

## 体験の柱

### 1. Explore Mode

- splat 世界を見回す
- hotspot を調べる
- `DistortionShard` を拾う
- `Echo Note` を読む

### 2. Photo Mode

- HUD を隠す
- 構図ガイドを表示する
- `1:1` / `4:5` / `9:16` の撮影比率を切り替える
- Dynamic Map Viewer らしい色味フィルタを選ぶ
- PNG を保存する

### 3. Live Mode

- 配信用に安全な UI 配置へ切り替える
- お気に入りの定点カメラに飛ぶ
- 「この世界に住んでいる」前提の話題導線を置く
- OBS などで VTuber アバターを前面合成しやすい背景として使う
- fragment ごとの stream scene を切り替える

## 技術方針

- renderer は browser-native を優先する
- splat は見た目、衝突は mesh / proxy collider で分離する
- 公開用 splat は SuperSplat などで軽量化した `.sog` を基本にする
- Marble 由来 asset は `raw_assets/marble/` に置き、配信用 asset は app 側 `public/` に置く
- VTuber 本体は最初から world 内 3D avatar に入れず、OBS 合成を前提にする

## アセットの流れ

1. Marble で世界を生成する
2. `PLY 2M` または高品質 splat と `Collider Mesh GLB` を export する
3. splat を SuperSplat / SplatTransform 系で軽量化する
4. browser 向けに `.sog` を作る
5. `npm run stage:fragment -- --fragment residency --source-dir /path/to/export --force` か `--splat https://... --collider https://...` で `public/splats/`、`public/colliders/`、manifest、bundle catalog へ流し込む
6. `npm run validate:studio` で参照切れを確認する

## ブラウザ MVP

### Milestone 1: 見せる

- `apps/dreamwalker-web` を Vite + React + PlayCanvas React で起動
- 1 つの `.sog` を読み込み表示
- orbit camera で世界の見た目を確認

### Milestone 2: 撮る

- `Photo Mode`
- 比率ガイド
- フィルタ UI
- PNG 保存

### Milestone 3: 住む

- `Live Mode`
- 配信用カメラプリセット
- `Echo Note` を話題導線として設置
- stream-safe overlay
- 3D world 座標に紐づく hotspot overlay
- `Walk Mode` で stage 内を grounded FPS で歩く
- 画面中央 reticle と `F` interact
- interact は中央からのズレと奥行きの両方で判定
- GLB collider 未設定時は proxy floor へ自動フォールバック
- asset manifest で Residency / Echo Chamber の world を別々に差し替えられる
- asset workspace を browser UI から保存して、配信環境ごとの world 差し替えを localStorage で持てる
- asset workspace JSON を paste / file import できると、複数配信環境への持ち回りが楽になる
- stream scene workspace を browser UI から保存できると、配信回ごとの title / topic / memo / branding をコード編集なしで回せる
- studio bundle で asset workspace と scene workspace、それに現在の stage state をまとめて持ち出せると、別マシン移行がかなり楽になる
- studio bundle shelf があると、複数の配信セットをブラウザ内に積んでおける
- bundle/catalog の health が見えると、配信前に demo fallback や参照切れへ気づきやすい
- current fragment 自体の local file を browser で確認できると、その場で asset 配置ミスに気づける

### Milestone 4: 遊ぶ

- shard loop
- gate / scene transfer

現状は browser prototype として、
`DistortionShard を拾う -> DreamGate が開く` の最小状態管理と、
proxy collider / Marble collider mesh のどちらでも歩ける grounded walk、
そして URL hash で切り替わる fragment 遷移と stream scene プリセットまで入っている。

## 最初の hotkey 案

- `P`: Photo Mode 切り替え
- `L`: Live Mode 切り替え
- `X`: Walk Mode 切り替え
- `F`: 中央 reticle の対象を interact
- `Space`: jump
- `1 / 2 / 3`: カメラプリセット
- `4 / 5 / 6`: stream scene
- `G`: 写真ガイド表示
- `K`: PNG 保存

## 実装メモ

- splat は深度や半透明の挙動が不安定なので、強い DOF より色味と FOV で絵作りする
- 歩行判定は raw splat ではなく proxy collider / Marble collider mesh を使う
- physics と collider GLB は Walk Mode まで遅延した方が browser の初回体感がよい
- OBS overlay route は PlayCanvas scene を lazy import しない方が browser source として扱いやすい
- browser stage は外部 font request を持たない方が配信前の立ち上がりが安定する
- fragment 遷移は URL hash と localStorage を使って browser 上で chapter を切り替える
- Live Mode は stream scene ごとに title / topic / camera preset を切り替える
- stream scene workspace があると、title / topic / memo / camera preset / branding override を配信前にブラウザだけで調整できる
- studio bundle があると、world 差し替えと配信 scene 差し替えを 1 回の import で復元できる
- studio bundle shelf があると、配信回ごとの setup をコード編集なしで何本も持てる
- studio bundle を URL で直接読めると、配信前に「今夜のセット」をリンク 1 本で開ける
- public bundle catalog があると、repo 同梱の配信セットを一覧運用できる
- validator があると、`public/splats/` と `public/colliders/` の参照切れを deploy 前に検出できる
- 実 Marble asset の投入は手編集より staging CLI で固定した方が、fragment 差し替えと配信 bundle の運用がぶれにくい
- local export path が無い場合でも remote URL から直接 staging できると、asset の受け渡し方法が増えて運用が止まりにくい
- stream scene は keyboard だけで切り替えられる方が配信運用しやすい
- overlay preset も keyboard で切り替えられる方が配信中の運用に向く
- chapter の変化は overlay branding でも見える方が配信視聴者に伝わりやすい
- stream scene ごとの talk memo を overlay の別 panel に出せると、配信中の話題保持がかなり楽になる
- 現在の stream scene 情報は JSON で copy / download できると OBS 連携しやすい
- `/overlay.html` の軽量 view があると browser source として扱いやすい
- `?relay=1` と小さな SSE relay を併用すると、OBS の別 process でも scene 情報を安定同期できる
- 写真映えスポットは最初から world に設計しておく
- Echo Note は配信のトークテーマにもなる
- 門や shard を「住人の私物」や「記憶の残骸」として扱うと世界観が立つ
