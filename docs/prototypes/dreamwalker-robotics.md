# Dynamic Map Viewer Robotics

## ねらい

Dynamic Map Viewer の Gaussian Splat world を、
配信・写真・探索だけでなく
`robotics simulation / teleoperation / perception sandbox`
としても使えるようにする。

ただし方針は明確に分ける。

- splat は `見た目` と `知覚入力`
- 物理は `collider mesh / proxy mesh / rigid body`
- ロボット本体や可動物は `通常メッシュ / URDF / 独立 physics`

つまり、`Gaussian Splat をそのまま物理 world にしない`。

## 何が良いか

- Marble 由来の実世界っぽい scene をすぐ browser で使える
- RGB ベースの perception / teleop / UI 実験と相性が良い
- Dynamic Map Viewer の `fragment / bundle / asset staging` をそのまま使い回せる
- 同じ world を
  - `Live / Photo / Explore`
  - `Robot Sandbox / Teleop / Eval`
 で共用できる

## 向いている用途

- mobile robot の視点確認
- teleop UI の試作
- camera placement / blind spot 確認
- waypoint / route 可視化
- traversability overlay
- RGB ベース policy の qualitative check
- closed-loop demo / product mock

## 向いていない用途

- splat だけで正確な接触物理を取ること
- CAD level の正確な衝突判定
- high-speed dynamics の厳密検証
- manipulation の接触計算を splat だけで完結させること

## リポジトリ方針

同じ asset pipeline を共有し、runtime は分ける。

```text
raw_assets/marble/
  -> SuperSplat / export
  -> apps/dreamwalker-web/public/splats/
  -> apps/dreamwalker-web/public/colliders/

apps/dreamwalker-web/
  -> Live / Photo / Explore / Walk

apps/dreamwalker-robotics-web/   # 将来
  -> Robot Sandbox / Teleop / Eval
```

最初は別 app を増やさず、
`Dynamic Map Viewer` に `robotics mode` を足すだけでもよい。
現時点ではこの最小モードを先に入れていて、
`robot base teleop / waypoint / route polyline / front-chase-top camera panel`
までは `apps/dreamwalker-web` 側で動かす。
ただし長期的には browser runtime を分けた方が保守しやすい。

## 最小構成

### 1. Shared world

- `stage:fragment` で Marble asset を staging
- `validate:studio` で参照切れ確認
- `Residency / Echo Chamber` を robot sandbox にも流用

### 2. Robot rig

- `base footprint`
- `heading arrow`
- `spawn pose`
- `teleop state`
- `camera rig`

最初は `URDF` まで行かず、capsule / box で十分。

### 3. Sensor view

- robot front camera
- top-down debug camera
- third-person chase camera
- optional depth-like debug overlay

### 4. Navigation layer

- collider mesh から walkable / non-walkable を分離
- waypoint marker
- local goal
- route polyline
- no-go zone / semantic zone

### 5. Runtime bridge

- websocket で pose / twist / goal を送受信
- 後で ROS2 bridge に拡張
- overlay は現行の broadcast overlay とは別 panel にする

## 実装優先順位

1. `robotics mode` route を追加
2. `robot base gizmo + spawn pose`
3. `teleop` と `waypoint`
4. `robot camera view`
5. `websocket bridge`
6. `traversability overlay`
7. 必要なら `URDF` / `ROS2`

上の 1〜4 は、いまの `Dynamic Map Viewer` に最小実装済み。
5 の websocket bridge も最小版を入れていて、browser から `robot-state` を publish し、
外部 client から `set-pose / teleop / set-waypoint` などを送れる。
いまは `dreamwalker-robotics/v1` を正規 protocol として固定し、CLI client からも叩ける。
さらに `rosbridge_suite` 互換 relay を足して、`Pose2D / Point / String(JSON)` に加えて
`PoseStamped / nav_msgs/Path` topic とも接続できる。
加えて repo 内に `gs-sim2real robotics-node` の ROS2-side scaffold を置き、
relay topic をそのまま subscribe して要約ログや startup command を出せるようにする。
さらに `semantic zone / costmap` も JSON config から生成して publish できる。

## MVP

ここまでできれば十分に価値がある。

- Residency world に robot を spawn
- keyboard / gamepad で teleop
- current pose を UI 表示
- front camera / top view を切替
- collider mesh 上を安全に動く
- waypoints を置ける
- scene bundle で「配信モード」と「robotics モード」を切り替えられる

現状は `keyboard/gamepad teleop + pose UI + front/chase/top camera + waypoint + route polyline + route JSON export/import + route replay shelf + public route preset / catalog + public mission manifest / catalog + route の world-aware preset compatibility + mission の route/zone/world compatibility badge + mission zoneMapUrl による current zone source override + mission-native launchUrl (?robotMission=...) + mission startup state (camera preset / robot camera / stream scene / mode) + browser 側 Mission Export panel + current draft / snapshot 両方の mission preflight health badge + current draft / snapshot 両方の preflight summary copy/download + current draft / snapshot 両方の publish report JSON copy/download + current draft / snapshot 両方の validate command copy/download + current draft / snapshot 両方の release command copy/download + browser 側 publish report preview JSON + browser 側 artifact pack preview JSON + browser 側 validate/release command preview + artifact-pack import 時の embedded preflight/report preview + CLI publish 側の Mission Preflight stdout / optional text output / optional report JSON output + draft bundle/artifact-pack native robot mission publish/release CLI + mission/route/zone から artifact-pack を再生成する bundle CLI + artifact-pack validator CLI + artifact-pack / splat / collider discovery CLI + release CLI の auto-discover bundle pickup + release CLI の local bundle 近傍 auto preflight/report output + preflight comment 付き validate/release/publish command preview + mission metadata editable fields (id / label / description) + mission fragment / route / accent / zone source / world metadata edit + mission route id edit + route label / description / accent edit + publish preflight summary + publish preflight world/frame/zone summary + publish preflight mission metadata summary + publish preflight route/startup summary + publish preflight label/description summary + published mission preview JSON + published mission preview copy/download + current draft からの launch/validate/release/publish/artifact pack copy/download + mission draft bundle / mission artifact pack / validate command / release command / publish command preview + mission draft bundle import preview + artifact-pack import preview + artifact-pack file import to shelf + mission draft bundle shelf + snapshot ごとの metadata edit + startup state edit + fragment / route / accent / zone source / world metadata edit + mission route id edit + route label / description / accent edit + publish preflight summary + publish preflight world/frame/zone summary + publish preflight mission metadata summary + publish preflight route/startup summary + publish preflight label/description summary + draft bundle / raw mission / published preview / launch / validate command / release command / publish command / artifact pack の copy/download 昇格導線 + route catalog validation + mission catalog validation + route publish CLI + route-vs-zone analysis CLI + uncovered/hazard/bounds tuning recommendation + route-to-zone suggestion CLI + hazard/bounds review zone suggestion + zone autotune CLI + robot mission publish/release CLI + semantic zone publish CLI + v1 websocket bridge + CLI client + rosbridge relay + ROS2 scaffold node + semantic zone/costmap + browser 側 current zone overlay + zone workspace editor + nav/cost panel + stage footprint overlay + robot/waypoint quick actions + route/bounds batch ops + studio bundle への semantic zone workspace / robot route 同梱 + stage overlay の depth fade / full contour culling`
まで。次段は `実 robot interface / 実 Marble world 向け zone tuning / route preset の本番導線詰め`。

引き継ぎ用の要約は [dreamwalker-handoff-plan.md](/media/sasaki/aiueo/ai_coding_ws/gs-sim2real/docs/prototypes/dreamwalker-handoff-plan.md) を参照。

## ローカル smoke (ROS なしで render→bridge を貫通)

ROS2 を立てずに render-server と bridge の結線を確認したい場合は
`scripts/robotics_smoke.py` を使う。学習済み gsplat PLY があれば
そのまま食わせられるし、無ければ内蔵 fixture で走る。

```bash
# 内蔵 fixture (5 gauss) でレンダリングのみ確認
python scripts/robotics_smoke.py --fixture --out artifacts/robotics-smoke

# 学習済み PLY を任意の pose から焼く
python scripts/robotics_smoke.py \
    --ply outputs/bag6_demo_train/point_cloud.ply \
    --out artifacts/robotics-smoke
```

出力: `rgb.png` + `depth.npy` + `payload.json` (Dynamic Map Viewer render-query v1
payload と `/dreamwalker/...` relay topic 名がまとめて入る)。
このスクリプトの regression は `tests/test_robotics_smoke_script.py` で
CI に乗っている。

## 重要な設計原則

- Dynamic Map Viewer の価値を壊さない
- robotics 機能は `別モード` として積む
- splat を physics source of truth にしない
- perception realism と interaction realism を分ける
- world asset staging は共通化したまま使う

## 主要ツール / CLI

- browser 本体: [App.jsx](../../apps/dreamwalker-web/src/App.jsx)
- artifact pack validator: [validate-robot-bundle.mjs](../../apps/dreamwalker-web/tools/validate-robot-bundle.mjs)
- mission publish: [publish-robot-mission.mjs](../../apps/dreamwalker-web/tools/publish-robot-mission.mjs)
- mission release: [release-robot-mission.mjs](../../apps/dreamwalker-web/tools/release-robot-mission.mjs)
- mission bundler: [bundle-robot-mission.mjs](../../apps/dreamwalker-web/tools/bundle-robot-mission.mjs)
- handoff plan: [dreamwalker-handoff-plan.md](./dreamwalker-handoff-plan.md)

## 参考

- SplatSim: https://splatsim.github.io/
- Splat-Nav: https://chengine.github.io/splatnav/
- Splat-MOVER: https://splatmover.github.io/
- Splat-Sim: https://cancaries.github.io/Splat-Sim/
