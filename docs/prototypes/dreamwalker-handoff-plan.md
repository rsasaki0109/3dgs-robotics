# gs-sim2real Handoff Plan (Codex 完全引き継ぎ版)

更新日: 2026-04-01

---

## 1. リポジトリ概要

- **GitHub**: https://github.com/rsasaki0109/gs-sim2real (旧 nerf-gs-playground、2026-04-01 リネーム)
- **Python パッケージ**: `gs_sim2real` (`src/gs_sim2real/`)
- **CLI**: `gs-sim2real` (entry point: `src/gs_sim2real/cli.py:main`)
- **ローカルパス**: `/media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground` (ディレクトリ名は旧名のまま、変更不要)
- **browser app**: `apps/dreamwalker-web/` (PlayCanvas + React, Vite 5.4)
- **ROS2**: Jazzy。同マシンで Autoware / LiDAR SLAM (genz_icp, rko_lio) も稼働中
- **テスト**: Playwright 31 tests (smoke.spec.js), pytest 6+ tests (test_robotics_topic_map.py 等)

---

## 2. ビジョン

**実世界撮影 → 3DGS 再構築 → GS world 内ロボット teleop → カメラ画像/depth を ROS2 に配信 → localization/perception テスト**

既存の SplatSim / Splat-Nav / SplatMOVER との差別化:

| 機能 | SplatSim | Splat-Nav | SplatMOVER | gs-sim2real |
|------|----------|-----------|------------|-------------|
| ブラウザ interactive teleop | x | x | x | **o** |
| ROS2 ネイティブ | x | x | x | **o** |
| 撮影→sim ワンコマンド | x | x | x | **o** |
| Depth render | o | implicit | x | P1 で追加 |
| Semantic zones/costmap | x | nav only | x | **o** |
| Autoware 互換 (Jazzy) | x | x | x | **o** |

---

## 3. 現在の完了状況

### 3.1 基盤 (リネーム前から存在)

- **3DGS パイプライン**: COLMAP / pose-free (DUSt3R) / gsplat / nerfstudio → PLY 出力
- **Dynamic Map Viewer web app**: Photo / Live / Walk / Robot モード
  - robot teleop (keyboard WASD / gamepad)
  - front / chase / top カメラ切替
  - waypoint / route / semantic zone / nav overlay
  - Mission Export / Draft Bundle Shelf / Artifact Pack import-export
  - OBS overlay / relay
- **publish pipeline**: validate / publish / release / bundle / discover CLI (全て apps/dreamwalker-web/tools/ 内)

### 3.2 2026-04-01 に追加したもの

#### リネーム
- GitHub repo: `nerf-gs-playground` → `gs-sim2real`
- Python パッケージ: `nerf_gs_playground` → `gs_sim2real`
- CLI コマンド: `gs-playground` → `gs-sim2real`
- 全31ファイルの参照を更新済み
- `pip install --user --break-system-packages -e .` で再インストール済み

#### end-to-end demo コマンド
- `gs-sim2real demo --images <dir>` → COLMAP → 学習 → Dynamic Map Viewer staging → Vite 起動
- `gs-sim2real demo --ply <file>` → 既存 PLY を直接 Dynamic Map Viewer に staging
- staging モジュール: `src/gs_sim2real/demo/stage_for_dreamwalker.py`
  - PLY を `apps/dreamwalker-web/public/splats/<fragment>-main.ply` にコピー
  - `apps/dreamwalker-web/public/manifests/dreamwalker-live.assets.json` の `splatUrl` を更新
- Streamlit app (`app.py`) に Robot Teleop タブ追加
- `scripts/demo.sh` ワンコマンドスクリプト

#### sim2real P0 (ブラウザフレーム配信) — 全4ステップ完了

**Step 1: FrameStreamBridge コンポーネント**
- ファイル: `apps/dreamwalker-web/src/DreamwalkerScene.jsx`
- `parseRobotFrameStreamConfigFromSearch()` (line 25): URL パラメータ `?robotFrameStream=1&robotFrameFps=10` を解析
- `FrameStreamBridge` コンポーネント (line 120): `useAppEvent('postrender')` で canvas.toBlob('image/jpeg', 0.85) キャプチャ
  - FPS スロットル (デフォルト 10Hz)
  - in-flight ガード (二重 toBlob 防止)
  - `onFrame(blob, { timestamp, width, height, fov, pose: { position, orientation } })` コールバック
  - `preserveDrawingBuffer: true` は line 335 で設定済み
- DreamwalkerScene の props に `onFrame` 追加 (line 365)
- `<FrameStreamBridge>` を scene 内に配置 (line 476)

**Step 2: Bridge protocol 拡張**
- ファイル: `apps/dreamwalker-web/src/robotics-bridge.js`
- `buildCameraFrameMessage(blob, metadata)` (line 161): async 関数
  - フォーマット: 4 byte (uint32 LE) JSON header length + JSON header bytes (UTF-8) + JPEG payload bytes
  - Header: `{ type: "camera-frame", timestamp, cameraId, width, height, fov, pose: { position: [x,y,z], orientation: [x,y,z,w] } }`
  - metadata のバリデーション (cameraId 必須、pose 必須、width/height/fov 必須)
  - `normalizeVector()` / `normalizeRequiredNumber()` で型チェック
  - 戻り値: `ArrayBuffer`
- ファイル: `apps/dreamwalker-web/src/App.jsx`
  - `handleRobotFrame` (line 4259): `useCallback` で blob 受信 → `buildCameraFrameMessage` でパック → `robotBridgeSocketRef.current.send(arrayBuffer)`
  - 接続チェック: async 前後で `socket.readyState === WebSocket.OPEN` を2回確認
  - `onFrame={shouldStreamRobotFrames ? handleRobotFrame : undefined}` (line 8263) で DreamwalkerScene に渡す

**Step 3: Rosbridge relay 拡張**
- ファイル: `apps/dreamwalker-web/tools/robotics-rosbridge-relay.mjs`
- topic map に追加 (line 31-32):
  ```
  cameraCompressed: '/dreamwalker/camera/compressed'
  cameraInfo: '/dreamwalker/camera/camera_info'
  ```
- rosbridge advertise (line 407-408):
  - `sensor_msgs/CompressedImage`
  - `sensor_msgs/CameraInfo`
- `parseCameraFrameMessage(payload)` (line 569): binary buffer → header + JPEG 分離
  - 4 byte LE header length → JSON header parse → JPEG payload 抽出
  - header.type === 'camera-frame' 検証
  - width/height/fov/cameraId/timestamp/pose の型チェック
- `toCompressedImage(frame)` (line 641): frame → `sensor_msgs/CompressedImage` ROS msg
  - format: "jpeg", data: base64(JPEG bytes), header.stamp: ROS timestamp
- `toCameraInfo(frame)` (line 649): frame → `sensor_msgs/CameraInfo` ROS msg
  - vertical FOV (degrees) → pinhole intrinsics:
    - `fy = height / (2 * tan(fov_rad / 2))`
    - `fx = fy * (width / height)`
    - `cx = width / 2`, `cy = height / 2`
  - `distortion_model: "plumb_bob"`, `D: [0,0,0,0,0]`
  - `K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]`
  - `R: [1,0,0, 0,1,0, 0,0,1]`
  - `P: [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]`
- bridge server (`robotics-bridge.mjs` line 80) に binary passthrough 追加 (text → JSON、binary → そのまま relay)
- ファイル: `apps/dreamwalker-web/tools/robotics-bridge.mjs`
  - binary frame を relay 先にそのまま転送するハンドラ追加

**Step 4: ROS2 node 拡張**
- ファイル: `src/gs_sim2real/robotics/ros2_bridge_node.py`
  - `--enable-image-relay` フラグ追加 (line 40)
  - `sensor_msgs.msg.CompressedImage` / `CameraInfo` を import (line 81)
  - `_on_camera_compressed()` (line 264): 受信ログ (bytes, format, timestamp, resolution)
  - `_on_camera_info()` (line 275): 受信ログ (timestamp, resolution, frame_id, fx/fy)
  - フラグが有効な時のみ subscriber 作成 (line 172)
- ファイル: `src/gs_sim2real/robotics/topic_map.py`
  - `camera_compressed` / `camera_info` フィールド追加 (line 21-22)
  - `build_ros_topic_map()` で `{root}/camera/compressed`, `{root}/camera/camera_info` 生成 (line 46-47)

#### テスト結果 (P0 完了時点)
- `npx playwright test tests/smoke.spec.js` → **31 passed** (Step 3 で追加されたカメラフレームテスト含む)
- `python3 -m pytest tests/test_robotics_topic_map.py` → **6 passed** (Step 4 で追加された `--enable-image-relay` テスト含む)
- `python3 -m pytest tests/test_cli.py` → **8 passed**
- `npm run build` → 成功 (chunk warning は既存の physics/playcanvas のみ)
- `npm run validate:studio` → 0 error, 3 warning (demo fallback のみ)

### 3.3 未完了

- 実 Marble `.sog` / collider `.glb` / `artifact-pack.json` が無い → mission release できていない (handoff plan の元々の blocker、sim2real とは独立)
- P0 の end-to-end 実機検証 (`ros2 topic echo /dreamwalker/camera/compressed` で実際にフレームが見える確認) は未実施
- P1 以降は未着手

---

## 4. アーキテクチャ図

### 4.1 sim2real フレーム配信パイプライン (P0, 完了)

```
[Browser: PlayCanvas GSplat]
  ↓ robot teleop (WASD/gamepad)
  ↓ FrameStreamBridge: canvas.toBlob(jpeg, 10Hz)
  ↓ handleRobotFrame → buildCameraFrameMessage
  ↓
  ↓ WebSocket binary frame (4B header_len + JSON header + JPEG payload)
  ↓
[robotics-bridge.mjs]  ws://127.0.0.1:8790/robotics
  ↓ binary passthrough
  ↓
[robotics-rosbridge-relay.mjs]
  ↓ parseCameraFrameMessage → toCompressedImage + toCameraInfo
  ↓ rosbridge protocol (JSON over WebSocket)
  ↓
[rosbridge_server]  ws://localhost:9090
  ↓
[ROS2 topics]
  /dreamwalker/camera/compressed    sensor_msgs/CompressedImage
  /dreamwalker/camera/camera_info   sensor_msgs/CameraInfo
  /dreamwalker/robot_pose_stamped   geometry_msgs/PoseStamped  (既存)
  /dreamwalker/robot_pose2d         geometry_msgs/Pose2D       (既存)
```

### 4.2 既存 robotics bridge (テキストフレーム、P0 以前から)

```
[Browser] → robotBridgeSocketRef → ws://127.0.0.1:8790/robotics
  ↕ JSON text frames: robot-state, set-pose, teleop, set-waypoint, set-camera, request-state

[robotics-bridge.mjs] ← CLI client も接続可能

[robotics-rosbridge-relay.mjs]
  → /dreamwalker/robot_state_json        (std_msgs/String)
  → /dreamwalker/robot_pose_stamped      (geometry_msgs/PoseStamped)
  → /dreamwalker/robot_pose2d            (geometry_msgs/Pose2D)
  → /dreamwalker/robot_waypoint          (geometry_msgs/Point)
  → /dreamwalker/robot_route_path        (nav_msgs/Path)
  → /dreamwalker/semantic_zone_summary_json (std_msgs/String)
  → /dreamwalker/current_zone_json       (std_msgs/String)
  → /dreamwalker/semantic_costmap        (nav_msgs/OccupancyGrid)
  ← /dreamwalker/cmd_json               (std_msgs/String)
  ← /dreamwalker/cmd_pose2d             (geometry_msgs/Pose2D)
  ← /dreamwalker/cmd_waypoint           (geometry_msgs/Point)
  ← /dreamwalker/cmd_vel                (geometry_msgs/Twist)
```

---

## 5. 次にやるべきこと (優先順)

### 5.1 P0 end-to-end 実機検証

P0 のコードは全て実装済みだが、実際に ROS2 で動く確認がまだ。以下を実行して動作確認:

```bash
# Terminal 1: Dynamic Map Viewer dev server
cd /media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground/apps/dreamwalker-web
npm run dev

# Terminal 2: robotics bridge server
cd /media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground/apps/dreamwalker-web
node tools/robotics-bridge.mjs

# Terminal 3: rosbridge_server (ROS2)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml

# Terminal 4: rosbridge relay
cd /media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground/apps/dreamwalker-web
node tools/robotics-rosbridge-relay.mjs

# Terminal 5: ブラウザで以下を開く
# http://localhost:5173/?robotFrameStream=1&robotFrameFps=10
# Robot モードに切り替え (R キー)、WASD で動く

# Terminal 6: ROS2 で確認
ros2 topic list | grep dreamwalker
ros2 topic hz /dreamwalker/camera/compressed
ros2 topic echo /dreamwalker/camera/compressed --no-arr

# Terminal 7: ROS2 bridge node (optional)
gs-sim2real robotics-node --enable-image-relay
```

問題が出たらデバッグして修正。問題なければ次へ。

### 5.2 P1: Depth レンダリング

**ゴール**: `/dreamwalker/depth/image` (sensor_msgs/Image, encoding `32FC1`) を配信

**方針**: PlayCanvas で depth render pass を追加

#### Step 1: Depth キャプチャ
- `apps/dreamwalker-web/src/DreamwalkerScene.jsx` に `DepthCaptureBridge` コンポーネント追加
- PlayCanvas の `RenderTarget` を使って depth を offscreen レンダリング
- GSplat shader は depth を既にソート用に計算しているため、depth 出力 pass を追加可能
- 方法案:
  - A: カスタム shader で `gl_FragData` に linear depth を書く
  - B: PlayCanvas の built-in depth texture を利用 (`Layer.renderTarget` + depth buffer readback)
- `readPixels()` で depth texture → Float32Array → Blob
- RGB frame と同じ timestamp で同期

#### Step 2: Protocol 拡張
- `robotics-bridge.js` に `buildDepthFrameMessage(depthBlob, metadata)` 追加
- `robotics-rosbridge-relay.mjs` に `/dreamwalker/depth/image` publish 追加
  - encoding: `32FC1` (32-bit float, 1 channel)
  - step: `width * 4`

#### Step 3: ROS2 node
- `ros2_bridge_node.py` に depth subscriber 追加
- `topic_map.py` に `depth_image` エントリ追加

#### テスト
- Playwright に depth frame テスト追加
- `test_robotics_topic_map.py` に depth topic テスト追加

### 5.3 P2: ヘッドレス gsplat render server

**ゴール**: ブラウザ不要で PLY から直接 RGB + depth をレンダリングし ROS2 配信

- 新モジュール: `src/gs_sim2real/robotics/gsplat_render_server.py`
- gsplat の rasterizer を直接使用 (GPU)
- 学習済み PLY をロード → 任意の pose から RGB + depth レンダリング
- WebSocket or ZMQ で pose query を受信 → レンダリング → ROS2 publish
- CLI: `gs-sim2real sim2real-server --ply model.ply --ros2 --width 640 --height 480 --fps 30`
- ブラウザ版 (P0) より高 FPS (>30) 可能
- CLI サブコマンドを `cli.py` に追加

### 5.4 P3: Localization ベンチマークループ

**ゴール**: GS world でロボットを走らせ、localization アルゴリズムの精度を自動評価

1. 既存 route の waypoint を辿って robot を自動走行
2. 各 waypoint で camera frame を ROS2 配信
3. localization アルゴリズム (ORB-SLAM3, visual place recognition 等) を実行
4. 推定 pose vs ground truth (route の waypoint 座標) を比較
5. ATE / RPE 等のメトリクスを出力
6. CLI: `gs-sim2real benchmark-localization --ply model.ply --route route.json --output results/`

### 5.5 README 更新

sim2real の機能を README に追加。star を狙うなら以下を入れる:

- sim2real のコンセプト図 (アーキテクチャ図)
- GIF/動画: ブラウザで teleop → ROS2 で rviz2 にカメラ画像表示
- Quick Start: `gs-sim2real demo --images ./photos/ && ros2 topic echo /dreamwalker/camera/compressed`
- 競合比較テーブル (SplatSim / Splat-Nav 等)
- 参考論文リンク

### 5.6 未コミットの変更をコミット

現在 git status は大量の変更がある (リネーム + demo + P0)。まとめてコミットするか、論理的に分けてコミットするかはオーナー判断。

---

## 6. 主要ファイルマップ

### 6.1 Python (CLI / training / robotics)

| ファイル | 役割 | 備考 |
|---------|------|------|
| `src/gs_sim2real/cli.py` | CLI エントリポイント | demo / train / preprocess / robotics-node 等 |
| `src/gs_sim2real/demo/__init__.py` | demo パッケージ | 空 |
| `src/gs_sim2real/demo/stage_for_dreamwalker.py` | PLY → Dynamic Map Viewer staging | PLY コピー + manifest JSON 更新 |
| `src/gs_sim2real/robotics/ros2_bridge_node.py` | ROS2 scaffold node | --enable-image-relay で画像 topic subscribe |
| `src/gs_sim2real/robotics/topic_map.py` | ROS2 topic 定義 | camera_compressed, camera_info 含む |
| `src/gs_sim2real/train/gsplat_trainer.py` | gsplat 学習 | → PLY 出力 |
| `src/gs_sim2real/preprocess/colmap.py` | COLMAP 前処理 | |
| `src/gs_sim2real/preprocess/pose_free.py` | DUSt3R pose-free 前処理 | |
| `src/gs_sim2real/benchmark.py` | 学習ベンチマーク | |
| `app.py` | Streamlit UI | 6タブ: Input/Preprocess/Training/Viewer/Export/Teleop |
| `pyproject.toml` | パッケージ定義 | name=gs-sim2real, scripts: gs-sim2real, gs-sim2real-robotics-node |

### 6.2 JavaScript (Dynamic Map Viewer browser app)

| ファイル | 役割 | 備考 |
|---------|------|------|
| `apps/dreamwalker-web/src/App.jsx` | メイン UI + モード制御 | ~8300 行。handleRobotFrame (line 4259) |
| `apps/dreamwalker-web/src/DreamwalkerScene.jsx` | PlayCanvas シーン | GSplat, カメラ, physics, FrameStreamBridge (line 120) |
| `apps/dreamwalker-web/src/robotics-bridge.js` | WebSocket bridge client | buildCameraFrameMessage (line 161) |
| `apps/dreamwalker-web/src/WalkRuntime.jsx` | 一人称歩行モード | lazy load |
| `apps/dreamwalker-web/tools/robotics-bridge.mjs` | WebSocket bridge server | ws://127.0.0.1:8790/robotics, binary passthrough (line 80) |
| `apps/dreamwalker-web/tools/robotics-rosbridge-relay.mjs` | rosbridge relay | parseCameraFrameMessage (line 569), toCompressedImage (line 641), toCameraInfo (line 649) |
| `apps/dreamwalker-web/tools/validate-robot-bundle.mjs` | artifact pack validator | |
| `apps/dreamwalker-web/tools/publish-robot-mission.mjs` | mission publish | |
| `apps/dreamwalker-web/tools/release-robot-mission.mjs` | mission release | |
| `apps/dreamwalker-web/tools/bundle-robot-mission.mjs` | mission bundler | |
| `apps/dreamwalker-web/tools/discover-robot-bundles.mjs` | bundle discovery | |
| `apps/dreamwalker-web/tests/smoke.spec.js` | Playwright テスト | 31 cases。camera frame テスト (line 2407) |

### 6.3 設定 / マニフェスト

| ファイル | 役割 |
|---------|------|
| `apps/dreamwalker-web/public/manifests/dreamwalker-live.assets.json` | splat / collider asset manifest |
| `apps/dreamwalker-web/vite.config.js` | Vite 設定。`allowedHosts` 設定あり |
| `apps/dreamwalker-web/package.json` | npm scripts: dev, build, validate:studio, discover:robot-bundles 等 |

### 6.4 テスト

| ファイル | 件数 | 内容 |
|---------|------|------|
| `apps/dreamwalker-web/tests/smoke.spec.js` | 31 | 全モード・teleop・route・mission・zone・bridge・camera frame |
| `tests/test_cli.py` | 8 | CLI help テスト |
| `tests/test_robotics_topic_map.py` | 6 | topic map, --enable-image-relay |
| `tests/test_robotics_zones.py` | ? | semantic zone テスト |

### 6.5 ドキュメント

| ファイル | 内容 |
|---------|------|
| `docs/prototypes/dreamwalker-handoff-plan.md` | このファイル |
| `docs/prototypes/dreamwalker-robotics.md` | robotics 設計概要 + 参考リンク + 主要ツールリンク |
| `docs/prototypes/dreamwalker-live.md` | streamer/photo/browser residency 方向 |

---

## 7. 座標系

| 環境 | Up | Handedness | Yaw 定義 |
|------|----|-----------|----|
| PlayCanvas | Y-up | Left-handed | degrees |
| ROS2 | Z-up | Right-handed | radians |

変換関数 (rosbridge relay 内):
- `toRosMapPosition(position)`: PlayCanvas position → ROS position
- `worldYawDegreesToRosYawRadians(degrees)`: PlayCanvas yaw → ROS yaw
- `rosYawRadiansToWorldYawDegrees(radians)`: ROS yaw → PlayCanvas yaw

CameraInfo 変換 (rosbridge relay 内):
- PlayCanvas vertical FOV (degrees) → `fy = height / (2 * tan(fov * PI / 360))`
- `fx = fy * aspectRatio`
- `cx = width / 2`, `cy = height / 2`

---

## 8. 開発環境メモ

### インストール
```bash
pip3 install --user --break-system-packages -e /media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground
cd /media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground/apps/dreamwalker-web && npm install
```

### よく使うコマンド
```bash
# CLI ヘルプ
gs-sim2real --help
gs-sim2real demo --help

# Dynamic Map Viewer 開発
cd apps/dreamwalker-web
npm run dev                          # Vite dev server (port 5173)
npm run build                        # production build
npm run validate:studio              # asset validation
npx playwright test tests/smoke.spec.js  # 全31テスト

# Python テスト
python3 -m pytest tests/test_cli.py -v
python3 -m pytest tests/test_robotics_topic_map.py -v

# robotics bridge
node tools/robotics-bridge.mjs       # bridge server
node tools/robotics-rosbridge-relay.mjs  # rosbridge relay

# ROS2 node
gs-sim2real robotics-node --enable-image-relay

# mission pipeline
npm run discover:robot-bundles -- --validate
npm run bundle:robot-mission -- --mission ./public/robot-missions/residency-window-loop.mission.json --output /tmp/test.artifact-pack.json
npm run validate:robot-bundle -- --bundle /tmp/test.artifact-pack.json --public-root ./public
npm run release:robot-mission -- --bundle /tmp/test.artifact-pack.json --public-root /tmp/public --force --validate
```

### Sandbox の注意 (Codex CLI)
- `codex exec --full-auto` はこの環境で bwrap sandbox が `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` で失敗する
- `codex exec -s danger-full-access` を使うこと
- Codex に投げるプロンプトには「まずファイルを読んでから実装せよ」「npm run build が壊れないこと」を必ず含めること

### cloudflared tunnel (リモートアクセス用)
```bash
/tmp/cloudflared tunnel --url http://localhost:5173
# vite.config.js の allowedHosts にトンネルのホスト名を追加する必要あり
```

---

## 9. 参考リンク

- SplatSim: https://splatsim.github.io/
- Splat-Nav: https://chengine.github.io/splatnav/
- Splat-MOVER: https://splatmover.github.io/
- Splat-Sim: https://cancaries.github.io/Splat-Sim/
- PlayCanvas React: https://github.com/playcanvas/react
- rosbridge_suite: https://github.com/RobotWebTools/rosbridge_suite

---

## 10. Codex への指示テンプレート

P1 以降を Codex に投げるときは以下のテンプレートを使え:

```
gs-sim2real リポジトリの sim2real P{N} Step {M} を実装せよ。

docs/prototypes/dreamwalker-handoff-plan.md を読んで対象セクションを理解せよ。

やること:
{具体的な実装内容を箇条書きで}

まず対象ファイルを読んで既存パターンを理解してから実装せよ。
既存の npm run build が壊れないこと。
既存の npx playwright test tests/smoke.spec.js が壊れないこと。
新機能のテストも追加すること。
```

実行コマンド:
```bash
codex exec -s danger-full-access "上記プロンプト"
```
