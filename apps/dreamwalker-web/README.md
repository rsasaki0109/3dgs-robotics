# DreamWalker Web

DreamWalker の browser-first 実験アプリです。

現在は `DreamWalker Live` の最小土台として、
PlayCanvas React 上で splat 表示、写真モード UI、配信用 UI、
そして `Echo Note hotspot` の枠組みを用意しています。

## セットアップ

```bash
cd apps/dreamwalker-web
npm install
npm run dev
npm run validate:studio
```

開発サーバーは通常 `http://localhost:5173` で起動します。

OBS 向けの relay 同期を試す場合は、別ターミナルでこれも起動します。

```bash
npm run relay
npm run robotics:bridge
npm run robotics:rosbridge -- --rosbridge ws://127.0.0.1:9090
```

## splat の置き方

1. Marble から Gaussian splat を export
2. SuperSplat などで browser 向けに軽量化
3. `.sog` と `Collider Mesh GLB` を source directory にまとめる
4. `npm run stage:fragment -- --fragment residency --source-dir /path/to/export --force`
5. script が `public/splats/`、`public/colliders/`、`public/manifests/`、`public/studio-bundles/` を更新する
6. `npm run validate:studio` で health を確認する

最短コマンド:

```bash
npm run stage:fragment -- \
  --fragment residency \
  --source-dir /absolute/path/to/residency-export \
  --bundle-id residency-pilot \
  --bundle-label "Residency Pilot"
```

このコマンドは次をまとめて行います。

- source dir から `.sog` と `collider GLB` を auto-detect
- `public/splats/` と `public/colliders/` へ copy
- `public/manifests/dreamwalker-live.assets.json` の target fragment を更新
- `public/studio-bundles/<bundle-id>.json` を生成
- `public/studio-bundles/index.json` に catalog entry を追加
- `/?studioBundle=/studio-bundles/<bundle-id>.json` の launch URL を出力

source dir の auto-detect が合わない場合は、明示指定できます。

```bash
npm run stage:fragment -- \
  --fragment echo-chamber \
  --splat /absolute/path/to/echo-main.sog \
  --collider /absolute/path/to/echo-collider.glb \
  --bundle-id echo-night
```

remote URL を直接渡すこともできます。

```bash
npm run stage:fragment -- \
  --fragment residency \
  --splat https://example.com/residency-main.sog \
  --collider https://example.com/residency-collider.glb \
  --bundle-id residency-remote \
  --force
```

事前確認だけしたい時は `--dry-run` を使います。

```bash
npm run stage:fragment -- \
  --fragment residency \
  --source-dir /absolute/path/to/residency-export \
  --dry-run
```

例:

```json
{
  "fragments": {
    "residency": {
      "splatUrl": "/splats/residency-main.sog",
      "colliderMeshUrl": "/colliders/residency-main-collider.glb"
    },
    "echo-chamber": {
      "splatUrl": "/splats/echo-chamber-main.sog",
      "colliderMeshUrl": "/colliders/echo-chamber-main-collider.glb"
    }
  }
}
```

別 manifest を使いたい場合は query で差し替えられます。

```text
/?assetManifest=/manifests/custom-world.json
```

studio bundle も query で直接読めます。

```text
/?studioBundle=/studio-bundles/dreamwalker-live.sample.json
```

public catalog を差し替えたい場合はこれです。

```text
/?studioBundleCatalog=/studio-bundles/index.json
```

large-scale 3DGS tile catalog を開く場合はこれです。左パネルの
`Open Large-scale Demo` からも同じ URL に入れます。

```text
/?tileCatalog=/manifests/outdoor-production-grid-large-tile-catalog.json&tilePreload=metadata&tilePreloadLimit=4&tileResidentLimit=6&robotRoute=/robot-routes/outdoor-production-grid-large-route.json&robotRoutePlayback=1&robotRoutePlaybackMs=1200&robotRoutePlaybackLoop=1
```

## 今の状態

- PlayCanvas React の最小 scene
- browser overlay
- `Photo Mode`
- `Live Mode`
- fragment ごとの `stream scene` プリセット
- `4 / 5 / 6` による stream scene 切替
- `7 / 8 / 9` による overlay preset 切替
- Live scene JSON の `コピー / ダウンロード`
- `/overlay.html` の専用 overlay view
- `?relay=1` による SSE relay publish
- `/overlay.html?relay=1` による OBS 向け overlay sync
- `Lower Third / Side Stack / Headline` の overlay preset
- fragment ごとの overlay branding 切替
- stream scene ごとの overlay badge / strapline 上書き
- stream scene ごとの overlay memo panel
- browser UI からの `Scene Workspace` 編集 / 保存 / reset
- `Scene Workspace JSON` の copy / download / paste import / file import
- `Studio Bundle` による asset + scene + stage state の一括 export / import
- `Studio Bundle Shelf` による複数配信セットの local 保存 / 再適用
- `public/manifests/dreamwalker-live.assets.json` による fragment ごとの asset 差し替え
- `?assetManifest=` による world manifest 切替
- `npm run stage:fragment -- --fragment <id> --source-dir <dir>` による実 Marble asset の staging
- browser UI からの `Asset Workspace` 編集 / 保存 / reset
- `World Health` と `Public Studio Bundles` の health badge
- `Walk Mode` (`X`) with grounded FPS + jump
- `Robot Mode` (`R`) with robot base teleop + waypoint sandbox
- robot trail / goal line overlay
- fragment ごとの semantic zone map load と current zone overlay
- Robot Mode 右パネルからの `Semantic Zone Workspace` 編集 / 保存 / reset
- semantic zone から描く `Nav / Cost Overlay` パネル
- Robot Mode stage 上の `zone footprint / cost overlay`
- zone editor の `Zone <- Robot / Waypoint` と `Robot -> Zone` quick action
- `Add Zones From Route / Fit Bounds To Zones / Clear All Zones` の batch ops
- `?robotBridge=1` による websocket robot bridge
- gamepad teleop
- `dreamwalker-robotics/v1` protocol と CLI client
- 中央 reticle による `F` interact
- reticle は `screen distance + depth` ベースで近距離対象だけ有効化
- Marble `Collider Mesh GLB` の optional 読み込み
- Walk Mode に入るまで physics / collider GLB を遅延
- `/overlay.html` は PlayCanvas scene を import しない軽量 route
- main / overlay ともに外部 web font を使わず、初回 request を抑えています
- proxy collider を使った床判定
- `Front / Chase / Top View` の robot camera 切替
- カメラプリセット UI
- world 座標 hotspot の DOM overlay 投影
- `DistortionShard` 収集
- `DreamGate` による fragment 遷移
- `Echo Note` モーダル
- PNG snapshot 保存
- URL hash による fragment 復元
- localStorage による fragment ごとの収集状態保持

## Robot mode

- `R` で `Robot Mode` に入ると、main stage が robot camera を追従します
- `W / A / S / D` か矢印キーで robot base を前後移動 / 旋回できます
- `V` で robot の前方に waypoint を置き、`Reset Robot Pose` で spawn pose に戻せます
- `C` で route を現在位置から引き直せます
- `?robotBridge=1` か `?robotBridgeUrl=ws://host:port/robotics` で robot bridge へ接続できます
- gamepad があれば left stick で前後/旋回、`A` waypoint、`X` clear waypoint、`B` clear route、`Y` reset pose、`LB/RB` で camera cycle が使えます
- right panel には `Pose / Heading / Front Camera Panel` を出し、`Front / Chase / Top View` を切り替えられます
- `Route Export` panel から current pose / waypoint / route を `Copy / Download / Import` でき、route JSON には `fragment / asset / frame` の `world` metadata も入ります
- `Save Route Snapshot` で route shelf に積み、reload 後でも `Apply / Download / Delete` できます
- `?robotRoute=/robot-routes/residency-window-loop.json` で public route preset を直接起動できます
- `?robotMission=/robot-missions/residency-window-loop.mission.json` で route + zone metadata を束ねた mission manifest を直接起動できます
- `public/robot-missions/index.json` に mission を並べると、左パネルの `Public Robot Missions` から `Apply / Launch / Copy URL` できます
- mission manifest の `zoneMapUrl` は `Semantic Zone Workspace` が無い場合、fragment 既定の zone map を上書きして current zone 判定へ反映されます
- mission manifest には `cameraPresetId / robotCameraId / streamSceneId / startupMode` も持てて、`Apply` や `?robotMission=` 起動時に robot camera と stage preset を合わせて再現できます
- Robot Mode の `Mission Export` panel から current route / zone source / camera preset / robot camera / stream scene を含む mission JSON をその場で `Copy / Download` できます
- 同じ panel で `Mission ID / Mission Label / Mission Description` を直接編集でき、draft bundle / publish command / mission JSON がその場で追従します
- 同じ panel で `Mission Fragment ID / Mission Fragment Label` も編集でき、draft bundle filename と publish preview の target fragment を browser 上で切り替えられます
- 同じ panel で `Mission Route ID` も編集でき、`/robot-routes/<id>.json` を自動で組み立てて route publish file 名を固定できます
- 同じ panel で `Mission Route URL` も編集でき、published preview と `publish:robot-mission --bundle` が使う route preset file を browser 上で固定できます
- `Published Mission Preview JSON` の下には `preflight fragment / route id / mission id / route file` も出るので、publish target を JSON を開かずに確認できます
- 同じ preflight には `world asset / frame / zone map` も出るので、route/world の組み合わせミスを panel 上で先に見つけられます
- 同じ preflight には `mission label / description / fragment label` も出るので、manifest 名と説明文の publish ミスも panel 上で見つけられます
- 同じ preflight には `route meta / accent / startup / preset / robot camera / scene` も出るので、route catalog の見た目と起動構図のズレも publish 前に読めます
- さらに `route description / preset label / robot camera label / scene label` も出るので、ID だけでなく人間向けの表示名でも preflight を確認できます
- `Mission Export` の header には `Mission Ready / Mission Warning` も出るので、current draft の fragment / route / zone / world のズレを publish 前にすぐ見られます
- 同じ panel で `Route Label / Route Description / Route Accent` も編集でき、`Robot Route JSON` と draft bundle に route catalog 用 metadata を持たせられます
- 同じ panel で `Mission Accent / Mission Zone Map URL` も編集でき、manifest の見た目と mission 専用 zone source を browser 上で先に詰められます
- 同じ panel で `Mission World Asset Label / Mission World Frame ID` も編集でき、publish preview と本番 manifest の world metadata を route preset から上書きできます
- `Published Mission Preview JSON` は `publish:robot-mission --bundle` を default option で流した時の public mission manifest を browser 上で先に確認するための preview です
- この preview は staged route preset の URL まで含めて解決するので、draft の `mission.routeUrl` ではなく `residency-route-snapshot` のような publish 後 URL が出ることがあります
- preview は `Copy Published Preview / Download Published Preview` でそのまま持ち出せて、`published file <mission-id>.mission.json` で最終ファイル名も見えます
- 同じ panel から `Copy Preflight / Download Preflight` も使えるので、publish target の health / world / route / startup を `.preflight.txt` として残せます
- 同じ panel から `Copy Publish Report / Download Publish Report` も使えるので、CLI 側 `--report-output` と同じ schema の publish report JSON を current draft から直接持ち出せます
- 同じ panel から `Copy Validate / Download Validate` も使えるので、`validate:robot-bundle` の artifact-pack preflight command を `.validate-command.txt` として固定できます
- 同じ panel から `Copy Release / Download Release` も使えるので、`release:robot-mission` 用の `.release-command.txt` も current draft から直接持ち出せます
- `Mission Release Command` には auto output comment も入るので、shell 実行前に `.preflight.txt / .publish-report.json` の保存先まで確認できます
- `Publish Report JSON` textarea もあるので、copy/download 前に browser 上で publish report schema と world/route/startup の解決結果をそのまま確認できます
- 同じ panel から `Copy Launch / Download Launch / Download Publish Command / Copy Artifact Pack / Download Artifact Pack` も使えるので、saved snapshot を作る前でも launch URL・publish command・artifact pack を current draft から直接持ち出せます
- 同じ panel で `Mission Draft Bundle JSON`、`Mission Artifact Pack JSON`、`Mission Validate Command`、`Mission Release Command`、`Mission Publish Command` も見られるので、browser 上で整えた mission / route / zones をそのまま `validate:robot-bundle`、release script、`publish:robot-mission -- --bundle /absolute/path/to/...artifact-pack.json` へ渡せます
- `Mission Publish Command` には `preflight / world / zone / launch` の comment も入るので、shell 実行前に publish target をテキストのまま確認できます
- `Mission Validate Command` にも同じ `preflight / world / zone / launch` comment が入るので、publish 前の artifact-pack 単体チェックを shell 上でも読みやすく回せます
- `Mission Draft Bundle Import` に pasted JSON か `.json` file を入れると、draft bundle だけでなく `artifact-pack` からでも mission / route / zones / startup state を browser 上へ preview 適用できます
- pasted `artifact-pack` には埋め込まれた `preflight-summary` と `publish-report` の preview も出るので、import 前に CLI 由来の検査結果まで読み返せます
- `Apply Pasted Draft Bundle To Shelf` を使うと、artifact-pack の label をそのまま snapshot label にして shelf へ保存できます
- `Import Draft Bundle File To Shelf` を使うと、artifact-pack file でも preview 適用と shelf 保存を一発で回せます
- `Save Draft Snapshot` で mission draft bundle shelf に積めるので、reload 後でも `Apply / Download / Delete` で同じ preview 状態を呼び戻せます
- shelf の各 snapshot には `Copy Bundle / Download Bundle` が付くので、保存した draft bundle 自体を apply 前に再利用できます
- shelf の各 snapshot には `Copy Publish / Download Publish` が付くので、保存した draft をそのまま `publish:robot-mission --bundle` へ昇格できます
- shelf の各 snapshot には `Copy Artifacts / Download Artifacts` も付くので、bundle / mission / preview / launch / publish command を 1 つの artifact pack JSON としてまとめて持ち出せます
- shelf の各 snapshot には `published file` と `launch` も出るので、保存済み draft から public mission 名と `?robotMission=...` をその場で確認できます
- shelf の各 snapshot から `Copy Mission / Download Mission` を使うと、saved draft の raw mission payload を clipboard か `.robot-mission.json` で持ち出せます
- shelf の各 snapshot から `Download Preview` も使えるので、saved draft の public mission manifest preview を apply 前に `.mission.json` として書き出せます
- shelf card でも `Snapshot Label / Snapshot Mission ID / Snapshot Mission Label / Snapshot Mission Description` を直接編集でき、published file と launch preview がその場で更新されます
- shelf card では `Snapshot Mission Fragment ID / Snapshot Mission Fragment Label` も編集でき、`fragment ... / label ...` note で publish 前の target fragment を確認できます
- shelf card では `Snapshot Mission Route ID` も編集でき、`route file ...` note で publish 前の route file 名を確認できます
- shelf card では `Snapshot Mission Route URL` も編集でき、`route ... / zone ...` note で publish 前の route preset と zone source を確認できます
- shelf card には `preflight fragment / route id / mission id` も出るので、saved draft の publish target を apply 前に確認できます
- shelf card の preflight にも `world asset / frame / zone map` が出るので、saved draft ごとの差分確認がしやすくなります
- shelf card の preflight にも `mission label / description / fragment label` が出るので、saved draft の public manifest 名を apply 前に比較できます
- shelf card の preflight にも `route meta / accent / startup / preset / robot camera / scene` が出るので、保存済み draft の起動構図差分を apply 前に確認できます
- shelf card の preflight にも `route description / preset label / robot camera label / scene label` が出るので、保存済み draft の見え方を panel 上だけで比べられます
- shelf card にも `Mission Ready / Mission Warning` が出るので、saved draft の publish target 差分を apply 前に比較できます
- shelf card からも `Copy Preflight / Download Preflight` を使えるので、saved draft ごとの preflight summary を text file として持ち出せます
- shelf card からも `Copy Report / Download Report` を使えるので、saved draft ごとの publish report JSON を CLI に渡す前に固定できます
- shelf card からも `Copy Validate / Download Validate` を使えるので、saved draft ごとの `validate:robot-bundle` command を apply 前に固定できます
- shelf card からも `Copy Release / Download Release` を使えるので、saved draft ごとの validate->publish release command を apply 前に固定できます
- shelf card では `Snapshot Route Label / Snapshot Route Description / Snapshot Route Accent` も編集でき、`route meta ...` note で publish 前の route catalog 見た目を確認できます
- shelf card では `Snapshot Startup Mode / Camera Preset / Robot Camera / Stream Scene` も編集でき、`effective ...` note で apply 時の起動構図を先に確認できます
- shelf card では `Snapshot Mission Accent / Snapshot Zone Map URL` も編集でき、`accent ... / zone ...` note で publish 前の manifest と zone source を確認できます
- shelf card では `Snapshot World Asset Label / Snapshot World Frame ID` も編集でき、`world ... / frame ...` note で publish 前の world metadata を確認できます
- shelf から `Copy Bundle / Download Bundle / Copy Mission / Download Mission / Copy Preview / Download Preview / Copy Launch / Download Launch / Copy Preflight / Download Preflight / Copy Report / Download Report / Copy Validate / Download Validate / Copy Release / Download Release / Copy Publish / Download Publish / Copy Artifacts / Download Artifacts` を使うと、saved draft を apply し直さずに draft bundle・raw mission JSON・preview JSON・launch URL・preflight text・publish report JSON・validate command・release command・publish command・artifact pack を持ち出せます
- `public/robot-routes/index.json` に preset を並べると、左パネルの `Public Robot Routes` から `Apply / Launch / Copy URL` できます
- `Public Robot Missions` は `Mission Ready / Mission Warning / Mission Missing` を出して、route / zone / world metadata のズレを先に見られます
- route shelf と `Public Robot Routes` は `World Match / Fragment Drift / Frame Drift / Legacy Route` を出して、current world とのズレを先に見られます
- `npm run stage:robot-route -- --source ./public/robot-routes/residency-window-loop.json --route-id residency-window-loop --force` で route JSON と public catalog を一緒に更新できます
- semantic zone map があれば robot pose から current zone を判定し、stage overlay と right panel の `Semantic Zone Panel` に出します
- sample zone map は `public/manifests/robotics-residency.zones.json` と `public/manifests/robotics-echo-chamber.zones.json` にあります
- `Semantic Zone Workspace` では current fragment の `bounds / resolution / default cost / zone list` を直接編集でき、`Save / Copy / Download / Import / Reset` できます
- `Nav / Cost Overlay` は top-down で zone cost、robot pose、route、waypoint をまとめて確認する panel です
- Top View では stage 上にも zone footprint を薄く重ね、safe / hazard 帯を camera view 上で読みやすくしています
- zone editor には `Add Zone At Robot / Waypoint`、`Zone <- Robot / Waypoint`、`Robot -> Zone`、`Duplicate` を入れてあります
- batch ops では robot trail を `route zone` として一括生成し、zone extents から bounds を自動再計算できます
- `npm run stage:robot-zones -- --source ./public/manifests/robotics-residency.zones.json --fragment residency --force` で tuned zone map を public manifest へ publish できます
- bridge は `dreamwalker-robotics/v1` を正規 protocol とし、legacy message も互換で受けます
- bridge では `robot-state` を publish し、`request-state / set-pose / teleop / set-waypoint / clear-waypoint / clear-route / reset-pose / set-camera` を受けられます
- `npm run robotics:client -- request-state` で現在 state を取得できます
- `npm run robotics:client -- teleop forward` や `npm run robotics:client -- set-camera chase` で外部 terminal から指示できます
- `npm run robotics:client -- watch` で bridge 上の v1 message を監視できます
- `npm run robotics:rosbridge -- --rosbridge ws://127.0.0.1:9090 --frame-id dreamwalker_map` で `rosbridge_suite` 互換 relay を起動できます
- ROS2 側では `gs-sim2real robotics-node --namespace /dreamwalker --request-state-on-start` で受信 scaffold を起動できます
- semantic zone を使う場合は `gs-sim2real robotics-node --namespace /dreamwalker --zones-file configs/robotics/dreamwalker_zones.sample.json` を使います
- relay は native topic と ROS map topic を両方 publish します
- native 側は `/dreamwalker/robot_state_json`、`/dreamwalker/robot_pose2d`、`/dreamwalker/robot_waypoint`、`/dreamwalker/robot_route_json` です
- ROS map 側は `/dreamwalker/robot_pose_stamped`、`/dreamwalker/robot_goal_pose_stamped`、`/dreamwalker/robot_route_path` です
- ROS2 scaffold 側は `/dreamwalker/semantic_zone_summary_json`、`/dreamwalker/current_zone_json`、`/dreamwalker/semantic_costmap` を追加 publish します
- relay は `/dreamwalker/cmd_json`、`/dreamwalker/cmd_pose2d`、`/dreamwalker/cmd_waypoint`、`/dreamwalker/cmd_vel` を受けます
- stage overlay には robot trail と waypoint への goal line を重ねます
- いまの robot mode は `teleop / camera / waypoint` の最小 sandbox で、physics source of truth は引き続き collider mesh / proxy collider です

## Asset workspace

- 左パネルの `Asset Workspace` から current fragment の `splatUrl / colliderMeshUrl / worldNote` を編集できます
- 変更はその場で preview に反映されます
- `Save Asset Workspace` を押すと `dreamwalker-live-asset-workspace` として localStorage へ保存されます
- `Reset Asset Workspace` で manifest file か template に戻せます
- `Copy / Download Asset Workspace JSON` で他環境へ持ち出せます
- `Apply Pasted Asset Workspace JSON` で clipboard から import できます
- `Import Asset Workspace File` で `.json` を読み込めます

## Scene workspace

- Live Mode 右パネルの `Stream Scene Workspace` から current scene の `label / title / topic / preset / memo / branding` を編集できます
- 変更はその場で preview と `Live Scene JSON` に反映されます
- `Save Scene Workspace` を押すと `dreamwalker-live-scene-workspace` として localStorage へ保存されます
- `Reset Scene Workspace` で config の template に戻せます
- `Copy / Download Scene Workspace JSON` で他環境へ持ち出せます
- `Apply Pasted Scene Workspace JSON` と `Import Scene Workspace File` で復元できます

## Studio bundle

- 左パネルの `Studio Bundle` は `Asset Workspace + Scene Workspace + Semantic Zone Workspace + Robot Route + 現在の fragment / scene / preset state` を 1 つの JSON に束ねます
- `Copy / Download Studio Bundle JSON` で別マシンや別配信環境へ持ち出せます
- `Apply Pasted Studio Bundle JSON` と `Import Studio Bundle File` で asset / scene / semantic zone / robot route の各 draft と stage state をまとめて復元できます
- bundle の適用は draft 更新なので、必要ならその後に `Save Asset Workspace`、`Save Scene Workspace`、`Save Zone Workspace`、`Save Route Snapshot` で永続化します
- `Save Studio Bundle Snapshot` で現在の bundle を shelf に積めます
- shelf には最大 8 件まで保存し、`Apply / Download / Delete` で運用できます
- `?studioBundle=/studio-bundles/dreamwalker-live.sample.json` で repo 内の bundle file を直接起動できます
- `public/studio-bundles/index.json` に bundle を並べると、左パネルの `Public Studio Bundles` から `Apply / Launch / Copy URL` できます
- `Public Studio Bundles` は bundle file を読んで `Ready / Demo Fallback / Missing` をその場で出します
- `World Health` は current fragment の local file まで確認して、`Missing Splat File / Missing Collider File` を出します
- `Public Studio Bundles` は `error` の entry を `Apply / Launch` できないようにしています

## Validation

- `npm run validate:studio` で `asset manifest / public studio bundle / public robot route catalog / semantic zone manifests` をまとめて検証できます
- `npm run validate:studio -- --public-root /tmp/dreamwalker-public --mission-only --robot-route-catalog /tmp/dreamwalker-public/robot-routes/index.json` を使うと temp public root の mission publish 結果だけを検証できます
- local public file を参照している `splatUrl / colliderMeshUrl / bundle url` は存在チェックします
- public robot route は `pose / route` 構造、`world` metadata、catalog の `fragmentId` 整合、local `splat / collider` 参照、`zoneMapUrl / frameId` の整合まで見ます
- zone map が local public path の場合は、route の `coverage / hazard node / bounds 外れ` まで集計します
- semantic zone manifest は `buildSemanticZoneMap` で parse し、`zones / frame / resolution` と fragment path の整合を見ます
- demo fallback や proxy collider は warning、bundle file 不在や splat file 不在は error になります
- 実 Marble asset を置いたあとにこのコマンドを回すと、配信前の参照切れをかなり減らせます

## Real Asset Staging

- `npm run stage:fragment -- --fragment residency --source-dir /path/to/export --force` で実 asset を DreamWalker 用の public 配置へ流し込めます
- `--splat https://... --collider https://...` を使うと、ローカル source が無くても remote export URL から直接 staging できます
- fragment ごとに `expectedSplatUrl / expectedColliderMeshUrl` をベースに staging するので、Residency と Echo Chamber の導線を揃えやすいです
- `--bundle-id` と `--bundle-label` を付けると、そのまま `Public Studio Bundles` から launch できる catalog entry が作られます
- `--manifest-path`、`--catalog-path`、`--bundle-path`、`--public-root` を使うと、別 manifest や temp public root にも出力できます
- 実運用では `stage:fragment -> validate:studio -> /?studioBundle=... で確認` の順が最短です

## Robot Route Publish

- `npm run stage:robot-route -- --source /absolute/path/to/route.json --fragment residency --route-id residency-patrol --force` で route preset を `public/robot-routes/` と catalog へ publish できます
- source route に `world` metadata が薄い場合でも、asset manifest と fragment config から `assetLabel / splatUrl / colliderMeshUrl / zoneMapUrl` を補完します
- `--route-label`、`--description`、`--accent` で catalog entry を調整できます
- `--asset-manifest`、`--catalog-path`、`--route-path`、`--public-root` を使うと temp 出力や別 catalog にも流せます
- local zone map がある場合、publish 時に `coverage / hazard node / bounds 外れ` の summary を返します
- 実運用では `stage:robot-route -> validate:studio -> /?robotRoute=... で確認` の順が最短です

## Route Zone Analysis

- `npm run analyze:robot-route -- --route ./public/robot-routes/residency-window-loop.json --zones ./public/manifests/robotics-residency.zones.json` で route と zone map の整合を単体で確認できます
- text 出力では `coverage / hazard / outside bounds / maxCost / visited zones` をまとめて見られます
- `--json` を付けると node ごとの `zoneLabels / tags / maxCost / hazard / outsideBounds` に加えて、`uncoveredSegments / hazardSegments / outsideBoundsSegments / recommendations` を含む詳細 report を stdout に出します
- `--corridor-padding 1.0` と `--bounds-padding 1.5` を使うと、safe corridor と zone bounds の suggestion を実 world に合わせて少し広めにできます
- `--output ./tmp/route-analysis.json` を付けると同じ report を file に書けます
- 実運用では `Route Export から Download -> analyze:robot-route -> zone 調整 -> stage:robot-route -> validate:studio` の順が最短です

## Route To Zone Suggestion

- `npm run suggest:robot-zones -- --route ./public/robot-routes/residency-window-loop.json --zones ./public/manifests/robotics-residency.zones.json --output ./tmp/residency-suggested.zones.json` で uncovered segment から safe corridor zone の叩き台を作れます
- 生成 zone は `shape=rect`、tags は `safe / corridor / suggested`、cost は既定で `15` です
- `--corridor-padding` で corridor の太さ、`--cost` で generated zone の cost、`--label-prefix` で生成ラベルを調整できます
- `--include-hazard-review` と `--include-bounds-review` を付けると、hazard overlap と bounds 外れ区間も `review` zone として追加できます
- hazard review zone は既定で `cost 65 / tags review,hazard-overlap,suggested`、bounds review zone は `cost 30 / tags review,bounds,suggested` です
- `--merge-bounds` を付けると route の padded bounds を zone manifest 全体の bounds に反映します
- 実運用では `analyze:robot-route -> suggest:robot-zones -> Semantic Zone Workspace で微調整 -> stage:robot-zones -> validate:studio` の順が最短です

## Zone Autotune

- `npm run tune:robot-zones -- --route /absolute/path/to/route.json --zones /absolute/path/to/zones.json --fragment residency --merge-bounds --include-hazard-review --include-bounds-review --force --validate` で `suggest -> stage -> validate` を 1 本で回せます
- suggestion の中身は `suggest:robot-zones` と同じで、safe corridor / hazard review / bounds review を一括生成します
- `--zone-path` と `--public-root` を使うと temp public root への dry-run staging もできます
- `--dry-run` を付けると final zone file は書かず、stage 側の planned output だけ見られます
- `--keep-temp` を付けると中間の suggested zone JSON を残せます
- 実運用では `Route Export -> tune:robot-zones -> Semantic Zone Workspace で微調整 -> validate:studio` の順が最短です

## Robot Mission Publish

- `npm run discover:robot-bundles` で repo / `raw_assets` / `public` / `~/Downloads` / `~/.claude/downloads` から `artifact-pack.json / mission manifest / splat / collider` をまとめて探せます
- `npm run discover:robot-bundles -- --validate` を付けると、見つかった artifact pack に `validate:robot-bundle` も掛けて summary まで出します
- `npm run bundle:robot-mission -- --mission ./public/robot-missions/residency-window-loop.mission.json --output /tmp/residency-window-loop.artifact-pack.json` で、committed な mission / route / zone から release 入力用の artifact pack を再生成できます
- `bundle:robot-mission` の output は `draft-bundle / mission / published-preview / launch-url / preflight-summary / publish-report / validate-command / release-command / publish-command` を含むので、そのまま validator と release CLI に流せます
- `npm run validate:robot-bundle -- --bundle /absolute/path/to/dreamwalker-live-residency-robot-mission-draft-bundle.artifact-pack.json` で artifact-pack 単体の preflight を先に見られます
- `npm run release:robot-mission -- --bundle /absolute/path/to/dreamwalker-live-residency-robot-mission-draft-bundle.artifact-pack.json --force --validate` で `validate:robot-bundle -> publish:robot-mission` を 1 本で回せます
- `npm run release:robot-mission -- --discover --force --validate` で default search roots から最新の artifact pack を自動選択して release できます
- `npm run release:robot-mission -- --discover --root /absolute/path/to/dropbox --force --validate` のように `--root` を足すと探索範囲を絞れます
- `release:robot-mission` は local artifact pack に対して、未指定なら `<artifact stem>.preflight.txt` と `<artifact stem>.publish-report.json` を bundle の隣へ自動出力します
- `--output-dir /absolute/path/to/release-artifacts` を付けると、その auto output 先だけを別ディレクトリへ逃がせます
- `npm run publish:robot-mission -- --bundle /absolute/path/to/dreamwalker-live-residency-robot-mission-draft-bundle.json --force --validate` で browser 側 `Mission Export` の draft bundle をそのまま public mission へ publish できます
- `npm run publish:robot-mission -- --bundle /absolute/path/to/dreamwalker-live-residency-robot-mission-draft-bundle.artifact-pack.json --force --validate` でも同じように publish できます
- `npm run publish:robot-mission -- --route /absolute/path/to/route.json --zones /absolute/path/to/zones.json --fragment residency --tune-zones --route-id residency-patrol --force --validate` で zone tuning と route publish をまとめて回せます
- `--bundle` は draft bundle JSON だけでなく artifact-pack JSON も受けられて、mission / route / zones / startup state をまとめて読み込み、CLI 側では必要な temp route / zone file を自動生成します
- `validate:robot-bundle` は embedded `preflight-summary / publish-report / validate-command / release-command / publish-command` と local public route / zone / mission URL をまとめて検査します
- `--tune-zones` を付けると `tune:robot-zones` を先に実行し、その zone map を参照する route preset を続けて publish します
- `--tune-zones` を付けない場合は `stage:robot-zones` 相当で zone file をそのまま publish してから route preset を出します
- `--route-id / --route-label / --description / --accent` は `stage:robot-route` 側へそのまま渡ります
- `publish:robot-mission` は route preset と zone manifest に加えて `public/robot-missions/*.mission.json` と `public/robot-missions/index.json` も更新します
- mission manifest の `launchUrl` は `?robotMission=...` で自分自身を起動する形で出力され、validator は legacy な `?robotRoute=...` launch を warning 扱いにします
- `--camera-preset / --robot-camera / --stream-scene / --startup-mode` を使うと mission 起動時の stage 状態も固定できます。未指定時は fragment default と `robot` mode を使います
- publish 実行時は browser 側 `preflight summary` と同じ順序の `Mission Preflight` を stdout に出します
- `--preflight-output /absolute/path/to/mission.preflight.txt` を付けると、その summary を text file として残せます
- `--report-output /absolute/path/to/mission.publish-report.json` を付けると、mission / route / zones / startup / preflight をまとめた machine-readable report JSON を残せます
- browser の `Mission Publish Command` は今の metadata 編集内容をそのまま反映するので、`Mission ID` を変えると artifact-pack 名も一緒に更新されます
- browser の `Mission Validate Command` も同じ artifact-pack 名と preflight comment を使うので、publish 前の bundle 単体検査を UI からそのまま CLI へ渡せます
- `--zone-path / --route-path / --public-root / --catalog-path / --asset-manifest` を使うと temp public root や別 catalog への一括 publish もできます
- `--validate` は custom `public-root` の時でも `mission-only` validator を自動で呼ぶので、temp publish 後の route / zone 整合までそのまま見られます
- 実運用では `Mission Export -> publish:robot-mission --bundle -> /?robotMission=... で確認 -> validate:studio` の順が最短です
- publish 前に 1 回 `validate:robot-bundle` を挟むと、artifact-pack だけで route/zone/launch のズレを早めに見つけられます

## Zone Publish

- `npm run stage:robot-zones -- --source /absolute/path/to/zones.json --fragment residency --force` で tuned zone map を `public/manifests/robotics-<fragment>.zones.json` へ publish できます
- `--frame-id`、`--resolution`、`--default-cost` で publish 時に共通値を上書きできます
- `--zone-path`、`--public-root` を使うと temp 出力や別 manifest path への書き出しもできます
- 実運用では `Semantic Zone Workspace で tuning -> Download Zone JSON -> stage:robot-zones -> validate:studio` の順が最短です

## 次に入れるもの

- collider debug toggle の UI 化
- fragment ごとの overlay accent / branding
- OBS overlay の見た目強化
- robot websocket bridge polish / route export
- ROS2 node からの semantic zone / cost map 連携

## OBS overlay

- 通常画面の Live Mode は `dreamwalker-live-overlay-state` を localStorage へ publish
- `/overlay.html` はその state を読む軽量な overlay view
- `Copy Overlay URL` で OBS browser source 向け URL をコピー可能
- `npm run relay` を起動したうえで `/?relay=1` を開くと、Live scene state を relay へ publish
- OBS browser source は `/overlay.html?relay=1` を使うと、別 process / 別 profile でも同期できる
- relay を別ホストへ置く場合は `?relayUrl=https://relay.example.com` を追加する
- overlay preset は `7 / 8 / 9` か Live Mode 右パネルから切り替え可能
- overlay payload には `overlayPresetId / overlayPresetLabel` も含まれます
- fragment が切り替わると `overlayBrandingId / accent / strapline` も更新されます
- `4 / 5 / 6` で stream scene を切り替えると、scene ごとの badge / strapline override も反映されます
- 同時に `overlayMemoTitle / overlayMemoItems / overlayMemoFooter` も更新されるので、話す内容を browser source 側へ持ち出せます
- `Scene Workspace` で title / topic / memo / preset / branding を編集すると、その内容が overlay payload にそのまま流れます

注意:
- いまの同期は same-origin / 同一ブラウザ系の localStorage 前提です
- `?relay=1` は localStorage を残したまま relay も併用するので、同一ブラウザ preview と OBS browser source を両立できます
