# 屋外 3D Gaussian Splatting / Physical AI Simulation 開発計画

更新日: 2026-06-10。star 獲得フェーズ第 2 弾(zero-install demo / ROS 2 live mapping / README 整理)を §1.3 に、次期開発ロードマップ「動画ワンライナー → VGGT backend → 3DGS localization」を §18 に追加。前回更新は 2026-06-09（Istanbul Dynamic Map Viewer / large-scale 3DGS promotion / PR #192 merge 後ロードマップ反映）

この文書は、3DGS Robotics の屋外 3DGS パイプラインと、その上に載せる Physical AI simulation / policy benchmark / scenario CI の現行計画をまとめる長めの handoff です。

古い PR ごとの transcript、`tuhh_day_04` の誤判定、MCD calibration 探索、実走ログ、個別コマンドの長い出力は [archive snapshot](archive/plan_outdoor_gs_2026_04_full_handoff.md) に残しています。本書は「次にどこへ進むか」を判断するための source of truth として更新します。

## 0. 読み方

1. まず **TL;DR** と **現在の主戦場** を読む。
2. 実装に入る前に **System Map** と **Scenario CI Pipeline** を確認する。
3. 屋外データ / viewer / external SLAM だけ触るなら **Outdoor 3DGS Track** を読む。
4. Physical AI / policy benchmark / CI を触るなら **Physical AI Simulation Track** を読む。
5. star 獲得まわり(zero-install demo / live mapping / 動画ワンライナー / VGGT / localization)だけ触るなら **§1.3 → §10 → §18** を読む。
6. 古い実験値、MCD calibration、`ntu_day_02` 実走値、PR #55〜#80 の履歴が必要な場合だけ archive を読む。

## 1. TL;DR

- 3DGS Robotics は、写真フォルダ、Autoware / MCD の robotics logs、MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR などの external SLAM artifacts を、3D Gaussian Splatting training / export / browser viewer へつなぐ repo。
- Public demo は GitHub Pages で公開済み。`docs/scenes-list.json` が README table / preview capture / hero GIF / viewer picker の source of truth。
- Production viewer picker は 9 scenes。2 supervised、4 pose-free、3 external-SLAM comparison。
- MCD `tuhh_day_04` の supervised GNSS 成功扱いは撤回済み。`/vn200/GPS` が all-zero なので production picker には入れない。
- Valid GNSS supervised MCD demo は `ntu_day_02`。production asset は `docs/assets/outdoor-demo/mcd-ntu-day02-supervised.splat`。
- External SLAM import は VGGT-SLAM 2.0 / MASt3R-SLAM / Pi3X comparison splat まで実走済み。LoGeR profile も artifact resolver 側に候補追加済み。
- 2026-06-10 時点の `main` は `0fe34b1`。当日の star growth chain(§1.3)で HF Spaces zero-install demo(RUNNING / cpu-basic)、Colab、ROS 2 live mapping、実データ「map grows」GIF(KITTI drive 0056)、README リンク削減(67→35)、GitHub About 簡素化まで public に出た。
- 次期開発は §18 の star growth ロードマップ: (a) 動画 1 本 → 地図ワンライナー、(b) VGGT feedforward backend、(c) 3DGS localization(設計済み・着手判断は後)。PyPI 公開と ZeroGPU 切替はユーザー判断で保留。告知はユーザーが実行。
- 2026-06-09 時点の `main` は PR #192 merge commit `cf7a22e`。README / Pages / Dynamic Map Viewer には Istanbul real rosbag2 pilot の結果 media と viewer catalog が入った。
- Dynamic Map Viewer は 2 系統を持つ。1 つは 9 production `.splat` から組んだ 87-tile regional mosaic、もう 1 つは Istanbul `rosbag2` 由来の 6-tile real-input pilot。
- Istanbul pilot は 291 registered camera frames / GNSS-seeded poses / 100,000 sparse seed points / 6 trained source tiles / 6 browser `.splat` / 6 viewer PLY tiles。Viewer PLY は 438,796 Gaussians / 103.8 MiB。
- `large-scale-3dgs-run` promotion は source XY tiles を viewer XZ PLY へ変換し、catalog に `viewerSplatUrl` / `viewerCoreBounds` / `viewerExpandedBounds` / `viewerTileIndex` / viewer metrics を載せる段階まで進んだ。
- 次の最優先は **GitHub Pages 反映確認** と **live PlayCanvas GSplat 表示の実証**。asset / catalog / route / README media は固まったが、headless 検証では live canvas の 3DGS 表示がまだ完全には証明できていない。
- 2026-04-24 時点では、屋外 3DGS だけでなく **Physical AI simulation benchmark environment** を目指す方向へ拡張中。
- Route policy benchmark 系は、dataset / imitation / registry / benchmark / history / scenario-set / matrix / sharding / CI manifest / workflow materialization / validation / activation / review bundle / workflow trigger promotion gate / promotion-backed trigger adoption / adoption-aware review bundle まで分割済み。
- 最新の merged public refresh は `cf7a22e` / PR #192。Tier 2 chain (#121–#134) で env-hardening + correlation gate plumbing が完成した後、6 月 chain (#188〜#192) で Dynamic Map Viewer / large-scale 3DGS / Istanbul real-input pilot が public docs に載った。
- adoption step + CLI (`3dgs-robotics route-policy-scenario-ci-workflow-adopt`) + adoption-aware review bundle まで実装済み。review には `--adoption-report` を渡すと Pages の `review.{json,md,html}` に trigger mode / branches / manual vs adopted YAML の unified diff が乗る。
- Pages `/reviews/` には synthetic smoke fixture 由来の `docs/reviews/smoke-route-policy-ci/` sample bundle を置く方針に変更。`scripts/build_pages_sample_review_bundle.py` が smoke chain を回し、temp path を self-contained な `sample-artifacts/` 相対リンクへ書き換えて index も再生成する。
- 2026-04-25 〜 26 の Tier 2 rollup で、real-vs-sim correlation library (#113/#115) → scenario-set run report への attach (#121) → review bundle への surface (#125) → regression gate (#126) → per-bag overrides (#128) → translation/heading pair-distribution gates (#129/#130) → time stratification (#131/#132) + equal-pair-count mode (#133) + per-window stats (#134) まで一気に完成。`3dgs-robotics route-policy-scenario-ci-review` の correlation gate は実用 production rollout で使える状態。
- MCD Profile 3 が参照していた `/d455t/color/image_raw` topic は MCDVIRAL 全 18 session に存在しないと判明 (Download page + calibration_atv.yaml で交差確認)。「data-blocked」と思われていた状態は実は spec ミスであり、2-camera (d455b + d435i) の `multi_2cam_300each_ba` に redefine 済み。詳細は §3.2 に recipe 化。
- 同時に env-hardening 側も IMU finite-diff renderer (#111) → ObstaclePolicy protocol (#112) → IMU + peer-aware features を gym adapter feature dict へ surface (#122/#123) → query_collision / score_trajectory に per-step peer cache を threading (#124/#127) で multi-agent サポートが整った。

### 1.1 2026-06-05: star 獲得に向けた 3DGS 見栄え改善フェーズ

2026-06-05 の作業方針は、単に機能を増やすのではなく、初見のユーザーが README と Pages を開いた瞬間に「これは実データの 3DGS プロジェクトだ」と判断できる状態へ寄せること。これまでの 3DGS Robotics は、Physical AI scenario CI、route-policy benchmark、external SLAM import、MCD/Autoware preprocessing などの技術要素が多く、repo の深さはある一方で、GitHub の first view では価値が見えにくかった。star を伸ばすには、最初に見えるものを「実際の屋外 splat」「比較できる SLAM 由来の地図」「自分の写真や外部 SLAM artifact から同じ流れに入れる CLI」に寄せる必要がある。

このため、6 月フェーズでは README first-view と public `.splat` 品質を優先した。README 冒頭は live 3DGS demo、Mission Control proof、Scenario CI reviews への導線を前面に出し、hero GIF も単なる装飾ではなく、shipped `.splat` binary から座標を読んで FPS-style に描く proof view へ切り替えた。最初の GIF は外側から点群を眺めるだけで「地図として使えるのか」が伝わりにくかったため、カメラを map 内側へ置き、top-down trace を重ねる形へ変更した。これにより README の main visual は「3DGS がある」ではなく「この地図の中を移動できる」に近い表現になった。

public asset 側では、既存の `.splat` が見栄えを壊す要因を分けて扱うことにした。新しく追加した `photos-to-splat --quality draft|balanced|clean|hero` は、新規に写真から splat を作るユーザーの入口を整えるための preset で、`--splat-max-scale-percentile` による adaptive scale gate を export 時に使えるようにした。さらに、すでに `.splat` になっている公開 asset やユーザーの手元 asset も後から直せるように、`3dgs-robotics splat-filter` を追加した。これは antimatter15 32-byte `.splat` binary を直接読み、opacity と scale の閾値で cloudy な gaussian を取り除く軽量な cleanup command で、source PLY や torch / numpy / gsplat を要求しない。

その後、cleanup を感覚だけで運用しないため `3dgs-robotics splat-inspect` を追加した。`splat-inspect` は opacity min/p10/p50/p90/max、scale max の p50/p95/p98/p99/max、low-opacity ratio、scale tail ratio を text または JSON で出す。これにより、「この splat は曇っている気がする」という主観を、`scale_p98` と `scale_max` の乖離、低 opacity gaussian の割合、ファイルサイズと gauss count の変化として説明できる。production picker に出る 9 scene については CI で `scale_p98 <= 0.5` と `opacity_p10 >= 0.02` を見る gate も入れた。これは全ての外れ値を禁止する強い gate ではなく、「大量の巨大・半透明 gaussian bulk を production として出さない」ための下限品質チェックとして設計している。

実 asset cleanup の最初の対象は `bag6-vggt-slam-20-15k.splat`。この scene は comparison-quality の外部 SLAM artifact で、元々 214,570 gauss / 約 6.9 MB だったが、scale tail と低 opacity の散り方が README proof GIF で目立ちやすかった。`splat-filter --min-opacity 0.08 --max-scale-percentile 95` 相当の p95 clean を採用し、178,073 gauss / 5.7 MB まで落とした。結果として `scale_p98=0.0751`、`scale_max=0.1568`、low-opacity 0% になり、README benchmark table も `5.4 MB / 178k gauss` に更新した。これは「VGGT-SLAM の品質を過剰に良く見せる」ためではなく、巨大で薄い gaussian が browser blend を汚す問題を落とし、比較 scene として正直に見せるための cleanup である。

次の cleanup 対象は `mcd-tuhh-day04.splat`。この scene は production picker に入っている MCD DUSt3R pose-free で、元の `scale_p98` は 0.0899 と bulk は健全だったが、`scale_max=3.5842`、`scale_tail_ratio=39.9x` という極端な外れ scale を持っていた。また opacity `<0.08` の gaussian が 25,668 / 400,000 点あり、`splat-inspect` は cleanup suggestion を出す状態だった。ここでは VGGT と違い、p98 まで強く切る必要はない。proxy render と top-down 比較では、`min_opacity=0.08` + `max_scale_percentile=99` が構造を残しつつ散った外れ点を落とす最も保守的な選択だったため、370,679 gauss / 11.9 MB の clean asset を採用する方針にした。cleanup 後は low-opacity 0%、`scale_max=0.0940`、`scale_tail_ratio=1.1x` になる。

重要なのは、このフェーズでは「きれいに見せる」ことと「実データの弱点を隠さない」ことを分けること。外部 SLAM comparison scene は、default demo ではなく比較のための scene であり、README でも comparison-quality と明記している。したがって、training をやり直して結果を盛るのではなく、browser viewer で不必要に悪く見える gaussian export の外れ値を落とす。逆に、pose recovery が弱い、地物が欠ける、スケールが不安定といった本質的な reconstruction quality は README / benchmark table / labels で隠さない。この線引きがあるから、public demo は star を狙いつつも技術的に誠実でいられる。

今後の優先順位は次の通り。第一に、README と Pages の visual proof を継続して強くする。具体的には、preview PNG を実 WebGL で再キャプチャできる環境を用意し、asset cleanup 後の public scene と thumbnails のズレを減らす。現ローカル環境には Playwright / GPU-backed WebGL / ffmpeg が揃っていないため、今は `.splat` byte 由来の proxy render で判断しているが、最終的には `scripts/capture_readme_splat_previews.py` と `scripts/record_demo_gif.py` を GPU 環境で再実行し、README の visual 一式を production asset と同期させる必要がある。

第二に、asset health report を release / launch kit とつなぐ。`splat-inspect --json` の出力を `docs/launch-kit.json` または generated report に取り込み、public demo が「9 scenes を shipped している」だけでなく、「各 scene の gauss count、size、opacity p10、scale p98、tail ratio がこうなっている」と説明できるようにする。これは SNS 投稿や awesome list 申請よりも地味だが、3DGS / robotics に詳しい人ほど効く。雑に作った demo ではなく、export quality を測って公開している repo だと伝わる。

第三に、ユーザー向け quick path を磨く。`photos-to-splat --quality clean` と `splat-inspect` / `splat-filter` の組み合わせを README だけでなく、short tutorial / docs page にする。ユーザーが「写真フォルダを入れる」「`.splat` を inspect する」「曇っていたら filter する」「Pages viewer で確認する」という 4 step を迷わず踏める状態にしたい。ここができると、3DGS Robotics は単なる research repo ではなく、外部 SLAM artifact と browser-viewable 3DGS をつなぐ practical toolkit として見える。

第四に、MCD `ntu_day_02` / Profile 3 の GPU rerun を戻す。README first-view と public asset cleanup は star 獲得のための短期施策だが、repo の中核は real robotics data → 3DGS → Physical AI benchmark の chain である。`multi_2cam_300each_ba` の redefine は済んでいるので、GPU 環境が使えるタイミングで preprocess/train/export を実行し、gate report を更新する。MCD supervised の valid-GNSS asset はすでに `mcd-ntu-day02-supervised.splat` として production picker にあるが、profile plan と実 asset の説明をさらに合わせる余地がある。

第五に、Physical AI scenario CI との接続を README first-view からより強く見せる。3DGS visual だけでは「綺麗な点群 demo」で終わる可能性がある。3DGS Robotics の差別化は、同じ scene catalog を policy benchmark / route-policy scenario CI / review bundle に流すところにある。したがって、visual proof の次は、scene → sim-scenes catalog → policy benchmark → Pages review の一連の成果を、短い animated proof または compact diagram で見せる。ここまで見せると、star する理由が「3DGS が見える」から「robotics evaluation stack として使えそう」に変わる。

### 1.2 2026-06-09: Dynamic Map Viewer / large-scale 3DGS フェーズ

6 月 9 日時点の主戦場は、単体 `.splat` demo から **dynamic map loading が見える 3DGS viewer** へ移った。README first-view の GIF を良くするだけではなく、実際に「広い地図を tile catalog と route playback で読み替える」体験を public demo として出すことが目的になっている。Autoware 風の dynamic map loading に寄せたい、ただし hero に lanelet2 vector layer のような説明過多な layer を載せる必要はない、という判断で、現在の README / Pages は 3DGS footprint と resident / preload / route window を中心に見せる方向へ整理した。

このフェーズでまず作ったのは synthetic regional mosaic。9 production outdoor `.splat` を 25 placement の 5x5 X/Z region に並べ、87 browser-ready route tiles として Dynamic Map Viewer に載せた。これは「large-scale に見えるか」を検証する fixture であり、実ロボットログの連続 reconstruction ではない。価値は、tile catalog contract、route playback、resident/preload/evicted overlay、README media 生成、Pages URL の扱いを先に固めるところにある。結果として、dynamic map loading の UI / docs / validation path はかなり進んだが、実データとしての説得力には限界があった。

その次に Istanbul `rosbag2` の real-input pilot を追加した。ここでは 291 registered camera frames、GNSS-seeded pose、100k sparse seed points から 6 source tiles を作り、browser `.splat` と viewer PLY の両方を promotion した。README には `docs/images/istanbul-bag6-pilot/dynamic-map-viewer.gif`、`large-scale-3dgs-result.png`、`dynamic-map-viewer-still.png` を載せた。ユーザーが指摘した「点群が何か分からない」「large-scale と言うには弱い」という問題に対し、まず real rosbag 由来であること、route と tiles が同じ座標系で動くこと、result still が大きな footprint として読めることを優先した。

Istanbul pilot の重要な実装差分は、training output と viewer output を分けた点。large-scale 3DGS training は source tile を XY のような training frame で持つことがあるが、PlayCanvas / Dynamic Map Viewer 側では X/Z plane を地面として扱いたい。このため `src/gs_sim2real/train/large_scale_3dgs.py` の promotion path で binary PLY の `x/y/z` float を読み替え、viewer 用 PLY を生成するようにした。catalog には source splat metrics と viewer splat metrics を別々に持たせ、runtime は `viewerSplatUrl` と viewer bounds を優先する。これにより docs media と route playback が同じ viewer coordinate で揃う。

Dynamic Map Viewer 側では `apps/dreamwalker-web/src/dynamic-map-loading.js` が viewer field を優先して読み、`App.jsx` は overview camera を viewer bounds から決める。catalog validator は optional `viewerSplatUrl` を public root と URL の両方で検証し、route coverage も viewer X/Z bounds を見て判断するようになった。`validate-dynamic-map-catalog` は Istanbul catalog + route playback で pass しており、sparse rectangular grid の warning だけが残る。これは 6 / 12 occupied tiles の構成なので仕様上許容する。

ただし、ここで完了と言い切ってはいけない。README / Pages media、catalog、route、asset delivery は固まったが、live PlayCanvas GSplat の画面内表示はまだ最後まで潰しきれていない。headless smoke では assets が 200 で取れ、manager は splat を認識しているが、canvas pixel として「3DGS が見えている」証明が弱い。したがって次の大きな実装は、generated GIF を良くすることではなく、Dynamic Map Viewer の runtime renderer を実際に信用できる状態へ持っていくこと。

次の Definition of Done は次の 5 点にする。

1. GitHub Pages の deploy が `cf7a22e` 以降で成功し、README / docs landing / `/dreamwalker/` の Istanbul pilot URL が live で取れる。
2. PlayCanvas GSplat path が desktop headed browser で非 blank canvas として確認でき、camera framing / scale / clipping / asset type のどこで詰まっていたかが documented される。
3. Istanbul pilot の viewer PLY 6 tiles が live viewer 上でも route playback と同じ地面座標で見える。
4. README の GIF / still は runtime で確認できた geometry と矛盾しない。proxy render だけで作った素材はその旨を明記するか、WebGL capture へ置き換える。
5. その後に、real rosbag2 の tile 数 / frame 数 / spatial coverage を増やして、synthetic mosaic ではなく real large-scale 3DGS と呼べる規模へ広げる。

短期ロードマップは、細かい UI polish ではなく次の順で進める。

| 優先 | Task | 完了条件 |
| ---: | --- | --- |
| 1 | Pages deploy / live smoke | `gh run` または Actions で deploy success を確認し、hosted README / Dynamic Map Viewer URLs が 200 で Istanbul assets を返す |
| 2 | Live GSplat renderer debug | PlayCanvas scene で actual splat pixels が出る。blank の場合は asset format / loader / material / camera / clipping の原因を 1 つに絞る |
| 3 | Runtime capture pipeline | `docs/images/istanbul-bag6-pilot/` の GIF / still を runtime viewer 由来で再生成できる。proxy render と runtime capture の差分を減らす |
| 4 | Real large-scale expansion | Istanbul pilot を 6 tiles から、より長い route / more registered frames / more tiles へ拡張する。catalog metrics と result media を同時更新 |
| 5 | Physical AI 接続 | Dynamic Map Viewer の route / tile catalog を sim scene catalog / route-policy review bundle に接続し、「見える map」から「評価できる map」へ戻す |

この順番にする理由は、今の blocker が「素材が足りない」ではなく「viewer runtime で見えることの信頼性」だから。素材を増やす前に renderer path を確定しないと、large-scale 化しても README media と live demo のズレが増える。逆に renderer が固まれば、real rosbag2 を増やす作業は catalog / route / media generation の反復で進められる。

### 1.3 2026-06-10: star 獲得フェーズ第 2 弾 — zero-install demo / live mapping / README 整理

2026-06-10 は、§1.1 の「見栄え改善」から一歩進めて、**初見ユーザーがインストールせずに体験できる入口**と、**「地図が育つ」ことを実データで証明する media** を作るフェーズだった。当日の merged chain は `eeeb9ec` から `0fe34b1` までの 13 commits。大きく 4 つの塊に分かれる。

第一の塊は zero-install demo。`d63497c` で HF Spaces demo(`apps/hf-space/`、`sync-hf-space.yml` で同期)と Colab notebook を追加し、`eeeb9ec` で pip install ユーザー向けに training config のビルトインデフォルトへのフォールバックを入れた。HF Space のビルドは torch/gsplat wheel の組み合わせで 3 回詰まり(`8bae7ff` → `e3684a5` → `91c36f6`)、最終的に Python 3.10 + torch 2.4.1 + gsplat pt24cu121 + gradio 5.49.1 で **RUNNING** に到達した。ただしハードウェアは cpu-basic のままで実推論には遅い。ZeroGPU への切替は提案したが、**ユーザー判断で skip**(2026-06-10)。再提案はユーザーから話が出るまでしない。

第二の塊は ROS 2 live mapping(`104559a`)。`3dgs-robotics-live-mapper` node(コアは rclpy-free な `gs_sim2real/robotics/live_mapping.py`)がカメラ topic を keyframe gate に通し、background thread で DUSt3R + gsplat の draft rebuild round を回して `live/latest.splat` を atomic に置き換える。`docs/splat.html` の `?refresh=` polling で browser が地図を in-place 更新する。詳細契約は `docs/live-mapping.md`。

第三の塊は README first-view media の刷新。lead GIF は 2 回作り直し(`09835be` → `6abbb2a`)、最終的に Istanbul Bag6 pilot の実 ortho render + 復元カメラ軌跡に沿った dynamic-map loading window(`1c0f54a`)になった。さらに `1b6b15b` で「**map grows as the robot drives**」GIF を実データで収録した。手順は再現可能な形で固定してある: KITTI raw drive 0056(`/media/sasaki/aiueo/ai_coding_ws/_kitti`、5 フレーム間引き 59 枚)を `scripts/run_live_mapping_demo.py` で replay(`--fps 0.05` にして rebuild round と frame 給餌を並行させるのがコツ。`--no-realtime` だと round 1 完了前に全フレームが入ってしまい growth が見えない)、6 round 成功後に `scripts/build_live_mapping_gif.py` が `rounds/round_*/train/point_cloud.ply` を gauge 整合して真上 ortho で合成する。各 round は pose-free full rebuild なので独自 gauge を持つ。整合は共有 keyframe のカメラ **姿勢**(回転は Σ R_dst·R_srcᵀ の SVD、scale/translation はカメラ中心)から相似変換を作り、連続 round 間でチェーンして最終 round に合わせる。共有 2 カメラで成立する(中心のみの Umeyama は 3 必要 + 回転 1 自由度が残る)。**罠**: `scene.splat` は export 時に normalize されるため `images.txt` の座標系と合わない。整合には必ず `train/point_cloud.ply` を使う。また COLMAP writer は image 行ごとに空の POINTS2D 行を吐くので、blank を除去してから `[::2]` で間引くとカメラが半分消える。CPU unit tests は `tests/test_build_live_mapping_gif.py`。

第四の塊は public 面の整理(`0fe34b1` + `gh repo edit`)。README のリンクを 67 → 35 に削減(badge 7→4、Try-it-first を 3 リンクに、file-path リンクは plain code path 化、重複 HF/Colab リンク除去)し、GitHub About を「Photos and robot logs to browser-ready 3D Gaussian Splat maps」に簡素化した。`tests/test_pages_assets.py` の first-view test も新しいリンク構成に追従済み。当日終了時点で 995 tests pass、`main @ 0fe34b1` が origin と同期。

このフェーズの残りは開発ではなく**告知**(HN "Show HN" / X / awesome-list PR、素材は `docs/launch-kit.md`)で、これは**ユーザー自身が実行する**。開発側の次の弾は §18 の star growth 開発ロードマップ(動画ワンライナー → VGGT backend → 3DGS localization)に切り出した。

### 1.4 2026-06-10(午後): §18 の 3 項目を即日完了、次期目標は 100 stars(§19)

同日午後に §18 の開発 3 項目を全て実装し merge した。`edb109d` が動画ワンライナー(`3dgs-robotics map my_drive.mp4` 系の mp4→splat 一気通貫、`extract_frames.py` 拡張、HF Space への `gr.Video` 入力追加込み)と VGGT feedforward backend(`vggt_backend.py`、`--method vggt`、live mapping のラウンド preprocess が実測 ~30–90 秒に短縮、`docs/live-mapping.md` に記載)。`d32c0ac` が 3DGS localization(`robotics/localize.py` の keyframe retrieval + gsplat photometric SE(3) refinement、`3dgs-robotics localize` CLI、KITTI 0056 の非キーフレームをクエリにした MVP 実走と GIF `docs/images/live-mapping/localization-kitti0056.gif`)。`2674c85` で README の長尺 runbook を docs リンクに退避して整理した。

これで「弾は揃っているが告知が未実施」という状態になった(stars はこの時点で 9)。次のスター目標とそのための開発トラックはユーザーと協議のうえ **§19(目標 100 stars / 年内、rosbag 直接入力 + ループクロージャ)** に確定した。§18 は完了記録として残す。

同日、リポジトリ名を **`rsasaki0109/3dgs-robotics`** にリネームした(変遷: `gs-mapper` → `3dgs-mapper` → `3dgs-robotics`。robotics 応用 = mapping / localization / simulation を名前で見せるユーザー判断。GitHub/PyPI とも空きを確認済み)。GitHub の旧名 URL はリダイレクトされるが **Pages はリダイレクトされない**ため、README 等に残っていた旧名の `github.io` リンクは 404 だった。リンクは全面的に新名称へ修正済み(`pages.yml` の `DREAMWALKER_BASE`、Colab の clone/cd、launch-kit はジェネレータ `scripts/generate_launch_kit.py` から再生成)。さらに同日、**ブランドも全面統一した**: pyproject の dist 名と CLI を `3dgs-robotics`(+ `3dgs-robotics-node` / `3dgs-robotics-live-mapper`)へ改名し、旧コマンド(`gs-mapper` / `gs-sim2real` 系)は legacy alias として温存。HF Space も `rsasaki0109/3dgs-robotics` へ move 済み(`move_repo` API)。表示ブランド「GS Mapper」は「3DGS Robotics」に置換した。**触らないもの**: 成果物スキーマ ID(`gs-mapper-route-policy-*/v1` 等の version 識別子 — 既存 artifact との互換のため凍結)と `docs/archive/`(スナップショット)。

### 1.5 2026-06-11: §19 の開発項目(rosbag 入力 + ループクロージャ 3 段階)を実装

§19 の開発トラックを A → B Step1 → Step2 → Step3 の順に実装した。

- `5c81cbb` **rosbag 直接入力(§19.3)**: `datasets/rosbag_frames.py`(bag 探索・image topic 解決・Image/CompressedImage デコードを MCD loader から汎用化、MCD は委譲に置換)。`run_live_mapping_demo.py --bag/--image-topic/--rate`(bag タイムスタンプ駆動の replay)と `3dgs-robotics map <bag>`(入力判別→フレーム抽出→既存 photos-to-splat)。pyproject に `rosbag` extra、dev に `rosbags` を追加して CI で synthetic bag テストが回る。完了条件 4 点は全て消化: KITTI 0056 から変換した rosbag2(59 frames、実タイムスタンプ)で `map` ワンショット→ `.splat` 生成、`--bag` replay → VGGT round が回り `live/latest.splat` 出力、synthetic bag fixture テスト(rosbags の Writer で生成、ROS 非依存)、README / docs/live-mapping.md の入口 2 本立て化。
- `2df778b` **ループクロージャ Step1+Step2(§19.4)**: `robotics/gauge_alignment.py`(GIF スクリプトの Sim3 整合を共通化)+ `SplatRebuilder` が各 round 完了時に直前 round と整合してから export(セッションゲージ固定、正規化は round 1 で凍結、`gauge_transform.json` 永続化、GIF は fast path で流用)。`ply_to_splat` に `similarity_transform` / `normalize_params` を追加。**実測**: KITTI bag replay で VGGT round 間のスケール差 3.5 倍が共有 2 カメラの整合で吸収された。Step2 は `RevisitDetector`(キーフレームゲートと同じ 64x64 サムネイル距離、時間 / index 分離ゲート)→ `live/loop_candidates.json` + `state.json` カウント + `scripts/plot_loop_candidates.py`(軌跡上にループ edge を描画)。
- **Step3(round 単位 Sim3 pose graph)**: `robotics/gauge_pose_graph.py`。round はノード、「キーフレームを共有する全 round ペア」(round は全履歴をストライドするので遠い round 同士も恒常的に共有する=ループ edge を内包)を相対 Sim3 edge とし、7 パラメータ chart 上の damped Gauss-Newton(pure numpy、新依存なし — scipy も不要だった)で全 round のセッションゲージ変換を refine。3 round 目以降ランタイムで毎回実行し、全 round の `gauge_transform.json` を `optimized: true` で書き直してチェーンは refined 最新から継続。**正直な限界も docs に明記**: 補正は round 間ゲージドリフトのみで、1 round 内(swin pair graph がループ両端を繋がない)のドリフトは backend 側の課題として残る。

未消化: Show HN 第 2 弾用の before/after GIF(ループを含む drive の選定から、§19.4 MVP)と、loop candidate からの per-pair 再推定 edge(v2)。

### 1.6 2026-06-11(続): robotics 応用 5 本 — localizer / Isaac Sim / カメラシミュレータ / nav2 グリッド / 変化検出

「3dgs-robotics の名前通り robotics 応用を考える」というユーザー指示で応用マップ(Localization / Navigation / Simulation / Perception)を整理し、おすすめ順の上位 2 本を即日実装した。

- `e4b9016` **ROS 2 localizer ノード(§18 から保留だったもの)**: `3dgs-robotics-localizer`。カメラ topic を subscribe し、live-mapping セッション地図に対して retrieval + photometric 精緻化(localize CLI と同じコア)で姿勢推定 → `PoseStamped` / `nav_msgs/Path` / `map->camera` TF を publish。localize コアに ndarray API(`localize_image`)とステートフル `SessionLocalizer`(地図キャッシュ + `--follow-latest` で新 round 追従)を追加。最新フレームのみ処理する worker でカメラレートに依存しない。実測: e2e_bag_live3 の保留キーフレームをゲージ相対誤差 0.20 で推定(40 iters × 2 scales、~4.5s/query 初回込み)。
- `e374bf8` **Isaac Sim エクスポート(simulation 柱の入口)**: `3dgs-robotics export-isaac`。Isaac Sim 5.0+ がネイティブ描画する Omniverse NuRec USDZ へ、nv-tlabs/3dgrut の公式 transcode スクリプト(CUDA ビルド不要パス、依存は plyfile/msgpack/usd-core のみ)をラップして変換。`--map <session>`(round 解決)or `--ply` 直指定。`docs/isaac-sim.md` に setup / import / 注意点(非メトリックゲージのスケール合わせ、ground plane 追加、draft round の再学習推奨)。うちの gsplat PLY は標準 INRIA レイアウト(f_dc + f_rest×45)で transcode がそのまま食える。**e2e 検証済み(2026-06-11)**: e2e_bag_live3 round 5(173,671 gaussians、SH degree 3)→ 20.5 MB NuRec USDZ を `export-isaac --map` で生成。3dgrut transcode の実際の追加依存は `nvidia-ncore` / `simplejpeg` / `tensorboard`(requirements 非記載分を実走で洗い出し、docs に反映)。途中で見つけた相対パスバグ(`cwd=3dgrut` で起動するため)は `1121443` で修正。Isaac Sim 本体への取り込み確認はユーザー側。

- **GS カメラシミュレータノード(simulation 柱の ROS 2 側、2026-06-11 続投で実装)**: `3dgs-robotics-camera-sim`。学習済み地図(セッション round or 任意の 3DGS PLY)を `HeadlessSplatRenderer`(gsplat CUDA / numpy フォールバック)でレンダし、`PoseStamped` 入力 or `--replay`(セッションのキーフレーム軌跡再生、`--gt-pose-topic` で真値も publish)から `CompressedImage`/`Image` + `CameraInfo` + 任意の depth を配信。セッションの COLMAP 内部パラメータをそのまま使うので localizer とレンダが一致する(`render_rgbd` に `intrinsics` 上書きを追加)。姿勢規約は localizer と同じ optical convention。simple バックエンドは y-up 投影なので出力を垂直反転して合わせる(`render_optical`)。**closed-loop e2e 検証済み**: camera-sim が e2e_bag_live3 round 5 の軌跡を replay → localizer がそのレンダ画像だけで自己位置推定。レンダ vs 実キーフレーム L1≈0.09–0.12(draft round 品質なり)、topic 実測 5.08 Hz(指定 5)、推定 center はキーフレーム間隔の約 7% 誤差(kf_000010: 推定 (-0.03,-0.01,0.37) vs GT (-0.035,-0.009,0.382)、spacing 0.18)。

- **nav2 OccupancyGrid エクスポート(navigation 柱、同日続投で実装)**: `3dgs-robotics export-grid`。セッション round のガウシアンを推定地面平面へ射影し、nav2 `map_server` がそのまま読む `map.pgm` + `map.yaml`(+ 3D フレームを記録する `map.json` sidecar)を出力。非メトリックゲージ対策として全パラメータを**カメラ対地高さ単位**に正規化(up はカメラ optical -y の平均、地面は below-camera 高さの下位パーセンタイル、セル解像度デフォルトはカメラ高さ/20)。占有 = 障害物帯(0.2〜2.0 カメラ高)のガウシアン密度、自由 = 地面点 + 軌跡の swept corridor(線分補間)、他 unknown。e2e_bag_live3 で検証: 2272×339 セル、占有 1.0% / 自由 9.4%、道路コリドーと道路脇の並木が正しく分離。draft round のフローター対策は `--min-opacity` / `--min-points-per-cell` でチューニング可能と docs に明記。

- **変化検出(perception 柱、同日続投で実装)**: `3dgs-robotics detect-changes`。2 つの地図(同一セッションの round 間 or 別セッション)を Sim3 でゲージ整合し、カメラ高さ単位のボクセル密度差分から「出現 / 消失」クラスタ(26 近傍 BFS 連結成分)を検出。整合は `--align shared`(共有キーフレーム、gauge_alignment 再利用)/ `--align localize`(B のキーフレームを A の地図で 3DGS 自己位置推定 → 対応姿勢から Sim3、別セッション間用、localizer 注入可能でテスト容易)/ none。出力は changes.json(A ゲージの centroid/extent)+ 真上 preview PNG。e2e_bag_live3 round 5 vs 4 で検証: shared 整合 11 keyframes / scale 1.003、厳しめ閾値(--min-cluster-voxels 25 --min-count 5)で**出現 1 クラスタ / 消失 0** = round 5 で新規地図化された道路先端(z≈2.0)をピンポイント検出。draft round 同士はスプラット再配置ノイズでスペックルが出るため閾値チューニングを docs に明記。

robotics 応用マップ(Localization / Simulation / Navigation / Perception)の 4 本柱はこれで全て最低 1 実装が揃った。

- **「面白い開発」残り 2 本も同日夜に完了**: ③ 地図マージ `acaa2f7` = `merge-maps`(Sim3 整合 → raw PLY プロパティレベル変換: 位置/法線回転・quat 合成・log-scale シフト → voxel-hash dedup。SH rest 非回転は文書化+--dc-only。round4+5 検証: 未踏区間 21k→31k gaussians、未見視点の L1 0.268→0.252)。② 言語クエリ `0e500e8` = `query-map "car"`(CLIPSeg を transformers 経由で lazy-load → キーフレーム関連度を COLMAP 姿勢でガウシアンへ持ち上げ → クラスタ化 → 各ヒットに `navigate --goal` 用座標を付与、CLI が実行可能コマンドを出力)。e2e: "car" が先行車の走行軌跡上に集中(動的物体は移動方向に滲む=文書化)、提示ゴールへ navigate が 409 步で到達 — **言語→ゴール→自律走行のフルコンボ成立**。transformers は optional 依存(ユーザー環境にインストール済み、4.48.3)。

- **splat 内自律走行 `50a532b`(2026-06-11 夜、ユーザー「全部面白そう!」→おすすめ順の第 1 弾)**: `3dgs-robotics navigate` = 外部シミュレータなしの full closed-loop。A*(占有グリッド+ロボット半径膨張、`GridParams.trajectory_wins` で走行済みコリドーのフローターを通行可能扱い)→ pure pursuit → ユニサイクル積分 → GS カメラシムでレンダ → 3DGS localizer で fix。制御は推定値のみ参照、`--odom-noise` の車輪スリップで真値が逸れ、fix(innovation ゲート + ブレンド適用)が引き戻す構図。**ハマりどころ 2 つ**: ①勾配のある街路では「地面+一定高さ」配置でカメラが路面下に潜る → 最寄りキーフレーム高さに追従させて解決。②理想デッドレコニングだと fix のノイズが純粋な害 → wheel-slip 注入で localization の価値が成立。e2e_bag_live3(2% slip): 補正なし = 未到達・cross-track 中央値 8.3 カメラ高 / 補正あり = **614 ステップで到達・0.65 カメラ高**。残り(ユーザー承認済み、おすすめ順): ③マルチセッション地図マージ → ② 言語クエリ(open-vocabulary)。

- **言語 3 部作の完結(2026-06-11、ユーザー「1,2,3! all!」で 3 案全承認)**: ① `splat-clean "car"` `b21c2f8` = 言語プロンプトで物体除去。query-map のスコア持ち上げ → クラスタ化 → 膨張シェル(`--dilate`、透明スメアを巻き込む)→ raw PLY 行削除(map_merge の機構を再利用、他属性は無傷・ゲージ不変)。e2e_bag_live3: "car" で 12,056/173,671 gaussians・14 クラスタ除去 — **道路中央の先行車ゴーストが消えて奥の路面が見える**(kf8 レンダ比較、docs/images/robotics/splat-clean.gif)。除去は走行軌跡上に等間隔の塊=動的物体の滲み跡と整合。② `navigate --to "car"` (同コミット)= query→goal→走行のワンコマンド化。e2e: goal (1.380, 0.092) 自動解決 → **762 步で到達**(fix 3 回)。③ ビューワオーバレイ `52c328d` = `export-overlay` がゲージチェーン(gauge_transform.json)+ 凍結ビューワ正規化(リベースアンカー round から再計算)を再構築して nav 軌跡/planned path/クエリヒットを splat 座標系 JSON に投影、`splat.html?overlay=` が viewProj で 2D キャンバスに描画(ラベル・スクリーン半径付き、?refresh= と併用可)。ヘッドレス Chrome で実描画確認済み(docs/images/robotics/viewer-overlay.png)。pose-graph refinement 後の古い round の微ズレは docs に明記。全 1188 テストパス。

- **robotics デモ GIF(README 看板、同日続投)**: `scripts/build_robotics_demo_gif.py`。1 つのセッションから closed-loop を 1 GIF に合成 — 上段 = GS カメラシミュレータの仮想視点(キーフレーム間は lerp+slerp 補間でスムーズに)、下段 = nav2 occupancy grid 背景に GT 軌跡(緑)と localizer 推定(オレンジ)が蓄積。localize はキーフレーム視点のみ(補間視点は retrieval シードから遠く photometric 収束域を外れるため render only、ノードと同じ seed-distance ゲート付き)。e2e_bag_live3 で 34 フレーム / 2.7MB、中央値誤差 0.043 gauge units = 0.24 キーフレーム間隔(12 中 2 キーフレームは街路の視覚エイリアシングで ~0.47 — 正直にそのまま掲載)。README に「Robotics applications — one map, four pillars」セクション+4 本柱コマンド表を追加。**Show HN 第 2 弾用 before/after GIF はユーザー判断で不採用(2026-06-11)**。メトリックスケール対応(nav_msgs/Odometry からスケール係数 1 個を拝借する設計)は**ユーザー判断でスキップ確定(2026-06-11)** — PyPI / ZeroGPU / 告知と同様、ユーザーから話が出るまで再提案しない。

### 1.7 2026-06-11(夜): v0.2.0 リリースと次開発 4 案(保留)

- **GitHub Release v0.2.0 公開済み**(ユーザー「release しようよ」)。バージョンバンプ `ebf7b86`(pyproject.toml + `__version__`)→ タグ `v0.2.0` push → `gh release create`。ノートは v0.1.0(旧 GS Mapper、2026-03-20)以降 319 コミットの総括で、ブランド改名と robotics スタック(live mapping / 4 本柱 / 言語 3 部作 / zero-install demo)を柱にした構成。リリース後の CI / Pages はグリーン。
- **PyPI スキップの再確認と `publish.yml` の手動化 `3be92c2`**: release トリガーだった `publish.yml`(PyPI trusted publishing)がリリース作成で自動起動し失敗(pypi.org に trusted publisher 未設定、claims は `environment: MISSING`)。ユーザー「pypi は skip!」で改めてスキップ確定し、トリガーを `workflow_dispatch:` 専用に変更 — 今後のリリースで赤い失敗ランが付かない。将来公開する場合は pypi.org で trusted publisher(project `3dgs-robotics` / workflow `publish.yml`)を登録して手動実行するだけ。
- **次開発の提案済み 4 案(未着手・保留、再開時はここから選ぶ)**: ① splat-grab/paste(言語でオブジェクト切り出し→別マップへ配置、query-map + merge-maps の組み合わせ)② ブラウザ click-to-go(splat.html クリック→navigate ゴール、overlay 基盤の逆写像)③ patrol(navigate + カメラシム + detect-changes を 1 コマンドに束ねた巡回点検)④ Isaac 連携深化(USDZ に nav 経路を USD レイヤ焼き込み)。おすすめ順は ①→②→③、④はユーザーの Isaac 興味シグナルあり。

### 1.8 2026-06-11(続): MCP サーバー「Talk to Your Map」実装(§20)

- 2026-06-11 のブレストで新ネタ 5 案(MCP サーバー / アクティブマッピング `--explore` / マルチロボット・ライブマージ / rerun.io 連携 / シーンインベントリ)を提示し、ユーザー「yattekou!」でイチオシの **MCP サーバー** に着手。§1.7 の保留 4 案は引き続き保留(消えていない)。
- 実装は §20 参照。`src/gs_sim2real/robotics/mcp_server.py` + `3dgs-robotics-mcp` console script + `[mcp]` extra。テスト 13 本(mcp パッケージ非依存)を含め全 1201 テストがグリーン。実セッション(KITTI drive 0056)で `query_map("car")` → 17 ヒット + navigate 提案の実走 smoke も成功。
- **codex CLI(gpt-5.5 / xhigh)をサブエージェントとして併用**(ユーザー指示)。この環境では Ubuntu 24.04 の AppArmor 制限(`apparmor_restrict_unprivileged_userns=1`)で codex 内蔵 bwrap サンドボックスが動かず、サンドボックス解除フラグは Claude Code 側の権限分類器に拒否されるため、**「生成専用モード」**(コンテキスト抜粋を渡し、コード全文をテキスト出力させて Claude 側で適用・修正・検証)が確立した運用。今後 codex を使う際もこの方式で。

### 1.9 2026-06-11(続々): 自律探索 `explore` 実装(§21)

- ユーザー「tugi yattekou! oususumede!」でおすすめの**アクティブマッピング(フロンティア自律探索)**に着手、即日完了。§20 の MCP に続き「言語で指示(navigate --to)→ 誰も指示しない(explore)」の自律性の階段を一段上った。
- 実装は §21 参照。`robotics/splat_explore.py` + CLI `explore` + MCP ツール `explore`。テスト 12 本追加で全 1212 グリーン。KITTI drive 0056 実走: 到達可能 30,468 セルの 97.9% を 23 個の自己選択ゴールで被覆(CPU-only 約 2.5 分)、トレース PNG + 58 フレーム GIF(docs/images/robotics/explore.gif)。
- codex 生成専用モード(§1.8 と同方式)で一式生成 → レビューで 2 点修正(select_frontier の距離スケールを camera_height に、explore_gif への params 伝播)。

### 1.10 2026-06-11(夜): マルチロボット・ライブマージ `merge-live` 実装(§22)

- ユーザー「プッシュ!次!」で explore を push(`495f48f`、CI 緑)→ 残候補からおすすめの**マルチロボット・ライブマージ**に着手、即日完了。
- 実装は §22 参照。`robotics/live_merge.py` + CLI `merge-live` + MCP ツール `merge_maps` + `scripts/build_live_merge_gif.py`。テスト 6 本追加で全 1218 グリーン。
- 実走: live_demo_kitti0056(A、2,448k gaussians)× e2e_bag_live3(B、174k)— 同じ場所の独立 2 セッションがキーフレーム名を共有していたため **shared 整合(GPU 不要)が 12 キーフレームで成立、ゲージ差 scale 1.802 を吸収**して 2,566k gaussians(56k 重複除去)。`--once` 47 秒。リプレイ GIF は 10 イベント(docs/images/robotics/live-merge.gif)。
- codex 生成専用モード 3 回目。手直しは merge_preview の見栄え(外れ値で bounds 爆発 → percentile bounds、長軸の水平化、配色)のみ。

### 1.11 2026-06-11(深夜): 巡回点検 `patrol` 実装(§23)

- ユーザー「次行こう!」で保留案③の **patrol** に着手、即日完了。merge-live push(`ce20a5f`)で CI も緑復帰(explore の format 修正 `aee6a33` 込み、ユーザー「プッシュ」)。
- 実装は §23 参照。`robotics/patrol.py` + CLI `patrol` + MCP ツール。waypoint 源 4 種(xy / keyframe / 言語 / **changes.json = 変化点へ見に行く**)。テスト 10 本追加で全 1228 グリーン。
- 実走(e2e_bag_live3、round5 vs 4 の changes): 全 119 クラスタ巡回 22 分(78/119 到達、到達不能は正直にスキップ記録)→ `--max-stops` を追加し大クラスタ優先 8 ストップ + `--return-to-start` で 97 秒、7/9 到達、視点レンダ 8 枚 + スライドショー GIF。トレース = docs/images/robotics/patrol-trace.png。
- codex 生成専用モード 4 回目。手直しは mkdir 1 箇所と `--max-stops` の後付けのみ。

### 1.12 2026-06-11(深夜続): アクティブマッピング(explore v2)実装(§24)

- ユーザー「順にやっていこう!」で残ネタ 3 案(explore v2 / splat-grab/paste / Isaac 深化)を順送りに。第 1 弾の **explore v2 = 本物のアクティブマッピング** を実装、patrol push(`c459148`)で CI 緑継続。
- 実装は §24 参照。`robotics/active_mapping.py` + `scripts/run_active_mapping_demo.py` + `LiveMappingSession.build_pending_round()`(本体追加 10 行)。テスト 7 本追加で全 1235 グリーン。
- KITTI 0056 実走(VGGT): 15 フレームのブートストラップ地図 2.09M gaussians → **ロボットが選んだ 6 ラウンドで 3.05M に成長**、追跡した全フロンティアが許容内に地図化(frontier_distance 0.05〜0.14 vs 閾値 0.08〜0.20)。GIF = docs/images/robotics/active-mapping.gif。
- codex 生成専用モード 5 回目。レビューで実害バグ 2 件を修正: ① codex が API 不確実時に入れた多重シグネチャ試行ハック 90 行 → 直接呼び出し化 ② **成長判定の「centers 追記」前提**(round はストライド再サンプルするので破綻)→「次 round の最近傍キーフレーム距離 ≤ 閾値」に修正。さらに設計バグ ③ inflate_obstacles が unknown も膨張させるためマップフロンティアが原理的に空 → フロンティアを到達圏近傍に再定義。

### 1.13 2026-06-11(深夜続々): splat-grab/paste 実装(§25)

- 順送り 3 案の第 2 弾。実装は §25 参照。`robotics/splat_grab.py` + CLI `splat-grab`/`splat-paste` + MCP ツール 2 本(計 11 ツール)。テスト 8 本追加で全 1244 グリーン。
- 実走: live_demo_kitti0056 から `splat-grab "car"` → 当初 19 クラスタ 219k gaussians が「ドライブ全長の先行車ゴースト」をまとめて掴む問題が発覚 → **best-cluster 既定動作を追加**(ベストヒット最近傍の連結成分のみ、`--all-clusters` で無効化)→ 1 台分 21.8k gaussians。`splat-paste --at 1.0,0.05` で e2e_bag_live3 へ **ゲージスケール 0.408 自動適用 + 接地配置**。プレビュー = docs/images/robotics/splat-grab-paste.png。
- codex 生成専用モード 6 回目。今回も防衛的シェイプ探りヘルパ 2 本を直接アクセスに置換(実 API は確定済みのため)。

### 1.14 2026-06-12(未明): Isaac ルートレイヤ実装(§26)— 順送り 3 案コンプリート

- 順送り 3 案(explore v2 → splat-grab/paste → Isaac 深化)の最終弾。実装は §26 参照。`robotics/isaac_route.py` + CLI `export-isaac-route` + MCP ツール(計 12 ツール)。テスト 5 本追加で全 1248 グリーン。
- 実走: e2e_bag_live3 で nav(クリーン走行 598 步到達)→ export-isaac(USDZ 20.5MB 再生成)→ route.usda。**pxr 読み戻しで USDZ 参照が解決され、NuRec Volume(OmniNuRecFieldAsset)とルート(BasisCurves 2 本 + goal Sphere)が同一ステージに合成されることを確認**。Isaac Sim 本体でのレンダ確認はユーザー側(docs に明記)。
- codex 生成専用モード 7 回目。手直し: pxr API 1 件(SetTypeAttr→CreateTypeAttr)、未使用 import/変数、テストの文字列 [-1] バグ。

### 1.15 2026-06-12: シーンインベントリ実装(§27)

- ユーザー「やってこう!」で残ネタ 3 案(インベントリ → rerun.io → click-to-go)に着手。第 1 弾の **inventory** 完了。Isaac route の CI 赤(usd-core 不在で suffix テストが ImportError)は検証順入替 `62f0de7` で即修正済み。
- 実装は §27 参照。`robotics/inventory.py` + CLI `inventory` + MCP ツール(計 13 ツール)。テスト 5 本追加で全 1253 グリーン。
- KITTI 0056 実走: デフォルト 8 語彙で **74 クラスタ / 5 カテゴリ**(tree 43 / car 17 / pole 9 / bush 4 / building 1、traffic sign・fence・person は not found)を 2 分 20 秒。素材 = docs/images/robotics/inventory.png。
- codex 生成専用モード 8 回目。手直しゼロ(deviations: none、統合時の整形のみ)。

### 1.16 2026-06-12(続): rerun.io 連携実装(§28)

- 残ネタ第 2 弾。実装は §28 参照。`robotics/rerun_bridge.py` + CLI `rerun-replay` + MCP ツール(計 14 ツール)+ `[rerun]` extra。テスト 7 本追加で全 1260 グリーン。
- 実走: e2e_bag_live3 → `session.rrd`(18MB、5 ラウンド / 885k 点 / nav 経路込み)を 8.6 秒で生成、`rerun rrd stats` で 5 エンティティパス・チャンク構成を検証。
- codex 生成専用モード 9 回目。手直しはテストの誤 import 1 行のみ。

### 1.17 2026-06-12(続々): click-to-go 実装(§29)— 残ネタ 3 案コンプリート

- 残ネタ最終弾。実装は §29 参照。`robotics/click_to_go.py`(HTTP サーバ + レイ→ゴール変換)+ `docs/splat-viewer/main.js` パッチ(初のフロントエンド改修、こちらは Claude 直書き)+ console script `3dgs-robotics-click-to-go`。テスト 5 本追加で全 1265 グリーン。
- 実走: e2e_bag_live3 で合成レイ POST → ゴール (1.18, 0.095) 復元 → navigate 350 步到達 → overlay 配信の全チェーン成立。headless Chrome(SwiftShader)で splat + クリック UI + 走行経路オーバレイの実描画を確認(素材 = docs/images/robotics/click-to-go.png)。
- codex 手直し 2 件: クラス属性に裸の関数を入れて self が束縛される記述子バグ(staticmethod 化)、main() の partial 越し属性アクセス(server.scene_frame に付け替え)。

## 2. 現在の主戦場

今の大きな方向転換は、単なる「屋外 3DGS のデモ生成」から、次のような **Dynamic Map Viewer + Physical AI 用 simulation / evaluation environment** に寄せることです。

1. Real-world robotics logs から 3DGS scene を作る。
2. Large-scale 3DGS training output を browser-ready tile catalog / viewer PLY / route playback に promotion する。
3. Dynamic Map Viewer で resident / preload / evicted tile を可視化し、広域 map loading を live demo として見せる。
4. Browser / local renderer / headless environment で観測を返す。
5. Route policy / navigation policy / query policy を benchmark する。
6. Scenario matrix を小さな shard に分け、CI で回す。
7. CI workflow 自体も生成、検証、activation、review publishing の段階に分ける。
8. Review bundle を GitHub Pages に出し、workflow trigger を広げる前に人間が inspected artifact を見られるようにする。
9. Promotion report で PR / branch trigger へ広げてよいかを記録し、trigger-enabled workflow の adoption を分離する。

この構成にした理由は、開発がスケールすると「1 個の巨大 E2E が落ちる」よりも、「小さい scenario / shard / validation / activation / review gate がどこで落ちたか分かる」方が速いからです。ユーザーが求めていた「モジュール分割、関数分割、クラス分割、依存の局所化、テスト単位の分離」「影響範囲を閉じ込め、検証単位を細かく設計する」は、この route policy scenario CI chain の設計方針そのものです。

2026-06-09 の実務上の主戦場は、まず Dynamic Map Viewer runtime です。Istanbul pilot の docs media と assets は merged したので、次は live Pages で deploy された状態を確認し、PlayCanvas GSplat 表示が browser pixels として成立しているかを潰す。そこが固まってから real-input 3DGS の範囲を広げる。Physical AI scenario CI は引き続き中核だが、しばらくは「見える dynamic map」を信頼できる基盤にすることが優先です。

2026-06-10 時点では、これに **star growth track** が並走している(§1.3 / §1.4 / §18 / §19)。zero-install demo(HF Space / Colab)、ROS 2 live mapping、実データ growth GIF、README 整理に加え、同日午後に §18 の 3 項目(動画ワンライナー / VGGT backend / 3DGS localization)も完了した(§1.4)。告知はユーザーが実行する。開発側の次の弾は **§19 の第 3 期ロードマップ**(rosbag 直接入力 → ループクロージャ、目標 100 stars / 年内)。Dynamic Map Viewer runtime の信頼性タスク(§1.2 の DoD)は無効になったわけではなく、star growth track と交互に進める。どちらを優先するかはユーザーのその日の指示に従う。

## 3. Recent Commits / 現在地

直近の主な流れは 3 層ある。最上層が 2026-06-10 の star growth chain、その下に 2026-06-08〜09 の Dynamic Map Viewer / large-scale 3DGS promotion、さらに下に 2026-04〜05 の Physical AI scenario CI chain が残っている。

最新の star growth chain(2026-06-10、main 直 push、全て merged):

| Commit | 内容 |
| --- | --- |
| `0fe34b1` | README link noise 削減(67 → 35 links)。badge 4 つ、Try-it-first 3 リンク、file-path リンクの plain code path 化。`test_readme_first_view_surfaces_demo_and_review_entrypoints` を追従。 |
| `1b6b15b` | 実データ「map grows as the robot drives」GIF。KITTI drive 0056 を live mapping replay → `scripts/build_live_mapping_gif.py`(round 間 gauge chaining + 真上 ortho 合成)→ `docs/images/live-mapping/live-mapping-grow.gif`。CPU tests 4 本追加。 |
| `1c0f54a` | lead GIF を復元カメラ軌跡追従の dynamic-map loading window 表現に。 |
| `6abbb2a` / `09835be` | lead GIF を Bag6 pilot の実 ortho render ベースの空撮地図へ刷新(2 段階)。 |
| `91c36f6` / `e3684a5` / `8bae7ff` | HF Space build 修正 3 連(gradio 5.49.1 / Python 3.10 pin / torch 2.4.1 + gsplat pt24cu121 wheel 組み合わせ)→ Space RUNNING。 |
| `1cbdcff` | README で zero-install demos と live mapping を前面に。 |
| `104559a` | ROS 2 live 3DGS mapping(`3dgs-robotics-live-mapper`、rclpy-free core、`live/latest.splat` atomic swap、polling viewer)。 |
| `d63497c` | zero-install photos-to-splat demos(HF Space `apps/hf-space/` + Colab)。 |
| `eeeb9ec` | pip install 環境で training config のビルトインデフォルトへフォールバック。 |
| `163209f` | 本 plan の前回 refresh。 |

合わせて GitHub About を `gh repo edit` で「Photos and robot logs to browser-ready 3D Gaussian Splat maps」へ更新済み(commit 外の作業)。

その下の Dynamic Map Viewer / large-scale 3DGS chain:

| Commit / PR | 内容 |
| --- | --- |
| `cf7a22e` / PR #192 | Istanbul Dynamic Map Viewer media を README / Pages / docs に追加。6 transformed viewer PLY tiles、GIF / still、Istanbul catalog / route URL を public docs に反映。 |
| `ae9a783` / PR #191 | Istanbul Bag6 3DGS pilot assets を追加。real rosbag2 capture 由来の tile catalog / route / browser assets を staged。 |
| `460746b` / PR #190 | Real 3DGS input gate runbook を追加。real rosbag2 入力を large-scale 3DGS pilot に入れる前の確認手順を文書化。 |
| `a62f793` / PR #189 | promoted grid route metadata を追加。large-scale tile catalog と route playback の metadata を viewer / docs で扱いやすくした。 |
| `784abea` / PR #188 | 3DGS viewer asset promotion を追加。large-scale training output を browser viewer 用 assets / catalog に promotion する基礎を追加。 |

この chain の到達点:

- README の "Large-scale 3DGS Dynamic Map Result" と "Real Rosbag2 3DGS Pilot Result" が分離された。
- `apps/dreamwalker-web/public/manifests/istanbul-bag6-pilot-tile-catalog.json` が viewer PLY / source splat 両方の metrics を持つ。
- `apps/dreamwalker-web/public/robot-routes/istanbul-bag6-pilot-route.json` は viewer X/Z 座標で route playback できる。
- `scripts/build_istanbul_3dgs_media.py` が Istanbul result GIF / still を生成する。
- `src/gs_sim2real/train/large_scale_3dgs.py` が source chunk PLY を探し、viewer axes へ transform して catalog/report に viewer metrics を出す。
- `apps/dreamwalker-web/tools/validate-dynamic-map-catalog.mjs` は viewer fields と route coverage を検証する。
- validation snapshot: catalog validator pass、dynamic-map-loading tests 17/17 pass、dynamic-map-catalog tests 13/13 pass、`tests/test_large_scale_3dgs.py` 23 pass、Vite build pass、`git diff --check` clean。

Historical Physical AI / scenario CI chain (2026-04-25 〜 26 の Tier 2 chain):

| Commit | 内容 |
| --- | --- |
| _PR #137_ | MCD Profile 3 (`ntu_day02_multi_3cam_300each_ba`) を 2-camera (d455b + d435i) に redefine。`/d455t/color/image_raw` は MCDVIRAL ATV に存在しない topic と判明したため `multi_2cam_300each_ba` に rename。`mcd_quality_plan.py` / `test_mcd_quality_plan.py` / `plan_outdoor_gs.md` を更新。 |
| `2262f22` | Per-window correlation stats (mean/p95/max/heading + bag-time span) を review bundle の Markdown / HTML に surface。 |
| `95f1ea4` | `pair_distribution_strata_mode` で `equal-pair-count` を選べるように。スパース bag でも各 window を統計的に成立させる。 |
| `cef2659` | aggregate-statistic (mean/p95/max/heading-mean) を per-window 評価に切り替え可能に。stratified 時は aggregate tag を suppress。 |
| `2abd640` | `pair_distribution_strata` で per-pair 分布ゲートを N 等分時間 window に分けて評価。 |
| `4629835` | heading 版 per-pair 分布ゲートを追加。heading-bearing subset を分母に使う。 |
| `0683f91` | translation per-pair 分布ゲート (`max_pair_translation_error_meters` + fraction) を追加。 |
| `3762717` | per-bag-topic correlation threshold overrides を `--correlation-thresholds-config` JSON で受け付ける。 |
| `6db678e` | `score_trajectory` に per-step peer cache を threading。hypothetical trajectory でも policy obstacle が peer を見える。 |
| `5a6edfd` | correlation regression gate (mean/p95/max/heading-mean) を `3dgs-robotics route-policy-scenario-ci-review` に追加、`--fail-on-review` で exit 2。 |
| `ab8fbbe` | scenario CI review bundle (Markdown + HTML) に Real-vs-sim correlation セクションを追加。 |
| `ea1b5f8` | `HeadlessPhysicalAIEnvironment.query_collision` に per-step peer cache を threading。 |
| `a038b61` | `RoutePolicyGymAdapter` に peer-aware obstacle features (`peer-min-separation-meters` 等) + `step_positions` 解決を導入。 |
| `9bb15d2` | `RoutePolicyGymAdapter` の feature dict に IMU 7 軸 (step_dt + ang_vel + lin_acc) を出力。 |
| `b40e4a3` | scenario-set run report に correlation reports を attach (`--correlation-report`)、Markdown サマリ surface。 |
| `5195130` | rosbag correlator に IMU orientation merge を追加 (`merge_navsat_with_imu_orientation`)。 |
| `9e3be8b` | `ObstaclePolicy` protocol + 4 reference impl (Waypoint / Chase / Flee / MaintainSeparation) を導入。 |
| `8bf29b1` | env に IMU kinematic finite-diff renderer を baked-in。`imu-proxy` sensor が `ready-via-kinematic-finite-diff`。 |
| `7127641` | real-vs-sim rosbag correlation library + CLI (`scripts/run_rosbag_correlation.py`) を新設。 |

この chain で **correlation gate plumbing** と **multi-agent obstacle plumbing** が両方とも production rollout に使える状態に到達した。

2026-05-16〜17 session で Sprint 1〜4 + Sprint 4 follow-up step 1(per-step min-peer-separation 配線)まで完走、`origin/main` まで push 済み(`main @ 13e3b56`):

| Commit | Sprint | 内容 |
| --- | --- | --- |
| `13e3b56` | 4 follow-up (step 1) | `_run_scenario` が rollout 中の `nearest-dynamic-obstacle-distance-meters` を walk して **最小値** を `min-peer-separation-meters` として `interactionMetricsValues` に書き出す。shard merge aggregator → review bundle → Pages index まで実値が流通(2026-05-17 セッションで sample bundle の `min-peer-separation-meters` ≈ 1.95m を確認)。新規 helper: `policy_scenario_set.py:_min_peer_separation_from_report()` / `NEAREST_PEER_CLEARANCE_FEATURE_KEY`。 |
| `99a3226` | doc | plan_outdoor_gs.md refresh(Sprint 1〜4 完走後の状態反映、§17.1 priority 表更新、§17.5 各 PR の commit hash 埋め込み)。 |
| `b3784be` | 4 (PR D6) | 2-agent crossing fixture を smoke chain へ landed。`_run_scenario` が `interactionMetricsValues` を書き出して D4 集計まで貫通。公開 sample bundle (`docs/reviews/smoke-route-policy-ci/`) も regenerate されて `multiAgent: true` に。 |
| `9981ed0` | 4 (PR D5) | Review bundle JSON / Markdown / HTML に "Multi-agent" pill + interaction-metrics block を追加。Pages index に `multiAgent` / `multiAgentCount`。 |
| `cb38a23` | 4 (PR D4) | Shard merge に `InteractionMetricsAggregate`(per-key mean/p95/max/sampleCount)を attach。値が無いとき `None`、JSON 出力もスキップ。 |
| `8ae08c6` | 4 (PR D3) | `synthesize_peer_roster_from_scenario_metadata()` 純粋関数 + `_run_scenario` 配線。`agents` / `population` から `DynamicObstacleTimeline` を deterministic に合成。 |
| `f84b90e` | 4 (PR D2) | `RoutePolicyMatrixSceneSpec` に optional `agents` / `population` / `interaction_metrics`。expander が `population.seed_count` で seed fan-out。 |
| `2e8e738` | 4 (PR D) | `AgentRoleSpec` / `PopulationSpec` / `InteractionMetricsSpec` の 3 record + JSON roundtrip + legacy ego-only fallback test。 |
| `5bca467` | 4 (設計) | §17.5 を multi-agent Tier 3 design draft に拡張(schema additions / hook points / risks / PR breakdown)。 |
| `e9d50a9` | 1 (PR A2 scaffold) | `scripts/publish_production_review_bundle.py` — 外部生成の production review.json を `docs/reviews/<id>/` に publish + index 再生成する thin wrapper(`provenance.kind!=production` は exit 2)。 |
| `be5c655` | housekeeping | `/*_handoff.md` を `.gitignore` に追加(point-in-time consultation snapshot を commit せず手元保持)。 |
| `3d28c31` | CI 自動化 | `nightly.yml`(scheduled + workflow_dispatch で smoke chain → `scenario-ci-bundles` artifact upload)+ `scenario-ci-promote.yml`(workflow_dispatch で promotion gate を回す)。 |
| `db0cb56` | 3 (PR C6) | Live trace emitter の cross-path integration tests(Gym + replanning + direct rollout が同 emitter / JSONL を共有しても整合)。 |
| `f183229` | 3 (PR C5) | Closed-loop replanning helpers に emitter pass-through。`replan_after_blocked_rollout` / `rollout_route_with_replanning` の kwargs に `trace_emitter` 等。 |
| `5f9cc99` | 3 (PR C4) | `rollout_route` (非 Gym 経路)に emitter wiring。`record_step` per-segment、最終 segment で `goal_reached` / `collision` terminal。 |
| `a98dac0` | 3 (PR C3) | baseline benchmark runner に per-policy emitter factory。複数 policy を同 run で評価しても episode_id が衝突しない。 |
| `54ecff6` | 3 (PR C2) | `RoutePolicyTraceEmitter` 状態機械 + `JsonlPolicyTraceEventStream` + Gym adapter hook。live 用 schema。 |
| `ace6143` | 3 (PR C) | `PolicyTraceEvent` 基本モジュール + post-hoc dataset → trace event 抽出 + JSONL ⇔ `CorrelationEventWindow` 変換。 |
| `2b79b5d` | 2 (PR B) | event-aligned correlation stratification。`pair_distribution_strata_mode="event-aligned"` で外部 event window 軸を受理。 |
| `090ee16` | 1 (PR A) | first-class `RoutePolicyScenarioCIReviewProvenance` 追加。CLI に `--kind` 系フラグ、Pages index に kind 列。 |

PR A2 だけは production benchmark データが届いていないため scaffold のみ、それ以外は実装完走。

### 3.1 Handoff snapshot

- 基準にする repository state は `main @ cf7a22e`(PR #192 squash merge、2026-06-09)。`origin/main` と同期済み。PR branch `codex/istanbul-viewer-result-media` は merge 後に remote delete 済み。
- PR #192 merge 直後の working tree は clean。`git stash list` も空。以後の doc refresh ではこのファイルの差分だけが発生する想定。
- GitHub connector token は前セッションで expired していたが、`gh` CLI は利用可能。PR 作成 / merge / `gh pr view` / Actions 確認は CLI fallback で進める。
- PR #192 local validation snapshot:
  - `node apps/dreamwalker-web/tools/validate-dynamic-map-catalog.mjs apps/dreamwalker-web/public/manifests/istanbul-bag6-pilot-tile-catalog.json --public-root apps/dreamwalker-web/public --site-url http://localhost:5173/ --preload-mode metadata --route apps/dreamwalker-web/public/robot-routes/istanbul-bag6-pilot-route.json --route-playback 1 --route-playback-ms 1200 --route-playback-loop 1` => pass、sparse rectangular grid warning 1 件のみ。
  - `npm --prefix apps/dreamwalker-web run test:dynamic-map-loading` => 17/17 pass。
  - `npm --prefix apps/dreamwalker-web run test:dynamic-map-catalog` => 13/13 pass。
  - `PYTHONPATH=src pytest tests/test_large_scale_3dgs.py` => 23 pass。
  - `npm --prefix apps/dreamwalker-web run build` => pass、Vite chunk-size warning のみ。
  - `git diff --check` => clean。
- Local dev smoke では catalog と first Istanbul PLY が HTTP 200 で配信されることを確認済み。ただし live PlayCanvas GSplat 表示については、「assets load / manager sees splats」までは確認済みで、headless canvas pixel としての visible splats はまだ完全には証明できていない。
- MCD quality / Physical AI chain の古い状態も残る。Tier 1 MCD rerun (`scripts/plan_mcd_quality_runs.py`) は 2/3 profile (`single_400_depth_long` L1=0.1951 / `single_800_ba` L1=0.2699) が gate pass。Profile 3 は `multi_2cam_300each_ba` (d455b + d435i) に redefine 済み。`ntu_day_02_d435i.bag` は取得 / topic 検証済みで、残課題は GPU 実走と gate report 更新。
- 次に触る candidate(優先順位順):
  1. **GitHub Pages 反映確認**: PR #192 merge 後の Pages deploy status を `gh run list` / Actions / live URL で確認。Istanbul GIF / still / catalog / PLY / `/dreamwalker/` pilot URL が hosted で 200 になることを smoke する。
  2. **Live PlayCanvas GSplat 表示の実証**: blank canvas を潰す。asset format、PlayCanvas GSplat loader、camera bounds、near/far clip、scale、material visibility を順に切り分け、headed screenshot と pixel check を残す。
  3. **Runtime capture pipeline**: `docs/images/istanbul-bag6-pilot/` の GIF / still を runtime viewer 由来で再生成できるようにする。proxy renderer は fallback とし、README media と live demo のズレを減らす。
  4. **Real large-scale expansion**: Istanbul pilot を 6 tiles から、より長い route / more frames / more tiles へ拡張。`large-scale` と呼べる spatial coverage を実データで作る。
  5. **Pairwise clearance histogram の実値化(§17.7)**: Dynamic Map Viewer の runtime blocker が解けたら Physical AI scenario CI 側へ戻る。`InteractionMetricsSpec.pairwise_clearance_histogram_bins` の runtime / aggregator / review surface はまだ未実装。
  6. **PR A2 production data**: production benchmark run の review.json が届いたら `scripts/publish_production_review_bundle.py` で `docs/reviews/<id>/` 公開。Pages index で `productionCount: 1` になる(data-blocked)。
- 反すべきでない方向:
  - Istanbul pilot を「完成した production-scale map」と言い切らない。現状は real-input pilot であり、large-scale fixture としては synthetic 87-tile mosaic と real 6-tile pilot を分けて説明する。
  - Hero / README に lanelet2 vector layer や説明文を載せすぎない。Dynamic Map Viewer の first visual は 3DGS footprint、resident / preload、route playback に絞る。
  - Generated media だけで live viewer の完成を主張しない。runtime renderer の visible proof を取るまでは、asset/catalog/route/media は pass、live 3DGS rendering は next blocker と書く。
  - raw rosbags、MCD calibration YAML、training output directory は commit しない。commit するのは public viewer assets、catalog、route、docs media、reproducible scripts / tests。
  - `src/gs_sim2real/datasets/mcd.py` の `DEFAULT_IMAGE_TOPICS` / `DEFAULT_IMU_TOPICS` から `/d455t/*` を削るのは scope 外。tolerant catalog として残しておく方が test_cli / test_mcd の synthetic-bag fixture を壊さない。
  - Profile 3 の `requires_full_folder=True` は「single d455b バッグ以外も要る」というヒントとして残す。リテラルに「14.8 GB 全部 download せよ」という意味ではない。

### 3.2 MCDVIRAL spec audit recipe (2026-04-26 d455t finding)

「Profile 3 が data-blocked」だと思われていたが、実際は **upstream に存在しない topic 名** (`/d455t/color/image_raw`) を含んでいたという spec ミスだった。同種の罠を避けるため、MCDVIRAL profile を新規追加 / 改修する際の verification recipe を以下に固定化する。

1. **Download page で session row を見る** — https://mcdviral.github.io/Download.html を `curl -sL ... > /tmp/mcd.html` で取得。各 NTU / KTH / TUHH session の row には `<a href="https://drive.google.com/file/d/<ID>" ...>d435i<br />(4.7 GB)</a>` のような per-bag リンクがある。提供されている camera は **d435i と d455b の 2 つだけ** (color)。`d455t` という camera は MCDVIRAL ATV / handheld rig どちらにも存在しない。
2. **Calibration YAML と交差検査** — `scripts/download_mcd_calibration.sh atv` で `data/mcd/calibration_atv.yaml` を落とし、`body:` 配下の sensor 名を確認。Profile が指す topic 名は必ずこの YAML に対応する extrinsic がある (`d455b_color`, `d455b_imu`, `d455b_infra1`, `d455b_infra2`, `d435i_imu`, `d435i_infra1`, `d435i_infra2`, `os_*`, `vn100_imu`, `vn200_imu`, `ltpb_tag*`, `mid70`)。`d455t_*` は無い。
3. **rosbag を直接覗く** (1 本でも download し終わったら) — 期待 topic が実 bag に居るか必ず確認。

   ```python
   from rosbags.highlevel import AnyReader
   from pathlib import Path
   with AnyReader([Path("data/mcd/ntu_day_02/ntu_day_02_d435i.bag")]) as reader:
       for t in sorted({c.topic for c in reader.connections}):
           print(t)
   ```

   `/d435i/color/image_raw` のような期待 topic がここに無ければ、`MCDQualityRunProfile.image_topics` 側 (= profile の spec) を直すのが先。Download し直しても解決しない。
4. **profile を組み立てる順序** — (a) MCDVIRAL の per-bag size を見て GPU + 帯域コストを試算、(b) calibration YAML の sensor list で extrinsic の有無を確認、(c) test bag を一本落として `AnyReader` で topic 列挙、(d) `MCDQualityRunProfile` の `image_topics` / `camera_frame` を埋める、(e) `tests/test_mcd_quality_plan.py` で構造 assert を追加。順序を守れば「download した後で topic が無いと判明」が起きない。

この recipe は memory にも `project_mcdviral_atv_cameras.md` として固定化済み (next session 起動時に自動で参照される)。

## 4. System Map

### 4.1 層構造

| Layer | 目的 | 主な files |
| --- | --- | --- |
| Data / assets | public demo assets、scene manifests、Pages viewer | `docs/scenes-list.json`, `docs/sim-scenes.json`, `docs/assets/outdoor-demo/`, `docs/splat.html`, `docs/index.html`, `docs/images/` |
| Preprocess | image / video / rosbag / external SLAM artifact から COLMAP sparse 相当を作る | `src/gs_sim2real/datasets/`, `src/gs_sim2real/preprocess/`, `src/gs_sim2real/preprocess/external_slam_artifacts/` |
| Train / export | gsplat / nerfstudio training、`.splat` / scene bundle export | `src/gs_sim2real/train/`, `src/gs_sim2real/viewer/web_export.py`, `src/gs_sim2real/cli.py` |
| Dynamic map viewer | large-scale 3DGS tile catalog、viewer PLY promotion、route playback、resident/preload UI | `apps/dreamwalker-web/src/dynamic-map-loading.js`, `apps/dreamwalker-web/src/App.jsx`, `apps/dreamwalker-web/public/manifests/`, `apps/dreamwalker-web/public/robot-routes/`, `apps/dreamwalker-web/tools/validate-dynamic-map-catalog.mjs` |
| Physical AI sim contract | scene environment、sensor rig、headless env、observations/actions | `src/gs_sim2real/sim/contract.py`, `interfaces.py`, `headless.py`, `gym_adapter.py`, `occupancy.py`, `costmap.py` |
| Policy benchmark | dataset、imitation、registry、benchmark、history gates | `policy_dataset.py`, `policy_imitation.py`, `policy_benchmark.py`, `policy_benchmark_history.py` |
| Scenario execution | scenario-set、matrix expansion、sharding、merge | `policy_scenario_set.py`, `policy_scenario_matrix.py`, `policy_scenario_sharding.py` |
| Scenario CI | CI manifest、workflow materialization、validation、activation、review publishing | `policy_scenario_ci_manifest.py`, `policy_scenario_ci_workflow.py`, `policy_scenario_ci_activation.py`, `policy_scenario_ci_review.py` |
| Experiment labs | design seams の比較実験と docs 生成 | `src/gs_sim2real/experiments/`, `docs/experiments.md`, `docs/experiments.generated.md` |

### 4.2 分割の基本方針

- 外部依存の重い front-end は repo 内で import しない。MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR は「artifact を吐いた後」に importer が受ける。
- Generated artifact は必ず versioned JSON / Markdown / HTML のどれかにする。
- CI workflow は手書きではなく manifest から生成する。
- Generated workflow はすぐ active path に置かず、validation → activation → review publishing を通す。
- Physical AI benchmark は single huge run にせず、scenario-set → matrix → shard → merge に分ける。
- Public Pages に出すものは `docs/` 配下だけ。実データ、rosbag、calibration YAML、training output は commit しない。
- Dynamic Map Viewer public assets は `apps/dreamwalker-web/public/` に置く。GitHub Pages build で `/dreamwalker/` 配下から配信されるので、catalog / route / splat path は Pages base path と local dev の両方で成立するように検証する。
- Large-scale 3DGS promotion は training coordinate と viewer coordinate を混同しない。source metrics と viewer metrics は catalog/report で別 field として持ち、runtime は viewer field があればそちらを優先する。

## 5. Production Assets / Viewer Contract

`docs/scenes-list.json` の production scene list:

1. `assets/outdoor-demo/outdoor-demo.splat` — Autoware 6-bag supervised default
2. `assets/outdoor-demo/outdoor-demo-dust3r.splat` — bag6 DUSt3R pose-free
3. `assets/outdoor-demo/mcd-tuhh-day04.splat` — MCD `tuhh_day_04` DUSt3R pose-free
4. `assets/outdoor-demo/bag6-mast3r.splat` — bag6 MAST3R pose-free metric
5. `assets/outdoor-demo/bag6-vggt-slam-20-15k.splat` — bag6 VGGT-SLAM 2.0 comparison
6. `assets/outdoor-demo/bag6-mast3r-slam-20-15k.splat` — bag6 MASt3R-SLAM comparison
7. `assets/outdoor-demo/bag6-pi3x-20-15k.splat` — bag6 Pi3X comparison
8. `assets/outdoor-demo/mcd-tuhh-day04-mast3r.splat` — MCD `tuhh_day_04` MAST3R pose-free metric
9. `assets/outdoor-demo/mcd-ntu-day02-supervised.splat` — MCD `ntu_day_02` supervised valid-GNSS demo

重要:

- `assets/outdoor-demo/mcd-tuhh-day04-supervised.splat` は diagnostic artifact として存在してもよいが、production picker / benchmark table に追加しない。
- production scene を増やしたら、README table、viewer picker 3 種、preview PNG、hero GIF の source of truth は `docs/scenes-list.json` に揃える。
- Drift は `tests/test_pages_assets.py` が検出する。

Dynamic Map Viewer は上記 9 scene list とは別の contract を持つ。source of truth は `apps/dreamwalker-web/public/manifests/*-tile-catalog.json` と `apps/dreamwalker-web/public/robot-routes/*.json`。ここでは 1 scene = 1 `.splat` ではなく、route playback に沿って複数 tile を resident / preload / evict する。

現行 public dynamic-map catalogs:

| Catalog | Type | 状態 |
| --- | --- | --- |
| `outdoor-production-grid-large-tile-catalog.json` | synthetic regional mosaic | 9 production `.splat` results を 25 placement / 87 route tiles に展開。dynamic loading UI と route playback の大規模 fixture。 |
| `istanbul-bag6-pilot-tile-catalog.json` | real rosbag2 pilot | Istanbul `rosbag2` 由来。6 ready tiles、12.5 MB browser `.splat`、6 viewer PLY、438,796 Gaussians / 103.8 MiB。 |

Tile catalog contract:

- `tiles[].splatUrl` は browser `.splat` tile の public URL。lightweight dynamic-map loading / metadata validation に使う。
- `tiles[].viewerSplatUrl` がある場合、Dynamic Map Viewer runtime は viewer-ready PLY として優先する。
- `coreBounds` / `expandedBounds` は source tile bounds。`viewerCoreBounds` / `viewerExpandedBounds` がある場合、route coverage と overview camera は viewer bounds を優先する。
- `tiling.viewerAxes` が `xz` の場合、viewer ground plane は X/Z。training source が XY でも runtime には X/Z として渡す。
- `viewerSplatBytes` / `viewerGaussianCount` は source `.splat` metrics と混ぜない。README では「browser `.splat` size」と「viewer PLY Gaussians / MiB」を別行で説明する。

Dynamic Map Viewer の validation baseline:

```bash
node apps/dreamwalker-web/tools/validate-dynamic-map-catalog.mjs \
  apps/dreamwalker-web/public/manifests/istanbul-bag6-pilot-tile-catalog.json \
  --public-root apps/dreamwalker-web/public \
  --site-url http://localhost:5173/ \
  --preload-mode metadata \
  --route apps/dreamwalker-web/public/robot-routes/istanbul-bag6-pilot-route.json \
  --route-playback 1 \
  --route-playback-ms 1200 \
  --route-playback-loop 1
```

この validator が通ることは「catalog / route / asset path が整合している」という意味。PlayCanvas canvas 上に visible splat が描けていることの証明ではない。runtime rendering は headed browser smoke / screenshot / pixel check で別途見る。

## 6. Outdoor 3DGS Track

### 6.1 目的

屋外 robotics data から 3DGS を作り、Pages viewer で公開できる `.splat` / bundle にする。

### 6.2 対応済み

- Autoware bag 系 supervised pipeline。
- DUSt3R / MASt3R pose-free preprocessing。
- MCD rosbag image / lidar / IMU / GNSS extraction。
- MCD static calibration downloader と TF handling。
- MCD GNSS zero guard。
- MCD CameraInfo 欠落時の PINHOLE 合成。
- MCD single-camera colorize / sparse depth export。
- IMU orientation CSV normalization。
- Angular-velocity yaw fallback。
- External SLAM artifact import facade。
- VGGT-SLAM 2.0 / MASt3R-SLAM / Pi3X comparison splat 実走。
- Pi3 / LoGeR profile / resolver candidate patterns。
- Pi3 / LoGeR smoke は archive に記録済み。
- README / Pages launch-kit / docs assets 整理。
- Large-scale 3DGS promotion の browser tile catalog / route metadata / viewer PLY transform。
- Dynamic Map Viewer の outdoor-production 87-tile synthetic regional mosaic。
- Istanbul `rosbag2` real-input pilot の 6-tile catalog / route / README media / docs landing integration。

### 6.3 未完

| Priority | Task | 状態 |
| --- | --- | --- |
| A | GitHub Pages PR #192 deploy smoke | merge 済み。次に Actions / hosted URL で README、docs landing、Istanbul GIF / still、catalog、PLY、`/dreamwalker/` URL を確認する。 |
| A | PlayCanvas live GSplat visible proof | catalog / route / assets は pass。headless では live canvas rendering の visible proof が弱い。headed screenshot + pixel check で renderer path を詰める。 |
| A | Runtime capture pipeline | Istanbul GIF / still は生成済み。次は actual viewer runtime 由来で再生成し、proxy render 依存を減らす。 |
| A | Real large-scale Istanbul expansion | 現状 6 tiles / 438,796 Gaussians は pilot。より長い route / more frames / more tiles に増やして、real large-scale と呼べる coverage にする。 |
| A | BYO photos / CoVLA mini 自己実証 | 外部入力待ち。ユーザ写真または HF access 承認が必要。 |
| A | 9-scene viewer smoke 継続運用 | `docs/scenes-list.json` source of truth 化済み。pre-PR で `pytest tests/test_pages_assets.py -q` を通す。 |
| B | Waymo 実データ E2E | code path / prereq script はあるが、実データと Python 3.10 環境が必要。 |
| B | Pi3 / LoGeR comparison production asset | Smoke は済み。production quality の full run は未実施。 |
| C | `ntu_day_02` quality push | `scripts/plan_mcd_quality_runs.py` と collector はある。実データ再実走は未実施。 |
| C | depth / appearance / sky の比較評価 | `outdoor-training-features` experiment lab はある。real metric run は未実施。 |

### 6.4 Outdoor 実装の読みどころ

| Area | Files | Notes |
| --- | --- | --- |
| MCD calibration / static TF | `src/gs_sim2real/datasets/ros_tf.py`, `scripts/download_mcd_calibration.sh` | MCDVIRAL official calibration YAML を downloader 経由で取得。YAML は CC BY-NC-SA なので repo に commit しない。 |
| MCD supervised sparse import | `src/gs_sim2real/cli.py`, `src/gs_sim2real/datasets/mcd.py` | `--mcd-static-calibration`、single-camera colorize/depth、CameraInfo 欠落時 PINHOLE 合成、zero-GNSS guard、IMU yaw fallback。 |
| MCD quality run planning | `src/gs_sim2real/experiments/mcd_quality_plan.py`, `scripts/plan_mcd_quality_runs.py`, `scripts/collect_mcd_quality_runs.py` | `ntu_day_02` baseline / single-camera BA / multi-camera BA の commands と summary。 |
| External SLAM import | `src/gs_sim2real/preprocess/external_slam.py`, `src/gs_sim2real/preprocess/external_slam_artifacts/` | facade + profile/resolver/materializer/importer/manifest 分割。artifact 未配置でも structured error manifest を出す。 |
| External SLAM planning | `scripts/plan_external_slam_imports.py`, `scripts/collect_external_slam_imports.py` | MASt3R-SLAM / VGGT-SLAM / Pi3 / LoGeR の dry-run gate matrix と collector。 |
| Outdoor feature comparison | `src/gs_sim2real/experiments/outdoor_training_features_lab.py` | depth supervision、appearance embedding、pose refinement、sky-mask profile 比較。 |
| Pages scene contract | `docs/scenes-list.json`, `scripts/pages_scene_manifest.py`, `tests/test_pages_assets.py` | README table、preview capture、hero GIF、viewer picker を manifest に揃える。 |
| README preview capture | `scripts/capture_readme_splat_previews.py` | WebGL は headed Chromium 推奨。headless では黒 canvas になることがある。 |
| Hero GIF | `scripts/record_demo_gif.py` | `docs/scenes-list.json` の production scenes を順に cycle する。 |
| Large-scale 3DGS promotion | `src/gs_sim2real/train/large_scale_3dgs.py`, `tests/test_large_scale_3dgs.py` | training output から browser splats / viewer PLY / tile catalog / route metadata へ promotion。viewer axes transform と metrics emission が重要。 |
| Dynamic map loading runtime | `apps/dreamwalker-web/src/dynamic-map-loading.js`, `apps/dreamwalker-web/src/App.jsx`, `apps/dreamwalker-web/src/DreamwalkerScene.jsx` | catalog fields、viewer bounds、PlayCanvas GSplat loading、overview camera、resident/preload overlay。 |
| Dynamic map validator | `apps/dreamwalker-web/tools/validate-dynamic-map-catalog.mjs`, `apps/dreamwalker-web/tests/validate-dynamic-map-catalog.test.mjs` | catalog URL / public root / route coverage / metadata preload / viewer fields を検証。 |
| Istanbul media generation | `scripts/build_istanbul_3dgs_media.py` | README / Pages 用 GIF / still を生成。次は runtime viewer capture への置き換えが課題。 |

## 7. External SLAM Track

### 7.1 現行方針

MASt3R-SLAM / VGGT-SLAM 2.0 / Pi3 / LoGeR を直接 dependency として repo に抱えない。各 front-end は repo 外で実行し、出力された trajectory / pose tensor / point cloud を 3DGS Robotics に渡す。

3DGS Robotics 側の責務:

- candidate artifact path を探索する。
- TUM / KITTI / NMEA / tensor pose を一時 trajectory に materialize する。
- point tensor / PLY / PCD / NPY を point cloud として読む。
- image directory と pose count を align する。
- dry-run manifest に selected artifact、candidate trace、missing reason、gate result を残す。
- COLMAP sparse に変換し、既存 training path に渡す。

### 7.2 Profile 状態

| System | 状態 | Notes |
| --- | --- | --- |
| MASt3R-SLAM | production comparison 実走済み | `bag6-mast3r-slam-20-15k.splat` |
| VGGT-SLAM 2.0 | production comparison 実走済み | `bag6-vggt-slam-20-15k.splat` |
| Pi3 / Pi3X | production comparison 実走済み | `bag6-pi3x-20-15k.splat`。camera_poses tensor + dense points/colors/confidence を `external-slam` importer で materialize。 |
| LoGeR | smoke 済み、profile 候補追加済み | `--output_txt` trajectory と `.pt` artifact 候補。production asset は未実走。 |

### 7.3 Dry-run examples

```bash
3dgs-robotics preprocess --method external-slam --images <images-dir> \
  --external-slam-system vggt-slam --external-slam-output <slam-output-dir> \
  --external-slam-dry-run --external-slam-manifest-format json \
  --external-slam-fail-on-dry-run-gate
```

```bash
python3 scripts/plan_external_slam_imports.py --format markdown
python3 scripts/plan_external_slam_imports.py --format shell
python3 scripts/collect_external_slam_imports.py --format markdown
```

## 8. Physical AI Simulation Track

### 8.1 North Star

3DGS Robotics を「3DGS demo generator」で止めず、Physical AI policy を検証できる simulation environment にする。

最小の完成形:

1. Real outdoor scene を 3DGS asset として持つ。
2. Scene metadata、bounds、sensor rig、coordinate frame を stable JSON contract として持つ。
3. Headless environment が pose / observation / collision / reward を返す。
4. Route policy baseline と imitation policy を同じ benchmark interface で評価できる。
5. Scenario matrix を生成し、CI shard で実行できる。
6. Workflow 生成から review bundle まで、自動化の各段階を小さく検証できる。

### 8.2 既存モジュール

| Module | Role |
| --- | --- |
| `contract.py` | `SimulationCatalog`, `SceneEnvironment`, `SensorRig`, `TrajectoryEpisode` などの contract。 |
| `interfaces.py` | `PhysicalAIEnvironment`, `Observation`, `AgentAction`, `Pose3D`, `TrajectoryScore`。 |
| `headless.py` | Headless environment。bounds / occupancy / trajectory scoring。 |
| `gym_adapter.py` | Route policy を gym-like interface で動かす adapter。 |
| `occupancy.py` | LiDAR observation から occupancy grid を作る utility。 |
| `costmap.py` | Collision query summary。 |
| `footprint.py` | Robot footprint。point collision ではなく body radius / height を見る。 |
| `planning.py`, `route_planning.py` | occupancy planning / candidate route / replanning。 |
| `observation_renderer.py`, `splat_renderer.py` | observation / splat render integration。 |

### 8.3 Policy benchmark modules

| Module | Role |
| --- | --- |
| `policy_dataset.py` | Route policy dataset collection / JSON / transitions JSONL。 |
| `policy_imitation.py` | Imitation model / action decoder / fit / evaluation。 |
| `policy_feedback.py` | Observation / reward / sample building。 |
| `policy_quality.py` | Dataset quality / baseline evaluation。 |
| `policy_replay.py` | Replay batches / feature schema / transition table。 |
| `policy_benchmark.py` | Goal suite / registry / benchmark report。 |
| `policy_benchmark_history.py` | Benchmark snapshots / regression gates / history report。 |

### 8.4 まだ弱いところ

| Area | 課題 |
| --- | --- |
| Observation realism | 現在は lightweight contract 中心。camera image / depth / splat render の統合を強める必要がある。 |
| Dynamics | Headless env は policy evaluation の最小実装。real robot dynamics / latency / actuation constraints は薄い。 |
| Sensor noise | Pose / goal position / heading は `RoutePolicySensorNoiseProfile` で scenario config に落ちた。LiDAR / camera / IMU raw noise はまだ扱っていない。 |
| Multi-agent / moving obstacles | 単一の moving obstacle は `DynamicObstacleTimeline` で scenario config に入った (`step_index` に対して線形補間する waypointed sphere)。gym adapter の feature dict に `dynamic-obstacle-count` / `nearest-dynamic-obstacle-distance-meters` / `nearest-dynamic-obstacle-bearing-radians / -x / -y` を追加し、learned policy が signal を拾えるように。Multi-agent 相互作用 / reactive policy 側の連携は今後。 |
| Real benchmark correlation | 実機 / rosbag replay と sim benchmark の相関検証は未実施。 |

## 9. Route Policy Scenario CI Pipeline

この chain が 2026-04-23 時点の最重要な進捗です。巨大な benchmark を一発で回すのではなく、設定生成、sharding、CI workflow 生成、検証、activation、review publishing を分割します。

### 9.1 Pipeline overview

```text
registry + scenes + goal suites + configs
  -> scenario matrix
  -> scenario sets
  -> shard plan
  -> shard run JSONs
  -> shard merge report + history gate
  -> CI manifest
  -> generated workflow YAML (manual-only)
  -> workflow validation report
  -> workflow activation report (manual-only active path)
  -> Pages review bundle
  -> trigger promotion report
  -> trigger-enabled adoption (re-materialize + re-validate + re-activate to a distinct active path)
```

### 9.2 Modules

| Stage | Module | CLI | Output |
| --- | --- | --- | --- |
| Scenario set execution | `policy_scenario_set.py` | `route-policy-scenario-set` | scenario-set run JSON / Markdown |
| Matrix expansion | `policy_scenario_matrix.py` | `route-policy-scenario-matrix` | scenario matrix expansion JSON |
| Sharding | `policy_scenario_sharding.py` | `route-policy-scenario-shards` | shard plan JSON / shard scenario-set files |
| Shard merge | `policy_scenario_sharding.py` | `route-policy-scenario-shard-merge` | shard merge JSON / history JSON |
| CI manifest | `policy_scenario_ci_manifest.py` | `route-policy-scenario-ci-manifest` | CI manifest JSON |
| Workflow materialization | `policy_scenario_ci_workflow.py` | `route-policy-scenario-ci-workflow` | generated YAML / workflow index JSON |
| Workflow validation | `policy_scenario_ci_workflow.py` | `route-policy-scenario-ci-workflow-validate` | validation JSON / Markdown |
| Workflow activation | `policy_scenario_ci_activation.py` | `route-policy-scenario-ci-workflow-activate` | activation JSON / Markdown / active workflow YAML |
| Review publishing | `policy_scenario_ci_review.py` | `route-policy-scenario-ci-review` | review JSON / Markdown / HTML bundle |
| Workflow trigger promotion | `policy_scenario_ci_promotion.py` | `route-policy-scenario-ci-workflow-promote` | promotion JSON / Markdown |
| Trigger-enabled adoption | `policy_scenario_ci_adoption.py` | `route-policy-scenario-ci-workflow-adopt` | adoption JSON / Markdown / adopted YAML under `.github/workflows/<id>-adopted.yml` |

### 9.3 Important contracts

- `RoutePolicyScenarioCIManifest` は shard jobs と merge job を構造化する。
- `RoutePolicyScenarioCIWorkflowMaterialization` は generated YAML と config を保持する。
- `RoutePolicyScenarioCIWorkflowValidationReport` は YAML parse / text checks / payload checks / manifest consistency を保持する。
- `RoutePolicyScenarioCIWorkflowActivationReport` は validation PASS、source path、destination path、content equality、overwrite を gate 化する。
- `RoutePolicyScenarioCIReviewArtifact` は shard merge / validation / activation を Pages 向け review bundle にまとめる。
- `RoutePolicyScenarioCIWorkflowPromotionReport` は review PASS、history PASS、review URL、trigger mode、allowed branches を gate 化する。
- `RoutePolicyScenarioCIWorkflowAdoptionReport` は promotion PASS、manifest / workflow id 一致、manual path と distinct な adopted active path、adopted YAML の trigger block / branch literal 出力、再 validation / activation の PASS を gate 化する。
- `RoutePolicyScenarioCIReviewAdoption` は review artifact の任意 sub-record で、adoption id / trigger mode / adopted active path / push・pull request branches / manual vs adopted YAML の unified diff を Pages 向けに保持する。review の `passed` gate 自体は変えず、purely additive presentation。

### 9.4 Example commands

Scenario matrix:

```bash
3dgs-robotics route-policy-scenario-matrix \
  --matrix path/to/matrix.json \
  --output-dir runs/scenarios/generated \
  --output runs/scenarios/matrix-expansion.json \
  --markdown-output runs/scenarios/matrix-expansion.md
```

Shard plan:

```bash
3dgs-robotics route-policy-scenario-shards \
  --expansion runs/scenarios/matrix-expansion.json \
  --output-dir runs/scenarios/shards \
  --max-scenarios-per-shard 4 \
  --shard-plan-id outdoor-demo-shards \
  --index-output runs/scenarios/shard-plan.json \
  --markdown-output runs/scenarios/shard-plan.md
```

Shard merge:

```bash
3dgs-robotics route-policy-scenario-shard-merge \
  --run runs/scenarios/ci/runs/shard-001.json \
  --run runs/scenarios/ci/runs/shard-002.json \
  --merge-id outdoor-demo-shard-merge \
  --output runs/scenarios/ci/shard-merge.json \
  --history-output runs/scenarios/ci/shard-history.json \
  --history-markdown-output runs/scenarios/ci/shard-history.md \
  --fail-on-regression
```

CI manifest:

```bash
3dgs-robotics route-policy-scenario-ci-manifest \
  --shard-plan runs/scenarios/shard-plan.json \
  --manifest-id outdoor-demo-ci \
  --report-dir runs/scenarios/ci/reports \
  --run-output-dir runs/scenarios/ci/runs \
  --history-output-dir runs/scenarios/ci/histories \
  --merge-id outdoor-demo-shard-merge \
  --merge-output runs/scenarios/ci/shard-merge.json \
  --merge-history-output runs/scenarios/ci/shard-history.json \
  --cache-key-prefix outdoor-demo-policy \
  --fail-on-regression \
  --output runs/scenarios/ci-manifest.json \
  --markdown-output runs/scenarios/ci-manifest.md
```

Workflow materialization:

```bash
3dgs-robotics route-policy-scenario-ci-workflow \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-id outdoor-demo-policy-shards \
  --workflow-name "Outdoor Demo Policy Shards" \
  --artifact-root runs/scenarios/ci \
  --workflow-output .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --index-output runs/scenarios/ci-workflow.json \
  --markdown-output runs/scenarios/ci-workflow.md
```

Workflow validation:

```bash
3dgs-robotics route-policy-scenario-ci-workflow-validate \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-index runs/scenarios/ci-workflow.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --output runs/scenarios/ci-workflow-validation.json \
  --markdown-output runs/scenarios/ci-workflow-validation.md \
  --fail-on-validation
```

Workflow activation:

```bash
3dgs-robotics route-policy-scenario-ci-workflow-activate \
  --workflow-index runs/scenarios/ci-workflow.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --active-workflow-output .github/workflows/outdoor-demo-policy-shards.yml \
  --output runs/scenarios/ci-workflow-activation.json \
  --markdown-output runs/scenarios/ci-workflow-activation.md \
  --fail-on-activation
```

Review bundle:

```bash
3dgs-robotics route-policy-scenario-ci-review \
  --shard-merge runs/scenarios/ci/shard-merge.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --activation-report runs/scenarios/ci-workflow-activation.json \
  --review-id outdoor-demo-policy-review \
  --pages-base-url https://rsasaki0109.github.io/3dgs-robotics/reviews/outdoor-demo-policy/ \
  --bundle-dir docs/reviews/outdoor-demo-policy \
  --fail-on-review
```

Workflow promotion:

```bash
3dgs-robotics route-policy-scenario-ci-workflow-promote \
  --review runs/scenarios/ci-review.json \
  --review-url https://rsasaki0109.github.io/3dgs-robotics/reviews/outdoor-demo-policy/ \
  --trigger-mode pull-request \
  --pull-request-branch main \
  --output runs/scenarios/ci-workflow-promotion.json \
  --markdown-output runs/scenarios/ci-workflow-promotion.md \
  --fail-on-promotion
```

### 9.5 Current next step: promotion-backed workflow adoption

目的:

- Promotion report が PASS したあとに、trigger-enabled workflow を再 materialize / validate / activate する手順を固定する。
- tiny fixture で matrix expansion から promotion までを一周する smoke recipe を追加する。
- adoption 手順は active workflow YAML を直接 mutation せず、manual-only workflow と trigger-enabled workflow の差分が review できる形にする。

実装済み API:

```python
promotion = promote_route_policy_scenario_ci_workflow(
    review_artifact,
    trigger_mode="pull-request",
    pull_request_branches=("main",),
    review_url="https://rsasaki0109.github.io/3dgs-robotics/reviews/outdoor-demo-policy/",
)
write_route_policy_scenario_ci_workflow_promotion_json(
    "runs/scenarios/ci-workflow-promotion.json",
    promotion,
)
```

Promotion checks:

- review artifact が PASS。
- validation が PASS。
- activation が ACTIVE。
- shard merge が PASS。
- history gate が PASS。
- review URL が absolute http(s) URL。
- trigger mode が allowed set。
- trigger mode に必要な branches が空でない。
- branches が literal branch name policy を満たす。
- active workflow path が `.github/workflows/*.yml` / `.yaml` に閉じている。

### 9.6 Scenario CI smoke recipe

`scripts/smoke_route_policy_scenario_ci.py` が tiny 1-scene / 1-policy fixture で `scenario matrix -> shard plan -> scenario-set run -> shard merge -> CI manifest -> workflow materialization -> validation -> activation -> review -> promotion -> adoption` を一周する。各 gate に `[PASS]/[FAIL] <name>` を出し、落ちた gate で `GateFailure` を上げて non-zero exit する。

狙い:

- chain 全体の integration smoke を、巨大 E2E ではなく 1 分未満で回せる形にする。
- workflow activation / adoption は `<tmpdir>/.github/workflows/...` に閉じ、repo 本物の `.github/workflows/` には触らない。
- review bundle / promotion / adoption report の JSON / Markdown / HTML を tmpdir に吐き、目視レビューしたいときは `--keep` / `--root <path>` で保持できる。

回帰検出:

- `tests/test_smoke_route_policy_scenario_ci.py` が `run_smoke()` を importlib で叩き、全 gate の PASS ログ、artifact path、promotion trigger config、manual vs adopted YAML の差分 (`workflow_dispatch` のみ vs `pull_request:` 追加) を snapshot-assert する。

### 9.7 Promotion-backed trigger adoption

`adopt_route_policy_scenario_ci_workflow` が promotion report PASS を受けて、manual-only workflow YAML を触らずに trigger-enabled 版を別ファイルとして生成する。

- 入力: `RoutePolicyScenarioCIWorkflowPromotionReport`、同じ `RoutePolicyScenarioCIManifest`、manual-only の materialization。
- 出力: `.github/workflows/<id>-adopted.yml`（活性化された trigger-enabled YAML）、`ci-workflow-adoption.json`（gate report）、同 Markdown レンダリング。
- 失敗時は materialize も write もせずに blocked report を返すので、manual path を絶対に上書きしない。
- Gate: `promotion-promoted`, `manifest-id`, `workflow-id`, `adopted-path-distinct-from-manual`, `adopted-source-path-distinct`, trigger block (`workflow-dispatch-retained`, `push-trigger-emitted`, `pull-request-trigger-emitted`), per-branch literal check (`push-branch:<name>` / `pull-request-branch:<name>`), `adopted-validation-passed`, `adopted-activation-active`。

CLI surface は `3dgs-robotics route-policy-scenario-ci-workflow-adopt` として追加済み。manifest / workflow index / promotion JSON と adopted source / active path を渡せば同じ gate を経由する。

### 9.8 Adoption-aware review bundle

review bundle は adoption の結果を任意で取り込める。`build_route_policy_scenario_ci_review_artifact(..., adoption=RoutePolicyScenarioCIReviewAdoption)`、または CLI の `--adoption-report` を渡すと、以下を追加で Pages に出す:

- `adoption` sub-record に adoption_id / trigger_mode / adopted active path / push・pull_request branches を埋める。
- manual-only と adopted YAML の unified diff (`difflib.unified_diff`) を `workflow_diff` として保持。
- Markdown renderer は `## Adopted Workflow` セクション + \`\`\`diff ブロックを追加。
- HTML renderer は "Adopted Workflow" セクションに trigger mode / branches / 色分け diff (`<pre class="diff">` + add / del / hunk span) を描く。

review の `passed` gate 自体は shard merge / validation / activation / history のままで変わらない。adoption は purely additive presentation。

smoke script は promotion + adoption 完了後に review を再 build して bundle を上書きするので、`<tmpdir>/pages/<review-id>/review.{json,md,html}` は最終 run で adoption 情報入りになる。

Pages `docs/reviews/` index は `scripts/build_pages_reviews_index.py` で生成済み。`docs/reviews/index.html` / `docs/reviews/index.json` は、review bundle が未公開でも "no review bundles published yet" placeholder を出すので Pages `/reviews/` が 404 にならない。公開済み bundle が増えたら次のコマンドで index を再生成する:

```bash
PYTHONPATH=src python3 scripts/build_pages_reviews_index.py \
  --reviews-dir docs/reviews \
  --html-output docs/reviews/index.html \
  --json-output docs/reviews/index.json
```

Public sample として `docs/reviews/smoke-route-policy-ci/` を生成する場合は次を使う。これは `scripts/smoke_route_policy_scenario_ci.py` の synthetic fixture から作るため production benchmark ではないが、review bundle / adoption diff / index discovery の Pages contract を実物として確認できる。

```bash
PYTHONPATH=src python3 scripts/build_pages_sample_review_bundle.py
```

生成後の bundle は `review.json` / `review.md` / `index.html` と、リンク先の `sample-artifacts/` を含む。`/tmp/...` や `https://example.test/...` は commit しない。

Production benchmark run を Pages `/reviews/` に公開する場合は `scripts/publish_production_review_bundle.py` を使う。`route-policy-scenario-ci-review` が外部の production 実行で吐いた `review.json`(`provenance.kind="production"`)を `docs/reviews/<bundle-id>/` に bundle 化し、`docs/reviews/index.{html,json}` も再生成する。

```bash
PYTHONPATH=src python3 scripts/publish_production_review_bundle.py \
  --review-json runs/outdoor-demo/ci-review.json \
  --bundle-id outdoor-demo-direct-baseline-001
```

`provenance.kind` が `production` 以外だと exit 2 で reject される。bundle id は lowercase kebab-case のみ。

## 10. Public / Launch Track

### 10.1 現状

- README に CI / License / HF Spaces / Colab の 4 badge(2026-06-10 に 7 → 4 へ削減)。リンク総数も 67 → 35 に整理済み。
- GitHub Pages live demo がある。
- **HF Spaces zero-install demo** `rsasaki0109/3dgs-robotics` が RUNNING(cpu-basic、gradio 5.49.1 / Python 3.10 / torch 2.4.1 + gsplat pt24cu121)。`apps/hf-space/` から `sync-hf-space.yml` で同期。ZeroGPU 切替はユーザー判断で skip(2026-06-10)。
- **Colab notebook** がある(README badge から)。
- **ROS 2 live mapping**(`3dgs-robotics-live-mapper`)と実データ growth GIF(`docs/images/live-mapping/live-mapping-grow.gif`、KITTI drive 0056)が README / `docs/live-mapping.md` に載っている。
- GitHub About は「Photos and robot logs to browser-ready 3D Gaussian Splat maps」に簡素化済み。
- `docs/index.html` は 3DGS Robotics の public landing として整備済み。
- `docs/launch-kit.md` / `docs/launch-kit.json` に external announcement 素材がある。**告知の実行(HN / X / awesome-list PR)はユーザー自身が行う。**
- README / Pages には Large-scale 3DGS Dynamic Map Result と Real Rosbag2 3DGS Pilot Result の 2 セクションがある。
- Istanbul pilot media は `docs/images/istanbul-bag6-pilot/` にあり、Dynamic Map Viewer URL は `/dreamwalker/?tileCatalog=...istanbul-bag6-pilot...` で開く。
- GitHub Pages 反映確認は PR #192 merge 後の次タスク。deploy が成功しても runtime GSplat visible proof は別途必要。

### 10.2 Star を増やすために効く方向

コード機能よりも「初見で何がすごいか分かる」「インストールせずに 5 分で wow に到達できる」ことが重要。

2026-06-10 の整理で、star に効く開発は §18 の専用ロードマップに切り出した。短く言うと、(a) **動画 1 本 → 地図ワンライナー**で対象ユーザーを「ROS が使えるロボット屋」から「動画を撮れる全員」へ広げる、(b) **VGGT backend** で速度と話題性を取る、(c) **3DGS localization** で SLAM フォロワー層に刺さる差別化を作る、の順。PyPI 公開は価値が高いがユーザー判断で一旦保留(§18.5)。

従来の優先リスト(まだ有効、§18 と交互に進める):

1. README / Pages の hosted Dynamic Map Viewer URLs を smoke し、broken asset / base path を潰す。
2. Live GSplat renderer の visible proof を取り、README media と live demo のズレを減らす。
3. Runtime capture 由来の Istanbul GIF / still に更新する。
4. Real large-scale Istanbul pilot を 6 tiles より大きくする。
5. External SLAM comparison table を維持する。
6. `docs/launch-kit.md` の copy を短くする。
7. Pi3 / LoGeR production comparison asset を足す。
8. Review bundle を Pages に出して、CI / benchmark の信頼性を見せる。
9. 使い方を `photos-to-splat` / `external-slam import` / `dynamic-map viewer` / `physical-ai benchmark` の 4 入口に分ける。

### 10.3 ただし今の主目的

「告知機能」だけを作りすぎない。現在の主目的は Dynamic Map Viewer と Physical AI simulation environment の品質を上げること。外向けの整備は、実装された実体を見せるためにやる。特に 2026-06-09 時点では、見た目を増やすより runtime renderer の信用を上げる方が優先。

## 11. Verification Commands

### 11.1 通常 pre-PR

```bash
ruff format --check src/ tests/ scripts/
ruff check src/ tests/ scripts/
PYTHONPATH=src pytest tests/ -q --ignore=tests/e2e
```

現行環境では `python` がない場合があるので `python3` を使う。

### 11.2 Full local validation

```bash
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/gs_sim2real/sim/policy_scenario_ci_review.py \
  src/gs_sim2real/sim/policy_scenario_ci_activation.py \
  src/gs_sim2real/sim/policy_scenario_ci_promotion.py \
  src/gs_sim2real/sim/policy_scenario_ci_workflow.py \
  src/gs_sim2real/sim/__init__.py
python3 -m compileall -q src/gs_sim2real tests
pytest -q
git diff --check
```

`src/gs_sim2real/cli.py` を含む mypy full pass は、現状では Waymo / MCD loader 周辺の既知型エラーが残っている。scenario CI slice の型確認は module 単位で切る。

### 11.3 Outdoor / Pages まわり

```bash
PYTHONPATH=src pytest \
  tests/test_pages_assets.py \
  tests/test_viewer.py \
  tests/test_mcd.py \
  tests/test_mcd_gnss_preflight.py \
  tests/test_external_slam.py \
  -q
```

Viewer assets だけなら:

```bash
PYTHONPATH=src pytest tests/test_pages_assets.py tests/test_viewer.py -q
```

Dynamic Map Viewer / Istanbul pilot:

```bash
node apps/dreamwalker-web/tools/validate-dynamic-map-catalog.mjs \
  apps/dreamwalker-web/public/manifests/istanbul-bag6-pilot-tile-catalog.json \
  --public-root apps/dreamwalker-web/public \
  --site-url http://localhost:5173/ \
  --preload-mode metadata \
  --route apps/dreamwalker-web/public/robot-routes/istanbul-bag6-pilot-route.json \
  --route-playback 1 \
  --route-playback-ms 1200 \
  --route-playback-loop 1
npm --prefix apps/dreamwalker-web run test:dynamic-map-loading
npm --prefix apps/dreamwalker-web run test:dynamic-map-catalog
PYTHONPATH=src pytest tests/test_large_scale_3dgs.py
npm --prefix apps/dreamwalker-web run build
```

Hosted Pages smoke(PR #192 以降):

```bash
gh run list --repo rsasaki0109/3dgs-robotics --branch main --limit 10
curl -I https://rsasaki0109.github.io/3dgs-robotics/
curl -I https://rsasaki0109.github.io/3dgs-robotics/dreamwalker/
curl -I https://rsasaki0109.github.io/3dgs-robotics/manifests/istanbul-bag6-pilot-tile-catalog.json
curl -I https://rsasaki0109.github.io/3dgs-robotics/splats/istanbul-bag6-pilot/tile_x000_y001.ply
```

### 11.4 Physical AI / scenario CI まわり

```bash
pytest tests/test_physical_ai_policy_benchmark.py tests/test_cli.py -q
```

絞り込み:

```bash
pytest tests/test_physical_ai_policy_benchmark.py -q -k "scenario_ci_workflow"
pytest tests/test_physical_ai_policy_benchmark.py -q -k "scenario_ci_review"
pytest tests/test_cli.py -q -k "scenario_ci"
```

### 11.5 Preview / GIF

README preview PNG:

```bash
export DISPLAY=:0
python3 scripts/capture_readme_splat_previews.py
python3 scripts/enhance_demo_sweep_previews.py --hero-gif
```

Hero GIF:

```bash
python3 scripts/record_demo_gif.py
python3 scripts/enhance_demo_sweep_previews.py --hero-gif
```

Istanbul Dynamic Map Viewer media:

```bash
python3 scripts/build_istanbul_3dgs_media.py
```

現状この script は docs media の生成には使えるが、live PlayCanvas renderer の visible proof ではない。runtime capture に置き換える場合は、headed browser で `/dreamwalker/?tileCatalog=...istanbul...` を開き、canvas pixel check と screenshot を同時に残す。

### 11.6 MCD quality planning

```bash
python3 scripts/check_mcd_gnss.py <session-dir> --gnss-topic /vn200/GPS
python3 scripts/plan_mcd_quality_runs.py --format markdown
python3 scripts/collect_mcd_quality_runs.py --format markdown
python3 scripts/collect_mcd_quality_runs.py --format benchmark
python3 scripts/collect_mcd_quality_runs.py --format gate --fail-on-gate
```

## 12. Backlog

### 12.1 A: Immediate next

新しい優先順位は §17 Roadmap に集約済み。本セクションは status トラッキング用に残す。

| Task | Why | Suggested slice |
| --- | --- | --- |
| Pages deploy smoke for PR #192 | Istanbul pilot は merge 済みだが、hosted Pages の base path / asset path / deploy status は live で見る必要がある。 | `gh run list` で Pages workflow を確認し、README、`/dreamwalker/`、Istanbul catalog、first PLY、GIF / PNG を `curl -I` と browser smoke で確認。 |
| Live PlayCanvas GSplat visible proof | catalog / route / assets の検証だけでは、Dynamic Map Viewer が実際に 3DGS を描けているとは言えない。 | headed Playwright か手元 browser で Istanbul URL を開き、nonblank canvas、camera framing、route playback、tile residency overlay を screenshot / pixel check で確認。 |
| Runtime capture pipeline | README media が live viewer とズレると、見栄えは良くても demo の信用が落ちる。 | `scripts/build_istanbul_3dgs_media.py` の proxy generation を残しつつ、runtime screenshot / GIF capture script を追加する。 |
| Real large-scale Istanbul expansion | 6-tile pilot は real-input proof だが、large-scale と呼ぶには coverage が足りない。 | longer route / more registered frames / more source tiles で rerun し、catalog metrics、route、docs media、README table を同時更新。 |
| Review bundle sample under docs | 完了。Pages `/reviews/` が空ではなく scenario CI review / adoption diff の形を見せられる | `docs/reviews/smoke-route-policy-ci/` を `scripts/build_pages_sample_review_bundle.py` で生成。synthetic smoke fixture であり production benchmark ではないことを bundle 内に明示。 |
| Real review bundle from production scenario CI | **進行中 (Sprint 1 / §17.2)**。GPT pro consultation で synthetic vs production の区別を `RoutePolicyScenarioCIReviewProvenance` で first-class 化し、`docs/reviews/<run-id>/` に production bundle を公開する PR A + PR A2 に分割。 | PR A: contract / CLI / Pages index 骨格 (`plan_review_bundle_provenance.md §1`). PR A2: 実 production run の `3dgs-robotics route-policy-scenario-ci-review --kind production --bundle-dir docs/reviews/<id>` 実行と index 再生成。 |

### 12.2 B: Physical AI env hardening

| Task | Status (2026-04-26) |
| --- | --- |
| Observation renderer integration | ✅ 完了。`RoutePolicyGymAdapter` の feature dict に IMU 7 軸 (#122) と peer-aware obstacle features (#123) を surface。残課題は scene bundle 側の input sensor を増やすこと (depth / LiDAR fan-out) — このセッション以降の別チケット。 |
| Sensor noise profiles (raw sensors) | ✅ 完了。env-side noise + IMU kinematic finite-diff renderer (#111) が実装され、gym adapter feature dict に流れる (#122) ので route policy benchmark から observation 経由で σ が乗る。physics / rosbag-replay 由来の IMU renderer は引き続き OOS。 |
| Dynamic obstacles (multi-agent) | ✅ 完了。`ObstaclePolicy` protocol + 4 reference impls (#112)、env / gym adapter に per-step peer cache (#123/#124/#127)、`MaintainSeparationObstaclePolicy` 等の policy obstacle が rollout 中に peer を観測可能。残課題は Pi3-style 大規模 multi-agent scenario の production 配信 — Tier 3 候補。 |
| Route policy replay viewer | 引き続き OOS。Policy trajectory と scene を Pages で inspect する viewer は未着手。 |
| Real-vs-sim correlation report | ✅ 完了。`scripts/run_rosbag_correlation.py` (#113/#115) → scenario-set run report への attach (#121) → review bundle への surface + regression gate (#125/#126) → per-bag overrides (#128) → translation/heading per-pair distribution + time stratification (#129〜#134) まで実装済み。`3dgs-robotics route-policy-scenario-ci-review --max-correlation-* --correlation-thresholds-config --correlation-pair-distribution-strata` が production rollout で使える。残課題は event-aligned stratification (#133 OOS、外部 event timestamp が必要)。 |

### 12.3 B: Outdoor / dynamic-map asset quality

| Task | Status |
| --- | --- |
| Outdoor production regional mosaic | 完了。9 production `.splat` を 25 placement / 87 browser route tiles に展開。synthetic fixture なので real-input large-scale とは区別して説明する。 |
| Istanbul real rosbag2 pilot | 進行中。6 ready tiles、6 browser `.splat`、6 viewer PLY、438,796 Gaussians / 103.8 MiB、README media まで merge 済み。次は live renderer visible proof と real large-scale expansion。 |
| PlayCanvas GSplat runtime | 未完。catalog / route / assets は validation pass。live canvas visible proof は次タスク。 |
| Pi3 production comparison | 完了。Pi3X VO 20 frames → `external-slam` import → gsplat 15k → `docs/assets/outdoor-demo/bag6-pi3x-20-15k.splat`。 |
| LoGeR production comparison | 引き続き OOS。External SLAM comparison の説得力が増す。要 GPU run。 |
| MCD `ntu_day_02` quality reruns | 部分完了。`single_400_depth_long` (L1=0.1951) と `single_800_ba` (L1=0.2699) は gate pass。元の `multi_3cam_300each_ba` 案は `d455t` topic が MCDVIRAL ATV に存在しないことが 2026-04-26 に判明したため `multi_2cam_300each_ba` (d455b + d435i) に redefine。`d435i.bag` (5.0 GB, 5,014,702,681 bytes) は同日 evening に `data/mcd/ntu_day_02/` へ取得済 + topic 検証済み。残るのは GPU 実走 (1〜2h) と gate report 更新。 |
| Waymo E2E | high-value だが dataset access と env blocker がある。 |

#### 12.3.1 MCD quality gate targets

Production rerun は `scripts/collect_mcd_quality_runs.py --format gate --fail-on-gate` が通る状態を目標にする。Gate 本体は `src/gs_sim2real/experiments/mcd_quality_gate.py` の `MCDQualityGatePolicy` で、default は:

| Check | Default threshold | Notes |
| --- | --- | --- |
| `artifacts` | `require_complete_artifacts=True` | plan の `expected_artifacts` が全部そろっている |
| `frames` | `min_frame_fraction=0.95` | 取れた image 数 / planned `max_frames` |
| `depth` | `min_depth_fraction=0.95` | depth map 数 / image 数 (depth export 有効時) |
| `registered` | `min_registered_fraction=0.90` | COLMAP `images.txt` の登録行数 / image 数 |
| `sparse_points` | `min_sparse_points=1` | `points3D.txt` の行数下限 |
| `trained_gaussians` | `min_trained_gaussians=1` | `point_cloud.ply` の vertex 数 |
| `splat_gaussians` | `min_splat_gaussians=1` | `.splat` byte / 32 |
| `final_l1` | `require_final_l1=True` | train log に final L1 が残っている |
| `final_l1_max` | `max_final_l1=None` | 数値上限が必要なときだけ set する |

`ntu_day_02` rerun profile (`ntu_day02_single_400_depth_long` / `ntu_day02_single_800_ba` / `ntu_day02_multi_2cam_300each_ba`) は `scripts/plan_mcd_quality_runs.py` が生成。production 実行後は上記 gate を全 profile で満たす ことが完了条件。`max_final_l1` は baseline run の実測が出るまで `None` のままにしておく (regression guard として後から絞る)。`multi_2cam_300each_ba` は当初 `multi_3cam_300each_ba` (`/d455t/color/image_raw` 含む) として定義されていたが、MCDVIRAL ATV rig には `d455t` が存在しないため 2 camera (d455b + d435i) に訂正済み。

### 12.4 C: Public launch polish

| Task | Why |
| --- | --- |
| Launch kit cleanup | Star を増やすには短い copy と画像が必要。Env-hardening (pose + raw sensor noise / multi-agent dynamic obstacles) を technical / community copy に反映、Physical AI docs link + topics (`gsplat` / `scenario-ci` / `route-policy-benchmark`) 追加済み。残りは実スクリーンショット / 動画素材の差し替え。 |
| Demo preview refresh | 完了。`scripts/enhance_demo_sweep_previews.py` で 9 production preview PNG を 1280x720 のまま foreground crop / punch-up し、`hero.gif` も production scene preview 由来の軽量 loop に更新。Pages landing は Outdoor GS capability / production scene wall を前面化済み。 |

## 13. Scope Boundaries

- Python package path `gs_sim2real` は compatibility のため維持する。屋外 pipeline work の一部として rename しない。
- Legacy `gs-sim2real` CLI alias は dedicated deprecation pass まで残す。
- Downloaded MCD calibration YAML、rosbag data、Waymo tfrecords、generated training outputs は commit しない。
- External SLAM implementation 本体を repo に vendor しない。artifact importer だけを持つ。
- `docs/splat-viewer/main.js` など vendored viewer code は、compatibility fix 以外で大きく触らない。
- Generated workflow は直接 `.github/workflows/` に置かず、validation / activation / review flow を通す。
- `docs/scenes-list.json` の production scene 追加は README / viewer / tests とセットで扱う。
- `apps/dreamwalker-web/public/splats/istanbul-bag6-pilot/*.ply` のような public viewer PLY は例外的に commit 対象。raw bag / training outputs / private reconstruction directories は対象外。
- Dynamic Map Viewer hero / README media に説明 layer を増やしすぎない。lanelet2 vector map 風 layer は、実際に必要な map contract として使う段階まで hero には載せない。

## 14. 既知の落とし穴

- MCD topic は `/vn200/GPS` の大文字 `GPS`。`/vn200/gps` ではない。
- `tuhh_day_04` の `/vn200/GPS` は all-zero。supervised GNSS demo には使わない。
- MCDVIRAL ATV / handheld rig は color camera が **d435i + d455b の 2 つのみ**。`/d455t/*` topic は upstream に存在しない (Download page 全 18 session で 0 件、calibration_atv.yaml にも `d455t_*` 無し)。新規 profile を組むときは §3.2 の audit recipe に従い、Download page + calibration YAML + rosbag の 3 点で必ず交差検査する。
- MCD calibration YAML は公式 Download page から取得できるが、license 上 repo に YAML を commit しない。
- IMU orientation CSV は zero-length / non-finite quaternion を無視し、全 identity のときだけ姿勢なし扱いにする。一定の non-identity mount orientation は有効な姿勢として残す。
- Orientation が全 identity でも `angular_velocity_z` が非ゼロなら yaw-only fallback として積分する。
- `capture_readme_splat_previews.py` は headless だと WebGL canvas が黒になることがある。CI では静的 contract test、実 capture は headed smoke。
- Waymo は code path があっても実データ E2E 未検証。Python 3.10 venv と dataset agreement を先に確認する。
- Review bundle は CI workflow の信頼性を示す artifact であり、benchmark の実行そのものを代替しない。
- Activation report の `activated=True` は workflow file が guardrail を通ったという意味。GitHub 上で workflow が成功したという意味ではない。
- Dynamic Map Viewer の catalog validator pass は asset path と route coverage の証明であり、runtime GSplat が canvas に描けた証明ではない。
- GitHub Pages では `/dreamwalker/` 配下の app から `/manifests/...` や `/splats/...` を読む。local dev と hosted Pages で base path が違うため、merge 後は必ず hosted URL を smoke する。
- `viewerSplatUrl` がある tile では viewer PLY を優先する。source `.splat` metrics と viewer PLY metrics を README で混ぜると、「12.5 MB」と「103.8 MiB」が矛盾して見える。
- Istanbul pilot は real-input proof だが 6 tiles なので、synthetic 87-tile mosaic と同じ意味の large-scale result として扱わない。docs では "pilot" と "regional mosaic" を分けて説明する。

## 15. Archive Map

古い詳細は [archive snapshot](archive/plan_outdoor_gs_2026_04_full_handoff.md) に残しています。

| Need | Archive section |
| --- | --- |
| PR #55〜#80 の時系列 | `## 15`, `## 15.1`, `## 15.2` |
| `tuhh_day_04` supervised 誤判定の詳細 | `## 15.3`, `## 15.4` |
| `ntu_day_02` valid-GNSS 実走値 | `## 15.5` |
| MCD calibration YAML discovery / Drive ID | `## 4.3.3.a`, `## 4.3.3.c`, `## 15.1` |
| 8-scene viewer smoke transcript | `## 15.3` |
| Pi3 / LoGeR smoke details | External SLAM sections near `Pi3X official model` and `LoGeR official reimplementation` |
| Legacy command blocks / one-off output paths | `## 9`, `## 15.*` |

## 16. Related Documents

| File | Role |
| --- | --- |
| `README.md` | Public-facing overview, live demo, benchmark table |
| `CONTRIBUTING.md` | Development workflow and issue/PR expectations |
| `docs/physical-ai-sim.md` | Physical AI simulation contract and route policy benchmark docs |
| `docs/experiments.md` | Public experiment-process index |
| `docs/experiments.generated.md` | Generated detailed experiment comparison tables |
| `docs/decisions.md` | Accepted/deferred design decisions |
| `docs/interfaces.md` | Stable interfaces that production code may depend on |
| `docs/launch-kit.md` | Public announcement / launch material |
| `docs/plan_review_bundle_provenance.md` | PR A / PR B (production review bundle provenance + event-aligned stratification) の contract 差分メモ |
| `docs/archive/plan_outdoor_gs_2026_04_full_handoff.md` | Full historical outdoor-GS handoff snapshot |

## 17. Roadmap (2026-06〜, Dynamic Map Viewer / Physical AI)

2026-05-15 の GPT pro consultation で切った Physical AI scenario CI roadmap は有効なまま残す。ただし 2026-06-09 時点では、外向けの説得力ボトルネックが「review bundle があるか」から「real 3DGS dynamic map が live viewer で見えるか」へ一時的に移っている。

したがって 6 月前半の優先順位は、Dynamic Map Viewer runtime を信頼できる状態にすることを最上位に置く。その後、real large-scale 3DGS の coverage を増やし、最後に scenario CI / Physical AI evaluation へ再接続する。詳細な production review bundle contract 差分は引き続き [`plan_review_bundle_provenance.md`](plan_review_bundle_provenance.md)。

2026-06-10 追記: 本 roadmap と並走する形で star growth 開発ロードマップが走る。§18(動画ワンライナー → VGGT backend → 3DGS localization)は同日中に完了し、現行は **§19(rosbag 直接入力 → ループクロージャ、目標 100 stars / 年内)**。どちらの track を先に進めるかはセッションごとのユーザー指示に従う。本表の優先順位自体は据え置き。

### 17.1 優先順位

| 優先 | Sprint | Task | 狙い | 状態 |
| ---: | --- | --- | --- | --- |
| 1 | DMap 1 | GitHub Pages deploy / hosted smoke for PR #192 | merge した Istanbul pilot が live で見える入口を保証 | 次タスク。Actions / hosted URLs / assets を確認 |
| 2 | DMap 2 | PlayCanvas live GSplat visible proof | Dynamic Map Viewer runtime の信用 | catalog / route / assets は pass。canvas visible proof は未完 |
| 3 | DMap 3 | Runtime capture pipeline | README GIF / still と live viewer のズレを減らす | `scripts/build_istanbul_3dgs_media.py` は proxy media 生成済み。runtime capture は未完 |
| 4 | DMap 4 | Real large-scale Istanbul expansion | synthetic mosaic ではなく real-input large-scale へ広げる | 6-tile pilot 完。more frames / more route / more tiles が次 |
| 5 | Bridge | Dynamic map -> Physical AI scene / route-policy review 接続 | 見える map から評価できる map へ戻す | 未着手。DMap 1〜4 後 |
| 6 | Sprint 1 | Real production review bundle 公開 (PR A + PR A2) | 外向け説得力 | PR A 完(`090ee16`)、PR A2 は scaffold 完(`e9d50a9`)、本実行は production benchmark データ着次第 |
| 7 | Sprint 2 | Event-aligned stratification (PR B) | 評価品質 / policy 行動レベル correlation の土台 | 完(`2b79b5d`) |
| 8 | Sprint 3 | Policy trace events (PR C → C6) | デバッグ・説明力、Sprint 4 viewer の入力 | 完(`ace6143` → `db0cb56`、6 PR) |
| 9 | Sprint 4 | Multi-agent Tier 3 production matrix (PR D 系) | env hardening、Tier 3 候補の本丸 | 完(`2e8e738` → `b3784be`、6 PR)、staging は smoke で 2-agent 通過。production 4+ agent は次フェーズ |
| 10 | — | Sprint 4 follow-up step 1: per-step min-peer-separation collection | aggregate を peer-count placeholder から実値に | 完(`13e3b56`、2026-05-17 セッション。`nearest-dynamic-obstacle-distance-meters` を rollout 全 step 最悪値で集計) |
| 11 | — | Sprint 4 follow-up step 2: pairwise clearance histogram 実値化 | `InteractionMetricsSpec.pairwise_clearance_histogram_bins` を spec から runtime / aggregator / review 配線へ | 後続タスク(§17.7 に design)。2026-06-09 時点の最優先ではない |
| 12 | — | LoGeR production comparison asset | asset 比較の厚み | §12.3 既存 backlog |
| 13 | — | MCD `ntu_day_02` `multi_2cam_300each_ba` GPU rerun | asset 品質補完 | §12.3 既存 backlog |
| 14 | — | Applanix `read_gsof_ins_pose_stream` | data input 拡張（vendor 依存） | §3.1 OOS |

「splat を 1 個増やす」だけでは弱い。一方で、live viewer が見えていない状態で evaluation stack を強調しても初見には刺さりにくい。現在の判断は、まず real-input dynamic map を live で信用できる状態にし、その上で policy evaluation / review bundle へ接続する、という順番。

### 17.2 Sprint 1: production-review-bundle-manifest

状態: **PR A 完了**(`090ee16`、`RoutePolicyScenarioCIReviewProvenance` + CLI flags + Pages index kind 列)。**PR A2 は scaffold のみ完了**(`e9d50a9`、`scripts/publish_production_review_bundle.py` — 外部生成の review.json を `docs/reviews/<id>/` に publish して index 再生成)。実 production benchmark データが到着し次第 1 コマンドで公開できる。

Goal: production benchmark run が `RoutePolicyScenarioCIReviewArtifact` として first-class に区別され、Pages `/reviews/` index で synthetic / production が一目で分かる。

主な追加:

- `RoutePolicyScenarioCIReviewProvenance` dataclass（`kind`, `generated_at`, `git_commit`, `scene_id`, `scenario_set_id`, `matrix_hash`, `policy_version`, `env_contract_version`, `correlation_threshold_profile`, `asset_source`, `extra`）。
- `RoutePolicyScenarioCIReviewArtifact.provenance` optional field。`provenance is None` 時は既存 v1 JSON とバイト等価。
- CLI に `--kind {synthetic,production}` 他 9 個のフラグ。`--kind production` は他 provenance フィールドの指定を warning で促す。
- Pages index (`scripts/build_pages_reviews_index.py`) に `Kind` / `Scene` / `Generated` 列、schema v2 へ bump。

PR 分割:

- **PR A**: contract / CLI / Pages index 骨格 + sample bundle を `kind=synthetic` に更新。
- **PR A2**: 実 production scenario run の bundle を `docs/reviews/<run-id>/` に置き、index 再生成。README / Pages landing に導線を追加。

詳細フィールド一覧、後方互換ルール、テストケースは [`plan_review_bundle_provenance.md §1`](plan_review_bundle_provenance.md)。

### 17.3 Sprint 2: event-aligned stratification

状態: **完了**(`2b79b5d`、PR B)。`pair_distribution_strata_mode="event-aligned"` で外部 event window 軸を受理、Sprint 3 の policy trace event を `source="policy_trace"` で乗せられる土台が整った。

Goal: correlation gate を「等間隔時間 window」「等 pair 数 window」に加え「外部 event timestamp window」で評価できる。

主な追加:

- `CorrelationEventWindow` dataclass（`name`, `start_time`, `end_time`, `tags`, `source ∈ {"external","policy_trace"}`）。**`source` を Sprint 2 で先に入れる**ことで Sprint 3 の policy trace event を後方互換で乗せられる。
- `_PAIR_DISTRIBUTION_STRATA_MODES` に `"event-aligned"` 追加。
- `RealVsSimCorrelationThresholds.event_windows_path` optional field。
- fallback chain: event-aligned 指定で windows が読めない / 0 window のときは **explicit fallback** to `equal-pair-count`、review bundle metadata に `correlationStratificationFallback` を立て stderr warning を出す（silent fallback はしない）。
- `RealVsSimCorrelationWindowStats` に optional `event_name` / `event_tags` / `event_source` を追加（未指定なら JSON 出力しない、v1 互換）。

詳細は [`plan_review_bundle_provenance.md §2`](plan_review_bundle_provenance.md)。

### 17.4 Sprint 3: policy trace events

状態: **完了**(`ace6143` → `db0cb56`、6 PR: C / C2 / C3 / C4 / C5 / C6)。live trace emission が以下 4 経路すべてで wire 済み:

- Gym adapter(PR C2/C3) — Gym-step granularity
- baseline benchmark runner(PR C3) — per-policy factory
- `rollout_route` 直接呼出(PR C4) — segment granularity
- closed-loop replanning(PR C5) — per-rollout episode

PR C6 で cross-path integration tests を入れ、4 経路を組み合わせた end-to-end trace consistency も検証。

Goal: route / imitation policy が rollout 中に `goal_reached` / `near_obstacle_slowdown` / `collision` / `near_miss` / `route_deviation` 等の event を吐き、Sprint 2 の event-aligned correlation の `source = "policy_trace"` 経路にそのまま流す。

Sprint 2 で schema を先に固めているので、Sprint 3 では event 検出ロジックと bag time 対応付けだけ実装すれば良い。Real-vs-sim 比較は GPT pro 提案の段階化（occurrence → order → timing → event-local pose → segment-level trajectory）で進める。

### 17.5 Sprint 4: multi-agent Tier 3

状態: **PR D → D6 完了**(`2e8e738` → `b3784be`、6 PR)+ **follow-up step 1 完了**(`13e3b56`、per-step `min-peer-separation-meters` 実値化)。contract → matrix → run loop → shard merge → review bundle → smoke chain まで multi-agent path が end-to-end で通り、`interactionMetricsValues` が rollout 実データから生成される段階に到達。

- D: 3 record(`AgentRoleSpec` / `PopulationSpec` / `InteractionMetricsSpec`) + JSON roundtrip + legacy fallback
- D2: matrix scene spec に optional fields + `population.seed_count` で seed fan-out
- D3: `synthesize_peer_roster_from_scenario_metadata()` 純粋関数 + `_run_scenario` 配線
- D4: shard merge に `InteractionMetricsAggregate`(per-key mean/p95/max/sampleCount)
- D5: review bundle JSON / Markdown / HTML に "Multi-agent" pill + interaction metrics block + Pages index に `multiAgent` / `multiAgentCount`
- D6: smoke chain に 2-agent crossing scene を landed、`_run_scenario` が `interactionMetricsValues` を書出して D4 集計まで貫通。公開 sample bundle が `multiAgent: true` になり Pages 経由で確認可能
- follow-up step 1 (`13e3b56`): `_run_scenario` が rollout の `nearest-dynamic-obstacle-distance-meters` feature を walk して per-scenario 最悪値を `min-peer-separation-meters` として `interactionMetricsValues` に追加。sample bundle で perKeyStats に 2 key、smoke fixture の `InteractionMetricsSpec.aggregate_keys=("peer-count", "min-peer-separation-meters")` 宣言済み

実 production matrix への staging(4 → 16 → 32+ agent)と pairwise clearance histogram の実値化(§17.7)は次フェーズ。Sprint 4 Definition of Done(4-agent route-conflict scenario の `kind=production` review bundle 公開)は **production benchmark データ着次第**(PR A2 と同 blocker)。

Goal: seeded multi-agent scenario を production matrix に載せる。既存 `DynamicObstacleTimeline` / `ObstaclePolicy` protocol / per-step peer cache の上に、scenario contract の追加 dimension として `agents` / `population` / `interaction_metrics` を載せる。

scenario matrix の段階拡張:

1. 2-agent deterministic crossing
2. 4-agent route conflict
3. 16-agent seeded population
4. 32+ agent Pi3-style dense

各段階で review bundle / shard merge gate が安定することを確認してから次の規模へ。最初の public scenario は 4〜8 agent 程度で良い（CI / shard / review contract が安定してから Pi3-style dense に拡張する方が安全）。

#### 17.5.1 Schema additions（design draft）

3 つの追加 dataclass / JSON record を新設する。いずれも既存 `RoutePolicyScenarioMatrix` を壊さず、フィールドは optional として追加する。

| 新規 record | 主なフィールド | 役割 |
| --- | --- | --- |
| `AgentRoleSpec` | `agent_id`、`role` ∈ {`ego`、`peer-obstacle`、`peer-coop`}、`start_pose` または `start_volume`、`goal_pose` (optional)、`policy_ref` (registry key) または `builtin_policy` (`waypoint` / `chase` / `flee` / `maintain_separation`)、`seed_offset` | 個別 agent の明示宣言。deterministic scenario でも、roster を population 由来にしない場合の入口 |
| `PopulationSpec` | `agent_count_per_scenario: int`、`peer_role_distribution: Mapping[str, float]` (例: `chase=0.25, flee=0.25, maintain_separation=0.5`)、`random_seed: int`、`spawn_volume: AxisAlignedBounds`、`homogeneous: bool` | population 由来の peer roster 生成。`agents` か `population` のどちらか一方だけ与える契約 |
| `InteractionMetricsSpec` | `min_separation_meters: float \| None`、`aggregate_keys: tuple[str, ...]`、`pairwise_clearance_histogram_bins: tuple[float, ...] \| None`、`require_ego_survives: bool` | rollout 中に収集する multi-agent metric の宣言。集計は shard merge 側で平均 / 最悪値 / histogram |

JSON schema 拡張は `gs-mapper-route-policy-scenario-matrix/v1` の追加 optional key のみ（version bump はしない）。旧 matrix 入力は引き続き ego-only として読まれる。

#### 17.5.2 既存 pipeline へのフック箇所

| Layer | 既存 module | 追加が必要な hook |
| --- | --- | --- |
| Scenario matrix | `policy_scenario_matrix.py` | 各 `RoutePolicyMatrixSceneSpec` に optional `agents` / `population` / `interaction_metrics` を持たせる。expansion 時に `population.random_seed` の値域を外側 axis として `(scenario, seed)` ペアを fan-out |
| Scenario set | `policy_scenario_set.py` | run loop が `agents` を解釈し、peer policy を `ObstaclePolicy` instance として instantiate。per-step peer cache (#123/#127) に rollout 中の peer poses を threading |
| Sharding | `policy_scenario_sharding.py` | shard 分割は `(scenario_id, seed)` ペア単位でハッシュ。同じ scenario でも seed 違いは別 shard に行ける |
| Shard merge | `policy_scenario_sharding.py` の `merge_route_policy_scenario_shard_runs` | ego metric に加え、scenario result の `interactionMetricsValues` を per-key で aggregate(mean / p95 / max / sampleCount)し、`InteractionMetricsAggregate` として shard merge report に attach |
| Review bundle | `policy_scenario_ci_review.py` | review JSON / Markdown / HTML に "multi-agent" badge と interaction-metrics block を surface。`agents.length` ≥ 2 の bundle のみ表示 |
| Correlation gate | `route-policy-scenario-ci-review` の `--correlation-*` | 実機 bag に peer GT が無いことを前提に、初期 delivery では **ego trajectory のみ correlation 比較**を続ける。peer correlation は OOS |

#### 17.5.3 Risks / open questions

1. **Seed determinism under sharding**: shard 数変更で peer spawn が drift しないよう、population sampling は `hash((scenario_id, scenario_seed, agent_index))` で行う。shard merge は roster identity だけ assert する。
2. **Peer rollout cost**: 32-agent dense は headless env の per-step cost が線形以上に効く可能性。staging plan（2 → 4 → 16 → 32）を厳守し、各段で smoke chain の wall-clock を観測する。
3. **Backwards compat**: 既存 matrix JSON は `agents` field を持たない。loader 側で empty/missing = legacy ego-only path に fallback、unit test で旧 fixture が壊れないことを assert。
4. **Real-vs-sim correlation の扱い**: 実機 bag に peer の GT pose が無い場合が大半なので、第一弾は ego trajectory correlation のみ。peer correlation は GT 付き synthetic 由来 bag が手に入った段階で別 PR。
5. **Review bundle schema versioning**: multi-agent surface は optional フィールドとして追加し、`route-policy-scenario-ci-review/v1` の中で表現する。schema bump は population spec が must field になる時点まで遅らせる。

#### 17.5.4 PR breakdown

| PR | scope | 既存 layer への影響 | 状態 |
| --- | --- | --- | --- |
| D | `AgentRoleSpec` / `PopulationSpec` / `InteractionMetricsSpec` dataclass + JSON roundtrip + ego-only legacy fallback test | 新規 module、既存触らず | 完(`2e8e738`) |
| D2 | scenario matrix loader / expander が新 field を受理、`(scenario, seed)` fan-out | `policy_scenario_matrix.py` のみ | 完(`f84b90e`) |
| D3 | scenario set run loop が peer policy を spawn し per-step peer cache に流す、interaction metrics を collect | `policy_scenario_set.py` | 完(`8ae08c6`、D6 で `interactionMetricsValues` 書き出しまで貫通) |
| D4 | shard merge が `interactionMetricsAggregate` を attach | `policy_scenario_sharding.py` の merge path | 完(`cb38a23`) |
| D5 | review bundle JSON / Markdown / HTML に multi-agent block / badge を追加、`docs/reviews/index.json` の per-entry summary に `multiAgent` | `policy_scenario_ci_review.py`、`scripts/build_pages_reviews_index.py` | 完(`9981ed0`) |
| D6 | 2-agent deterministic crossing fixture を smoke chain に追加、`scripts/smoke_route_policy_scenario_ci.py` を multi-agent path にも対応させる | smoke recipe + `tests/test_smoke_route_policy_scenario_ci.py` + 公開 sample bundle 再生成 | 完(`b3784be`) |

D6 完了後、production matrix で 4-agent route conflict scenario を 1 つ走らせて `kind=production` review bundle を `scripts/publish_production_review_bundle.py` 経由で publish するのが Sprint 4 の Definition of Done。

### 17.6 Sprint 1 完了後に有効化する CI 自動化

| CI | 頻度 | 目的 | 起点 | 状態 |
| --- | --- | --- | --- | --- |
| PR smoke | every PR | contract drift / unit regression 検出 | 既存 `pytest tests/ -q --ignore=tests/e2e` | 既存(`.github/workflows/ci.yml`) |
| Nightly scenario CI smoke | scheduled + `workflow_dispatch` | smoke chain を回して `scenario-ci-bundles` artifact を upload(synthetic 1-scene fixture ベース、operator が inspect / promote する起点) | `.github/workflows/nightly.yml`(2026-05-16 追加、`3d28c31`) | 完 |
| Manual promotion | `workflow_dispatch` | nightly artifact を download して `route-policy-scenario-ci-workflow-promote --fail-on-promotion` を回し、promotion JSON/MD を artifact 化 | `.github/workflows/scenario-ci-promote.yml`(2026-05-16 追加、`3d28c31`) | 完 |
| Nightly production review | scheduled | `outdoor-demo` 系 production scene で scenario CI を回し `docs/reviews/<run-id>-<date>/` を生成 (commit + push) | PR A2 production execution 後 | PR A2 データ着次第有効化 |

Sprint 1 完了前に nightly を走らせると synthetic と production を区別する手段が無いので、**必ず PR A → PR A2 → CI 自動化** の順で進める。Sprint 1 の `provenance.extra["runTrigger"]` に `nightly` / `manual` / `pr` を入れれば、`kind=production` 内でも nightly / manual / pr を区別できる。

### 17.7 Sprint 4 follow-up step 2: pairwise clearance histogram(codex 引き継ぎ）

このセクションは 2026-05-17 セッションで Claude → codex に渡された handoff 用 design doc を、後続タスクとして残したものです。2026-06-09 時点の `git stash list` は空なので、途中状態を pop する前提はない。下記設計を見て scratch から書き直す方が早い。

#### 17.7.1 ゴール / 完成定義

- `InteractionMetricsSpec.pairwise_clearance_histogram_bins` を spec dataclass field("declared intent")から runtime → aggregator → review bundle まで実値化する。
- smoke chain の 2-agent crossing scene で histogram が `perKeyHistograms["pairwise-clearance"] = {binEdges: [...], counts: [...]}` として review JSON / Markdown / HTML に surface し、`tests/test_smoke_route_policy_scenario_ci.py` が assert する。
- `docs/reviews/smoke-route-policy-ci/` 公開 sample bundle が histogram を含む(`scripts/build_pages_sample_review_bundle.py` で regenerate)。
- 既存 test + 新規 test が pass し、対象 PR で validation command を明記する。

#### 17.7.2 設計 — 全体像

「per-scenario で binning 済みの `{binEdges, counts}`」を per-scenario metadata に書き、aggregator は element-wise sum で roll up、という最小拡張。raw 連続値を全 scenario 横断で持ち回らない(bounded payload)。

- **per-scenario metadata key 追加**: `interactionMetricsHistograms = {key_name: {binEdges, counts}}`(parallel to 既存 `interactionMetricsValues`)。
- **新規 dataclass**: `InteractionMetricsHistogram(bin_edges, counts)` を `policy_scenario_multi_agent.py` に追加。
- **`InteractionMetricsAggregate` 拡張**: 新 field `per_key_histograms: Mapping[str, InteractionMetricsHistogram]`(default empty dict)を追加。`to_dict` で空でない時のみ `perKeyHistograms` を payload に含める(後方互換)。
- **aggregator 拡張**: `aggregate_interaction_metrics_across_scenarios()` が `interactionMetricsHistograms` も walk し、key 単位で `bin_edges` が一致する scenario のみ counts を element-wise sum。不一致 key は roll-up から drop(undefined behavior 回避)。
- **runtime hook**: `_run_scenario()` が `scenario.metadata["interactionMetrics"]["pairwiseClearanceHistogramBins"]` を読んで bins を取得し、rollout の per-step `nearest-dynamic-obstacle-distance-meters` samples を binning して `interactionMetricsHistograms["pairwise-clearance"]` に書く。
- **review surface**: Markdown は ASCII bar(`█` で長さ表現)+ bin range labels、HTML は inline span を使った簡易 bar chart。

#### 17.7.3 file-by-file change list

| File | 変更内容 |
| --- | --- |
| `src/gs_sim2real/sim/policy_scenario_multi_agent.py` | 新規 const `INTERACTION_METRICS_HISTOGRAM_VERSION = "gs-mapper-route-policy-interaction-metrics-histogram/v1"`、`SCENARIO_INTERACTION_METRIC_HISTOGRAMS_KEY = "interactionMetricsHistograms"`。新 dataclass `InteractionMetricsHistogram(bin_edges, counts)` + `interaction_metrics_histogram_from_dict()` + `bin_pair_clearance_samples(samples, bin_edges) -> InteractionMetricsHistogram` 純粋関数(closed-open binning、最後のみ closed-closed、out-of-range silent drop)。`InteractionMetricsAggregate` に `per_key_histograms: Mapping[str, InteractionMetricsHistogram] = field(default_factory=dict)` を追加し `to_dict()` に conditional emission、`interaction_metrics_aggregate_from_dict()` を histograms にも対応。`aggregate_interaction_metrics_across_scenarios()` を拡張(下記 §17.7.4 参照)。`__all__` 追加: `INTERACTION_METRICS_HISTOGRAM_VERSION`、`InteractionMetricsHistogram`、`SCENARIO_INTERACTION_METRIC_HISTOGRAMS_KEY`、`bin_pair_clearance_samples`、`interaction_metrics_histogram_from_dict`。 |
| `src/gs_sim2real/sim/__init__.py` | 上記 5 シンボルを re-export + `__all__` に追加(既存 5 シンボルと同じ alphabetical 位置)。 |
| `src/gs_sim2real/sim/policy_scenario_set.py` | import を `from .policy_scenario_multi_agent import (SCENARIO_INTERACTION_METRIC_HISTOGRAMS_KEY, SCENARIO_INTERACTION_METRIC_VALUES_KEY, bin_pair_clearance_samples, synthesize_peer_roster_from_scenario_metadata)` に拡張。新規 helper `_peer_clearance_samples_from_report(report) -> list[float]`(既存 `_min_peer_separation_from_report` と同じ walk pattern、`min(...)` ではなく list を返す)。新規 helper `_pairwise_clearance_histogram_bins_from_metadata(metadata) -> tuple[float, ...] | None`(`metadata.get("interactionMetrics", {}).get("pairwiseClearanceHistogramBins")` を tuple 化、None / 空 / non-sequence は None)。`_run_scenario()` 末尾で histogram bins ＋ samples が両方揃った時のみ `result_metadata[SCENARIO_INTERACTION_METRIC_HISTOGRAMS_KEY] = {"pairwise-clearance": {"binEdges": [...], "counts": [...]}}` を書く。既存 `_min_peer_separation_from_report` は `_peer_clearance_samples_from_report` を呼び出す薄いラッパに refactor(または samples を `_run_scenario` で 1 回計算して min と histogram に流す)。 |
| `src/gs_sim2real/sim/policy_scenario_ci_review.py` | review JSON `to_dict()` の `interactionMetricsAggregate` ブロックは extra 変更不要(`InteractionMetricsAggregate.to_dict()` が `perKeyHistograms` を載せる)。**Markdown renderer**: 既存「## Multi-agent interaction metrics」セクションの per-key stats 表の後ろに、histogram があるキーだけ subsection を出す。例: `### pairwise-clearance histogram`、bin range 列 + count 列 + ASCII bar 列の 3 列 table。`█ * (count / max_count * 20)` で固定幅 20 文字。**HTML renderer**: 同セクションに `<table class="histogram">` + `<span class="bar" style="width: {pct}%">` 程度の最小実装。新規 CSS class `.histogram { ... } .bar { display: inline-block; background: #4a7fb8; height: 12px; }`。 |
| `scripts/build_pages_reviews_index.py` | 変更不要(per-entry summary は `multiAgent` 旗で既に表現済み)。 |
| `scripts/smoke_route_policy_scenario_ci.py` | 既存 `crossing_scene` の `InteractionMetricsSpec` に `pairwise_clearance_histogram_bins=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0)` を追加(`aggregate_keys` はそのまま、histogram は別 surface なので並存)。histogram 用の追加 PASS gate(`_gate(log, "review pairwise clearance histogram", review.interaction_metrics_aggregate is not None and "pairwise-clearance" in review.interaction_metrics_aggregate.per_key_histograms, ...)`)。 |
| `tests/test_smoke_route_policy_scenario_ci.py` | 新 assertion: `aggregate.get("perKeyHistograms")` non-empty、`"pairwise-clearance"` キー存在、`binEdges == [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]`、`sum(counts) >= 1`(2-agent 1-step なら ≥1)、HTML / Markdown bundle に `pairwise-clearance histogram` テキスト存在。 |
| `tests/test_policy_scenario_multi_agent.py` | `InteractionMetricsHistogram` validation 群: bin_edges < 2 で ValueError、strictly-increasing 違反、counts 長さ不一致、negative count、sample_count 計算、to_dict / from_dict roundtrip(min 4 ケース)。 |
| `tests/test_policy_scenario_multi_agent_aggregate.py` | `aggregate_interaction_metrics_across_scenarios` の histogram path: (a) 2 scenarios with matching bin_edges → element-wise sum、(b) mismatched bin_edges → key dropped、(c) histograms + values 混在 → 両方 aggregate、(d) histograms only(no scalar values)→ midpoint fallback で per_key_stats も生成、の min 4 ケース。 |
| `tests/test_policy_scenario_ci_review_multi_agent.py` | Markdown renderer が `### pairwise-clearance histogram` を出す ＋ HTML に `<table class="histogram">` 出す ＋ histograms 無い時は section 出さない、の min 3 ケース。 |
| `docs/reviews/smoke-route-policy-ci/` 配下全部 | `PYTHONPATH=src python3 scripts/build_pages_sample_review_bundle.py` で regenerate。新 `perKeyHistograms` が見える状態に。 |

#### 17.7.4 `aggregate_interaction_metrics_across_scenarios` 拡張の擬似コード

```python
def aggregate_interaction_metrics_across_scenarios(scenario_metadata_iter):
    per_key: dict[str, list[float]] = {}                       # 既存
    per_key_histogram_samples: dict[str, list[InteractionMetricsHistogram]] = {}  # 新
    scenario_count_with_values = 0
    for scenario_metadata in scenario_metadata_iter:
        contributed = False
        # 既存 scalar path
        values = scenario_metadata.get(SCENARIO_INTERACTION_METRIC_VALUES_KEY)
        if isinstance(values, Mapping) and values:
            for k, v in values.items():
                n = _coerce_finite_float(v)
                if n is None:
                    continue
                per_key.setdefault(str(k), []).append(n)
                contributed = True
        # 新 histogram path
        histograms = scenario_metadata.get(SCENARIO_INTERACTION_METRIC_HISTOGRAMS_KEY)
        if isinstance(histograms, Mapping) and histograms:
            for k, payload in histograms.items():
                if not isinstance(payload, Mapping):
                    continue
                try:
                    h = interaction_metrics_histogram_from_dict(payload)
                except (KeyError, ValueError):
                    continue
                per_key_histogram_samples.setdefault(str(k), []).append(h)
                contributed = True
        if contributed:
            scenario_count_with_values += 1
    if not per_key and not per_key_histogram_samples:
        return None
    per_key_stats = {k: InteractionMetricKeyStats(mean=fmean(s), p95=_p95(s), maximum=max(s), sample_count=len(s)) for k, s in per_key.items()}
    per_key_histograms: dict[str, InteractionMetricsHistogram] = {}
    for k, hs in per_key_histogram_samples.items():
        ref = hs[0].bin_edges
        if not all(h.bin_edges == ref for h in hs):
            continue  # ambiguous; drop
        summed = [0] * (len(ref) - 1)
        for h in hs:
            for i, c in enumerate(h.counts):
                summed[i] += int(c)
        per_key_histograms[k] = InteractionMetricsHistogram(bin_edges=ref, counts=tuple(summed))
    # `InteractionMetricsAggregate` の per_key_stats 不空 invariant を保つため、
    # histogram only key からも midpoint で stats を合成する fallback を必要に応じて挟む。
    if not per_key_stats:
        for k, h in per_key_histograms.items():
            mids = _histogram_midpoint_samples(h)
            if not mids:
                continue
            per_key_stats[k] = InteractionMetricKeyStats(
                mean=fmean(mids), p95=_p95(mids), maximum=max(mids), sample_count=len(mids)
            )
    if not per_key_stats:
        return None
    return InteractionMetricsAggregate(
        per_key_stats=per_key_stats,
        sample_scenario_count=scenario_count_with_values,
        per_key_histograms=per_key_histograms,
    )
```

#### 17.7.5 `_run_scenario` 拡張イメージ

既存(`13e3b56`):

```python
result_metadata = _json_mapping(scenario.metadata)
if dynamic_obstacles is not None:
    interaction_values: dict[str, float] = {
        "peer-count": float(dynamic_obstacles.obstacle_count),
    }
    min_peer_separation = _min_peer_separation_from_report(report)
    if min_peer_separation is not None:
        interaction_values["min-peer-separation-meters"] = min_peer_separation
    result_metadata[SCENARIO_INTERACTION_METRIC_VALUES_KEY] = interaction_values
```

拡張後(目安):

```python
result_metadata = _json_mapping(scenario.metadata)
if dynamic_obstacles is not None:
    samples = _peer_clearance_samples_from_report(report)
    interaction_values: dict[str, float] = {
        "peer-count": float(dynamic_obstacles.obstacle_count),
    }
    if samples:
        interaction_values["min-peer-separation-meters"] = min(samples)
    result_metadata[SCENARIO_INTERACTION_METRIC_VALUES_KEY] = interaction_values
    bins = _pairwise_clearance_histogram_bins_from_metadata(scenario.metadata)
    if bins and samples:
        histogram = bin_pair_clearance_samples(samples, bins)
        if histogram.sample_count > 0:
            result_metadata[SCENARIO_INTERACTION_METRIC_HISTOGRAMS_KEY] = {
                "pairwise-clearance": {
                    "binEdges": list(histogram.bin_edges),
                    "counts": list(histogram.counts),
                },
            }
```

`_peer_clearance_samples_from_report` は既存 `_min_peer_separation_from_report` の min 化を外したもの。両方を呼ぶ代わりに samples を 1 回計算して `min(samples)` と histogram に流す、というふうに refactor すると DRY。

#### 17.7.6 検証手順

1. 上記 file change list 通りに実装(順序: multi_agent → __init__ → scenario_set → ci_review → smoke fixture → tests)。
2. `PYTHONPATH=src python3 -m pytest tests/test_policy_scenario_multi_agent.py tests/test_policy_scenario_multi_agent_aggregate.py tests/test_policy_scenario_ci_review_multi_agent.py tests/test_smoke_route_policy_scenario_ci.py -x -q` で逐次 green を確認。
3. `PYTHONPATH=src python3 scripts/build_pages_sample_review_bundle.py` で sample bundle を再生成。
4. `PYTHONPATH=src python3 -m pytest tests/ -q --ignore=tests/e2e` で full suite を確認。
5. `python3 -m ruff check src/ tests/ scripts/` + `python3 -m ruff format --check src/ tests/ scripts/` で lint clean。
6. commit message テンプレ: `[codex] Wire pairwise clearance histogram through scenario CI chain`。CLAUDE.md 規約により Co-Authored-By trailer は付けない、PR 説明文に AI 生成表記は入れない。

#### 17.7.7 意図的に scope 外にしたもの

- **3+ peer scenario の追加**: smoke fixture は引き続き 2-agent のままで OK。histogram の意味的妥当性(複数 peer 時の peer-to-peer pair-wise 距離)は §17.7.8 で別途扱う。
- **gym adapter 改修**: 現状 `nearest-dynamic-obstacle-distance-meters`(ego→nearest) のみ使う。「all pair-wise distances per step」を出すには adapter / env の改造が必要だが、histogram の最低限の data path 通過には不要。
- **Histogram schema versioning**: `InteractionMetricsAggregate.to_dict()` は backward compat のため `perKeyHistograms` が空なら emission しない。version bump は将来必要になった時点で。
- **Production data**: pairwise histogram の production 適用は Sprint 4 Definition of Done 後(PR A2 production benchmark データ着信後)。

#### 17.7.8 後続(Sprint 5 候補)

- **全 pair-wise distance 採取**: `RoutePolicyGymAdapter` または `HeadlessPhysicalAIEnvironment` で per-step に全 peer pair 距離を info dict に exposing するか、`policy_scenario_set.py` 側で peer trajectory を再構築して計算する。3+ agent scenario で histogram が「全 pair 距離分布」になる。
- **`require_ego_survives` enforcement**: `InteractionMetricsSpec.require_ego_survives` を runtime gate にする(false なら ego collision で scenario 結果は pass のまま、true なら fail)。
- **Multi-key histogram**: `min-peer-separation-meters` 自体も binning して histogram 化、`peer-count` の分布化、など。aggregator は既に key-agnostic。

## 18. Star Growth 開発ロードマップ(2026-06-10〜)

2026-06-10 のセッションで、star 獲得に効く次期開発を 4 候補に絞り、ユーザーと優先度を確定した。

**ステータス(2026-06-10 午後更新): 採用 3 項目はすべて完了**(§1.4)。本節は完了記録 + 設計の参照先として残す。次期ロードマップは §19。

| 候補 | 内容 | 判断 |
| --- | --- | --- |
| 1 | PyPI 公開(`pip install 3dgs-robotics`) | **一旦 skip**(ユーザー判断 2026-06-10)。名前 `3dgs-robotics` は PyPI 上でまだ空いていることを確認済み。§18.5 参照 |
| 2 | 動画 1 本 → 地図のワンライナー | **完了**(`edb109d`、HF Space の `gr.Video` 入力込み) |
| 3 | VGGT feedforward backend | **完了**(`edb109d`、preprocess 実測 ~30–90 秒/round を `docs/live-mapping.md` に記載) |
| 4 | 3DGS 地図に対する localization | **完了**(`d32c0ac`、PR 分割案 1+2 まで。案 3 の ROS 2 node 化は §19.4 で rosbag 入力を優先し保留) |

この順にした理由: 2 は既存部品の配線だけで対象ユーザーが最大に広がる(ROS 不要、スマホ動画で入れる)。3 はラウンド時間短縮という実利と「VGGT 対応」という検索・SNS 流入の両取り。4 は工数最大だが、ユーザーの SLAM フォロワー層に最も刺さる差別化で、単体で次の「Show HN」級の弾になる。告知(HN / X / awesome-list)は全てユーザー自身が実行する。

### 18.1 前提: いま repo にある部品

- 動画フレーム抽出は `src/gs_sim2real/preprocess/extract_frames.py` が既にある。
- pose-free backend は DUSt3R / MASt3R の 2 系統が `src/gs_sim2real/preprocess/pose_free.py` 経由で動いている(`scripts/run_dust3r.py` / `scripts/run_mast3r.py`)。backend 追加の抽象はこの 2 例に倣う。
- live mapping は `gs_sim2real/robotics/live_mapping.py`(rclpy-free core)+ `scripts/run_live_mapping_demo.py`(image-folder replay)。ラウンドは DUSt3R で 24 frames / 1500 iters ≒ 2〜4 分(RTX 4070 Ti SUPER 16GB)。
- keyframe / `images.txt` / `points3D.txt` は live mapping の各 round に保存されており、localization の初期化素材としてそのまま使える。
- gsplat の rasterization は viewmat に対して微分可能(4 の photometric refinement の前提)。

### 18.2 動画ワンライナー(最初に着手)

**ゴール**: `3dgs-robotics map my_drive.mp4` 相当の 1 コマンドで、mp4 → フレーム抽出 → pose-free 復元 → gsplat training → `.splat` export → ローカル viewer 起動までを一気通貫にする。

設計方針:

- 新 CLI subcommand(名前は `map` か `video-to-splat`。既存 `photos-to-splat` との対称性なら後者、告知映えなら前者)。実体は `extract_frames.py` → 既存 `photos-to-splat` パスへの thin 配線で、新しい reconstruction logic は書かない。
- フレーム間引きは「動画長 → 目標フレーム数」から自動決定(default 24〜48 frames、`--num-frames` で上書き)。ブレ・露出の悪い frame の除去(laplacian variance gate)は v2 でよい。
- `--quality draft|balanced|clean|hero` preset を `photos-to-splat` から継承する。
- 完了時に `docs/splat.html?url=...` をローカル HTTP で開くところまで面倒を見る(live mapping demo の status page と同じ要領)。
- HF Space にも動画アップロード入力を足す(`apps/hf-space/app.py` に gr.Video 入力)。cpu-basic では遅いので、Space 側は frame 数を強めに絞る。

完了条件:

1. mp4 1 本から追加引数なしで `.splat` が出る(GPU ローカル)。
2. README の Quickstart が「動画 1 本で試す」を最初の入口として案内する。
3. `tests/test_cli.py` 系に synthetic 動画(数フレームの生成 mp4)での配線テストを足す。ffmpeg 依存はテスト時 skip 可能にする。
4. HF Space で動画アップロード → splat ダウンロードが通る。

リスク / 注意:

- ffmpeg の有無で挙動が変わる。`imageio-ffmpeg` を optional dependency にして、無ければ明示エラー。
- 縦動画(スマホ)で DUSt3R の 512 リサイズがどう効くかは早めに 1 本実写で確認する。

### 18.3 VGGT backend(2 の次)

**ゴール**: `--method vggt` で feedforward 復元を選べるようにし、live mapping のラウンド時間を短縮する。

動機は 2 つ。実利としては、VGGT は DUSt3R 系の pairwise + global alignment と違い 1 パスの feedforward なので、24-frame round の 2〜4 分を大幅に縮められる見込みがあり、「地図が育つ」デモのラウンド間隔が短くなる。広報としては「VGGT(CVPR 2025 Best Paper)対応」が検索・SNS 流入に効く。なお external SLAM import には既に VGGT-SLAM 2.0 の comparison 実走があるが(§7)、あれは repo 外で実行した artifact の import であり、ここでやるのは **repo 内 preprocess backend** としての VGGT 統合。役割が違う点を README でも混同させない。

設計方針:

- `pose_free.py` の DUSt3R / MASt3R と同列の backend として追加。モデルは HF hub(`facebook/VGGT-1B`)から取得し、DUSt3R と同じく「クローン or hub id」両対応にする。
- 出力(per-frame pose + depth/point map + confidence)を既存の COLMAP sparse 変換に流す。confidence 閾値は DUSt3R 側の既存処理に合わせる。
- live mapping の `--method` に `vggt` を追加し、round 時間の実測を docs に載せる(before/after 表は告知素材になる)。

完了条件:

1. `3dgs-robotics preprocess --method vggt` で KITTI / Bag6 フレームから sparse が出て、既存 train path で `.splat` まで通る。
2. live mapping で `--method vggt` のラウンドが DUSt3R より実測で速い(数値を `docs/live-mapping.md` に記載)。
3. 品質比較(同一入力で DUSt3R vs VGGT の splat)を README の comparison 文脈に 1 行追加。
4. checkpoint 未取得・VRAM 不足時のエラーメッセージが actionable。

リスク / 注意:

- VGGT-1B は重い。16GB VRAM でフレーム数いくつまで载るか最初に計測し、`--num-frames` の安全 default を決める。
- ライセンス確認(モデル weight のライセンスと商用可否)を統合前に行い、README に明記する。vendor はしない(§13 の方針通り、依存は optional)。
- torch バージョン互換(現環境 torch 2.10.0+cu128)で動くかを最初の smoke で確認。

### 18.4 3DGS localization(設計ドラフト — 着手は 2/3 完了後に再判断)

**ゴール**: 構築済みの 3DGS 地図に対してカメラ画像 1 枚 / ストリームの 6-DoF 姿勢を推定する。「マッピングからローカライズまで閉じる」ことが他の splat ツールとの決定的差別化で、ユーザーの SLAM フォロワー層に最も刺さる。

2 段構え(iComMa / MonoGS 系の確立パターン):

1. **粗い初期姿勢(リローカライズ)** — マッピング時の副産物を使う。クエリ画像を `keyframes/*.jpg` と照合し(v1 は縮小画像の類似度で十分、v2 で DINOv2 global descriptor)、最近傍 keyframe の `images.txt` 姿勢を初期値にする。`points3D.txt` を使った特徴点マッチング + PnP(hloc 流)は代替案として温存。
2. **微分可能レンダリングで精密化(トラッキング)** — gsplat で候補姿勢から地図をレンダリングし、実画像とのフォトメトリック誤差(L1 + SSIM)を SE(3) 上で Adam 最適化。収束半径確保のため coarse-to-fine の画像ピラミッド。連続フレームでは前フレーム姿勢が初期値になるため、段階 1 は最初の 1 回とロスト時だけでよく、フレームあたり数百 ms のトラッキングになる。

**MVP の定義**(これだけで GIF と Show HN 第 2 弾が成立する):

- KITTI drive 0056 の live mapping セッション(§1.3 と同じデータ)で、keyframe に採用されなかった中間フレームを「未知のクエリ」としてローカライズ。
- 推定軌跡をマッピング軌跡(`images.txt`)と重ねた GIF を `build_live_mapping_gif.py` の部品(ortho render / trajectory overlay)を再利用して生成。
- 成功判定は地図ゲージ内での相対誤差(隣接 keyframe 間隔に対する比)。メートル精度は主張しない。

PR 分割案:

1. `localize` core module(retrieval + SE(3) photometric refinement、rclpy-free、CPU でも遅いが動く)+ synthetic unit tests(既知姿勢からの摂動を回復できるか)。
2. CLI(`3dgs-robotics localize --map <session-dir> --query <image|dir>`)+ KITTI MVP 実走 + 結果 GIF。
3. (任意)ROS 2 node 化(`3dgs-robotics-live-localizer`、pose を topic で publish)— ここまでやると robotics 文脈で完結する。

リスク / 正直な見立て:

- **地図品質が最大のリスク**。1500-iter のドラフト地図はブロブ気味でフォトメトリック整合が収束しにくい可能性が高い。localization デモ用の地図は iterations を上げて作り直す(まず 7k〜15k で試す)。
- pose-free 単眼地図はスケール不定。評価はゲージ内相対で行い、docs にもそう明記する(§1.2 の「盛らない」原則と同じ)。
- 照明変化・動物体は実運用の課題だが、KITTI リプレイの MVP では同一データなので回避できる。限界は docs に正直に書く。
- 工数はコア数日 + GIF 込みで 2 の一回り上。ただし火力は最大。

### 18.5 やらないこと / 保留(再提案しない)

- **PyPI 公開**: 価値は高い(導入 1 行化、告知効果倍増)が、ユーザー判断で一旦 skip(2026-06-10)。ユーザーから話が出たら §18 冒頭の表を更新して着手する。それまで無断で publish しない。
- **HF Space の ZeroGPU 切替**: ユーザー判断で skip(2026-06-10)。cpu-basic のまま運用。再提案はしない。
- **告知の代行**: HN / X / awesome-list PR はユーザー自身が実行する。開発側は素材(`docs/launch-kit.md`、GIF、before/after 数値)を揃えるところまで。

## 19. Star Growth 第 3 期ロードマップ(2026-06-10〜): 目標 100 stars、rosbag 直接入力 + ループクロージャ

§18 の 3 項目が即日完了した(§1.4)のを受け、2026-06-10 にユーザーと次期目標・開発項目を確定した。

### 19.1 目標: GitHub 100 stars(〜2026 年内)

- 起点は **9 stars**(2026-06-10、告知未実施)。ユーザー選択は段階案(30 / 50)ではなくストレッチの **100 / 年内**。
- 100 の前提は「Show HN がフロントページ級に拾われる」か「第 2 弾(本節のループクロージャ成果)込み」のどちらか。**開発をいくら積んでも告知なしでは到達しない**ので、最大のレバーは引き続きユーザー実行の告知(HN / X / awesome-list PR、素材は `docs/launch-kit.md`)である点を毎セッション思い出すこと。
- 中間チェックポイントの目安(目標管理用であってコミットメントではない): 告知第 1 弾後に 30、ループクロージャ GIF を弾にした第 2 弾後に 100。伸びが止まったら原因(告知未実施 / 素材が刺さらない / 入口の摩擦)を切り分けてから次の開発に進む。

### 19.2 前提: 現状の軌跡推定の正直な整理(2026-06-10 の棚卸し)

開発項目を選ぶ前にユーザーと確認した現状。live mapping は専用のオドメトリ / SLAM を持たない:

1. **キーフレーム選別** — 画像の動き量ゲート(pose topic があれば `min_translation_m: 0.5` の移動距離ゲート)。外部 pose はゲートにしか使われず、地図の姿勢推定には入らない。
2. **ラウンド = 毎回ゼロからのバッチ pose-free SfM** — 各 round で「全軌跡から等間隔ストライドしたキーフレーム集合」を `PoseFreeProcessor.estimate_poses()` に渡して一括再推定(DUSt3R/MASt3R はペアワイズ + global alignment、VGGT は 1 パス feedforward)。逐次トラッキング・フレーム間オドメトリ・IMU 融合は存在しない。
3. **ラウンド間ゲージは後処理頼み** — 各 round は独立復元で gauge(スケール・座標系)がバラバラ。共有 keyframe の姿勢からの Sim3 チェーン整合は **GIF 生成時(`build_live_mapping_gif.py`)にしか行われず**、ランタイムの `live/latest.splat` は round ごとに gauge が飛ぶ。

構造的な弱点は (a) ランタイム地図が累積的でない(gauge が round ごとに変わる)、(b) 再訪しても過去の地図と整合させる仕組みがない(ループクロージャなし)、(c) round あたりフレーム数に VRAM 上限があり、軌跡が伸びるほどストライドが粗くなる。本節の 2 項目はこの (a)(b) を直接潰す。

### 19.3 項目 A: rosbag 直接入力(先に着手、工数小)

> **状況: 2026-06-11 完了**(`5c81cbb`、実走検証込み。§1.5 参照)

**ゴール**: ROS 2 ランタイムを立てずに、rosbag2(`.db3` / `.mcap`)を `3dgs-robotics` に直接食わせて live mapping replay / splat 生成ができる。現状は `ros2 bag play` + live mapper node という「ROS 環境必須」の入口しかなく(`docs/live-mapping.md` 参照)、「ロボットのログがある人」への摩擦が大きい。

ユーザーの言葉(2026-06-10): ROS 2 localizer ノードの追加よりも「**ros2 の rosbag も読み込めるくらいがいい**」。

設計方針:

- 読み込みは **`rosbags` ライブラリ(pure-Python、ROS 環境不要)** を使う。`src/gs_sim2real/datasets/mcd.py` の MCD loader が同ライブラリで既に実績を持つので、そこから bag→image 抽出の流儀(topic 列挙、`sensor_msgs/Image` と `CompressedImage` のデコード)を一般化して流用する。新しい依存は増えない。
- 入口は 2 つ:
  1. `scripts/run_live_mapping_demo.py` に `--bag <path> --image-topic <topic>` を追加し、image-folder replay と同列の frame source にする。bag のタイムスタンプから実レート / `--fps` 換算の給餌を行い、「rebuild round と給餌を並行させる」§1.3 の replay 流儀をそのまま使う。
  2. `3dgs-robotics map my_drive.bag`(または既存 video-to-splat 系 subcommand の入力判別)で、bag → フレーム抽出 → 既存 photos-to-splat パスへの thin 配線。`edb109d` の動画ワンライナーと同じ「新しい reconstruction logic は書かない」原則。
- topic 未指定時は bag 内の image 系 topic を列挙して 1 つなら自動採用、複数なら候補を表示して終了(actionable error)。フレーム間引きは動画ワンライナーと同じ「目標フレーム数からの自動決定」。

完了条件:

1. 実 bag(Bag6 系 rosbag2 か KITTI から変換した bag)で `--bag` replay → round が回り `live/latest.splat` が出る。
2. `3dgs-robotics map <bag>` で ROS なし環境から `.splat` まで通る。
3. synthetic bag(数フレームを `rosbags` で書いた fixture)での配線テスト。ROS 2 本体には依存しない。
4. README / `docs/live-mapping.md` の入口を「ROS 2 で live」「bag を直接 replay」の 2 本立てに更新。

リスク / 注意:

- `CompressedImage`(jpeg/png)と raw `Image` の encoding 分岐は MCD loader の既存処理を共通化してから使う。重複実装しない。
- bag の topic 型が想定外(`theora` 等)の場合は明示エラーで落とす。対応拡大は要望が来てから。

### 19.4 項目 B: ループクロージャ / ラウンド間ゲージ整合のランタイム昇格(本丸、Show HN 第 2 弾)

> **状況: 2026-06-11 に Step 1–3 実装完了**(Step1+2 = `2df778b`、Step3 は続くコミット。§1.5 参照)。Step3 の edge は「キーフレームを共有する全 round ペア」で構成(ストライドの性質上ループ edge を内包)し、loop candidate からの per-pair 再推定 edge は v2 へ。**before/after GIF(Show HN 第 2 弾素材)が未消化** — ループを含む drive の選定から。

**ゴール**: 「地図が育つ」を見た目だけでなく幾何的に本物にする。再訪時に過去の地図と整合し、長尺セッションでもドリフトが閉じる。これが §19.2 の弱点 (a)(b) の解消で、3DGS ライブマッピングとしての技術的差別化の核になる。

3 段階に分け、各段が単体で merge 可能・単体でデモになる:

1. **Step 1: Sim3 チェーン整合のランタイム昇格(最初にやる、効果/工数比が最大)** — `build_live_mapping_gif.py` にある「共有 keyframe のカメラ姿勢から相似変換を作り round 間でチェーンする」整合(回転は Σ R_dst·R_srcᵀ の SVD、scale/translation はカメラ中心、共有 2 カメラで成立。§1.3 の罠: 整合には `train/point_cloud.ply` を使い `scene.splat` は normalize 済みなので使わない)を `gs_sim2real/robotics/live_mapping.py` 側へ移植する。各 round 完了時に直前 round と整合してから `live/latest.splat` を export し、**ランタイム地図の gauge を固定する**。これだけで browser の `?refresh=` 表示が「round ごとに座標が飛ぶ地図」から「累積的に育つ地図」に変わる。GIF 側は後処理整合が不要になり、`scripts/build_live_mapping_gif.py` は整合済み出力をそのまま合成する fast path を持つ。
2. **Step 2: revisit 検出** — 新 keyframe を過去 keyframe 群と照合してループ候補を出す。`robotics/localize.py` の keyframe retrieval(v1: 縮小画像の類似度)をそのまま流用し、「時間的に離れているが視覚的に近い」ペアをループ候補として記録する(直近 round の隣接は除外)。v1 はログと可視化(軌跡上にループ edge を描く)までで補正はしない — 検出品質を先に見る。
3. **Step 3: ループ補正** — round を node、Step 1 の連続整合と Step 2 のループ候補(共有視野での pose-free 再推定 or localize.py の photometric refine で相対 Sim3 を出す)を edge とする **round 単位の軽量 Sim3 pose graph** を最適化し、各 round の splat / `images.txt` を再整合して export し直す。gtsam 等の新依存は入れず、まずは scipy ベースの小さな実装で済むか確認する(node 数は round 数 = 高々数十)。photometric な edge 精密化は v2。

**MVP / デモの定義**(Show HN 第 2 弾の素材):

- ループを含む走行データで「補正なしだと再訪時に二重写し / ズレる地図」vs「ループ検出で閉じる地図」の before/after GIF。KITTI drive 0056 にループがあるかは未確認なので、**最初にループのある drive(または Bag6 系)を選定する**ところから始める。
- 成功判定は地図 gauge 内の相対誤差(ループ閉合前後の再訪 keyframe 間距離)。メートル精度・SLAM ベンチマーク(ATE/RPE の絶対値)は主張しない — §1.2 の「盛らない」原則。pose-free 単眼でスケール不定である旨を docs に明記する。

リスク / 正直な見立て:

- 工数は §18 の localization の一回り上。Step 1 だけでも独立した価値があるので、**Step 1 を先に merge して旗を立てる**。
- draft 地図(1500 iters)のブロブ品質で revisit 検出の類似度がどこまで効くかは未知。retrieval は keyframe **元画像**同士の照合なので地図品質に依存しない点は救い(localize.py で実証済み)。
- Step 3 で過去 round の splat を再整合すると export 量が増える。round 数十規模なら許容、tile 化が要るほどになったら §1.2 の Dynamic Map Viewer 側と接続する。

### 19.5 着手順とやらないこと

- 着手順: **A(rosbag 入力)→ B Step 1 → B Step 2 → B Step 3**。A と B Step 1 は独立なので逆転可。B の途中でも告知第 1 弾(ユーザー実行)は並行で打てる。
- **ROS 2 localizer ノード**(§18.4 PR 分割案 3): rosbag 入力を優先するユーザー判断(2026-06-10)で保留。robotics 文脈の完結としての価値は変わらないので、B 完了後に再検討。
- **スマホ実写動画デモ**: 今期は未採用。§18.2 の「縦動画 × DUSt3R/VGGT の 512 リサイズ」リスクは**未消化のまま残っている**ので、video-to-splat を告知に使う前に 1 本実写で確認するのが安全。
- **小粒(.spz 等圧縮 export / v0.1 GitHub Release)**: .spz は引き続き保留。Release は **v0.2.0 として 2026-06-11 に公開済み**(§1.7)。
- **PyPI 公開 / ZeroGPU / 告知の代行**: §18.5 のとおり。再提案しない。

## 20. MCP サーバー「Talk to Your Map」(2026-06-11 実装完了)

> **状況: 2026-06-11 完了**(実装 + テスト 13 本 + KITTI 実走 smoke。§1.8 参照)

**ゴール**: 3DGS 地図を LLM エージェントのツール群にする。「あなたの 3DGS 地図が MCP サーバーになる」は 2026 年の文脈で最もスター獲得力のある切り口で、既存の言語 3 部作(query-map / navigate --to / splat-clean)+ detect-changes + export-overlay を 1 つの会話型インターフェースに束ねる。デモは「Claude に『駐車場の車を消して、入口まで行って』と頼む」一発で伝わる。

設計(thin 配線の原則そのまま):

- `src/gs_sim2real/robotics/mcp_server.py` + console script `3dgs-robotics-mcp`(stdio)。`mcp` SDK は optional extra `[mcp]`(`mcp>=1.2`)。モジュール自体は mcp / torch / rclpy を import せず即起動、重い処理は `python -m gs_sim2real.cli` のサブプロセスに委ね、CLI が書く JSON を読み戻して返す。**新しい reconstruction logic はゼロ**。
- ツール 7 本: `list_map_sessions` / `map_info`(in-process のセッション発見)、`query_map`(ヒット 10 件 cap + 最良ヒットへの navigate 提案を自動添付)、`navigate`(`to` / `goal_xy` / `goal_keyframe` の排他指定、`path_vertices` を落とした要約 + trace PNG / 任意 GIF)、`splat_clean`、`detect_changes`(クラスタ 10 件 cap)、`export_overlay`(ブラウザビューワ連携)。出力は `<session>/mcp/` にタイムスタンプ付きで隔離。
- docstring がエージェント向け API。ゲージ単位(camera-height、メートルではない)の注意を全ツールに明記 — §1.2 の「盛らない」原則。
- ドキュメント: README「Talk to Your Map — MCP server」セクション + `docs/mcp.md`(ツール表、Claude Code / Claude Desktop 設定例)。

検証: mcp パッケージ非依存のテスト 13 本(argv 構築 / JSON 読み戻し / cap / エラーパスを `_run_cli` seam のモックで)→ 全 1201 テストグリーン。FastMCP 実体での tools/list + stdio エンドツーエンド(initialize → list_map_sessions 呼び出し)+ KITTI drive 0056 実セッションで `query_map("car")` 17 ヒット実走を確認。

次の弾(未着手、ユーザーと要相談): ② アクティブマッピング `navigate --explore`(フロンティア探索 + 地図品質ヒートマップ、技術的本丸)③ マルチロボット・ライブマージ ④ rerun.io 連携 ⑤ シーンインベントリ。§1.7 の保留 4 案(splat-grab/paste / click-to-go / patrol / Isaac 深化)も生きている。MCP との相性は click-to-go(ビューワ経由)と patrol(エージェントが巡回を指揮)が良い。


## 21. 自律探索 `explore` — フロンティアベースのアクティブマッピング(2026-06-11 実装完了)

> **状況: 2026-06-11 完了**(実装 + テスト 12 本 + KITTI 実走 + GIF。§1.9 参照)

**ゴール**: 自律性の最終段。navigate --to は「言語で行き先を指示」だったが、explore は**誰も何も指示しない** — ロボットが可視スキャンから「観測済み空間と未観測の走行可能空間の境界(フロンティア)」を検出し、効用(クラスタサイズ / 距離、camera_height でスケール不変)が最大のフロンティアへ既存 A* + pure pursuit で走行、を到達可能 free セルのカバレッジ目標まで繰り返す。

設計(`src/gs_sim2real/robotics/splat_explore.py`、CPU-only、torch/rclpy 非依存):

- `visible_cells` = レイキャスト可視判定(occupied と unknown でレイ停止 — 壁と「未マップの霧」は見通せない)。`reachable_free_mask` = ロボット半径インフレート後の BFS 到達圏。カバレッジ分母は到達可能 free セルのみ(壁の向こうのポケットは数えない)— §1.2 の「盛らない」原則。
- `run_exploration` ループ: スキャン → frontier_clusters(8 連結、min_cells フィルタ)→ select_frontier(size/(distance+camera_height))→ plan_path 失敗クラスタは skip-set へ → run_navigation セグメント走行(observe_every ステップごとに途中スキャン)。停止理由 = coverage-target / no-frontiers / all-frontiers-unreachable / stuck(連続 2 セグメント未到達)/ max-goals。
- localizer 統合: navigate と同じ `--localize-every`(default 0 = CPU-only デッドレコニング、>0 で 3DGS localizer + innovation gate がそのまま効く)。
- 可視化: trace PNG(観測領域を緑 tint、番号付きゴール、残フロンティア黄)+ GIF(scans リプレイで観測領域が地図を掃く)。ExploreResult.scans は (step, x, y) のみ保持し GIF 側で可視判定を再計算するメモリ設計。
- CLI `3dgs-robotics explore`(cmd_navigate と同じセッション解決/グリッド構築/observer 配線)+ MCP ツール `explore`(coverage_history は要約から落とす)。

検証: 新テスト 12 本(可視レイの壁/霧停止、到達圏、フロンティア検出、効用選択、コリドー e2e、CLI/MCP argv)→ 全 1212 グリーン。KITTI drive 0056: **97.9% / 30,468 セル / 23 自己選択ゴール / stop=coverage-target**、約 2.5 分(GIF 込み)。

正直な整理(docs/live-mapping.md にも明記): occupancy grid は静的なので、explore が育てるのは**ロボットの観測領域**であって地図データそのものではない。実ロボットでは選択されたゴールが「次のマッピングフレームを撮りに行く場所」になる — live mapping のラウンド給餌と接続すれば真のアクティブマッピングになる(v2 候補)。

次の弾(未着手): マルチロボット・ライブマージ / rerun.io 連携 / シーンインベントリ / §1.7 保留 4 案(splat-grab/paste / click-to-go / patrol / Isaac 深化)。explore の v2 として「live mapping ラウンドとの接続(本物のアクティブマッピング)」も選択肢に加わった。


## 22. マルチロボット・ライブマージ `merge-live` — collaborative live mapping(2026-06-11 実装完了)

> **状況: 2026-06-11 完了**(実装 + テスト 6 本 + 実セッション実走 + GIF。§1.10 参照)

**ゴール**: 2 台のロボットが独立にライブマッピングしている 2 セッションを、ランタイムで 1 枚の地図に合流させ続ける。「two robots, one map」。既存ビューワの `?refresh=` ポーリング契約をそのまま満たすので、ブラウザには**2 台分の地図が 1 枚として育つ**様子が映る。

設計(`src/gs_sim2real/robotics/live_merge.py`、thin 配線):

- 整合とマージは**既存 `merge_sessions()` をそのまま呼ぶ**(auto: shared keyframes → localize フォールバック)。新規 reconstruction logic ゼロ。
- `watch_and_merge`: 両セッションの `live/state.json` をポーリングし、`lastSuccessfulRound` ペアが変わるたびに `merge_once` → `<output>/live/merged.ply` + `latest.splat` を tmp+os.replace で原子的に publish(live mapping の `_publish` と同じ契約)。state.json にマージログを書く。
- ビューワ正規化は**初回マージで凍結**(`compute_splat_normalization` を 1 回だけ計算し再利用)— 地図が育っても再センタリングで飛ばない(live mapping の凍結正規化と同じ思想)。
- 失敗ペア(書き込み途中の round 等)は記録してスキップ、タイトループで再試行しない。`--once` / `--max-merges` でデモ・テスト制御。
- `merge_preview`: A=青 / B=オレンジの俯瞰スキャッタ。外れ値 floater で bounds が爆発しないよう **percentile(0.5/99.5)bounds**、長軸を水平に揃える。
- CLI `merge-live`(--map-a/--map-b/--output/--align/--dedup-radius 0.1/--interval/--once/--max-merges/--preview)+ MCP ツール `merge_maps`(ワンショット merge-maps の薄ラップ)+ リプレイ GIF スクリプト `scripts/build_live_merge_gif.py`(完了済み 2 セッションの round を交互到着スケジュールで再生)。

検証: テスト 6 本(read_last_round / merge_once の原子性と正規化凍結 / watch ループのペア検出 / once の actionable エラー / 失敗ペア耐性 / MCP argv)→ 全 1218 グリーン。実走は live_demo_kitti0056 × e2e_bag_live3(同一 KITTI drive 0056 の独立セッション、ゲージ差 1.8 倍): shared 整合 12 キーフレームで GPU 不要、`--once` 47 秒で 2,566k gaussians(56k dedup)。GIF 10 イベント = docs/images/robotics/live-merge.gif。

正直な整理: 今回の実走は「同じドライブの 2 セッション」なのでキーフレーム名共有による shared 整合が効いた。完全に独立な 2 ロボットでは `--align localize`(GPU)になり、その経路は merge-maps/detect-changes で実証済みだが merge-live としての実走は未消化。ラウンドごとに再整合するコスト(localize 時)は v2 で Sim3 キャッシュ化の余地あり。

次の弾(未着手): rerun.io 連携 / シーンインベントリ / §1.7 保留 4 案 / explore v2(live mapping 接続)。MCP には query→clean→navigate→merge が揃ったので「LLM が複数ロボットの統合地図を操作する」デモが組める状態になった。


## 23. 巡回点検 `patrol` — 変化点へ見に行くロボット(2026-06-11 実装完了)

> **状況: 2026-06-11 完了**(実装 + テスト 10 本 + 実走 + 素材。§1.11 参照)

**ゴール**: 点検ストーリーの完結。「地図を作る(live mapping)→ 何が変わったか分かる(detect-changes)→ **変わった場所へロボットが見に行く(patrol --from-changes)**」。保留案③の実装だが、当初案の「detect-changes を内包する 1 コマンド」ではなく **compose 設計**(detect-changes の changes.json を入力に取る)にした — 各コマンドが単機能のまま、点検ループは 2 行で繋がる。

設計(`src/gs_sim2real/robotics/patrol.py`、thin 配線):

- waypoint 源 4 種: `--goals "x,y;x,y"` / `--goal-keyframes` / `--to "car;traffic sign"`(プロンプトごとに query_map の最良ヒット)/ `--from-changes changes.json`(クラスタ重心を grid 平面へ射影、`--change-kinds` で appeared/disappeared 選択、`--max-stops` で大クラスタ優先)。無指定なら `--num-waypoints` 個の等間隔キーフレーム巡回。
- 走行は waypoint ごとに既存 plan_path + run_navigation。**計画不能な stop は記録してスキップ**(変化クラスタは走行可能回廊の外にも出る — 正直に planned: false)。`--return-to-start` で帰投。
- `--render`: 各 stop 到着時に HeadlessSplatRenderer + pose2d_to_camera_pose(最寄りキーフレーム高さ追従)でロボット視点を PNG 保存、`--gif` でラベル付きスライドショー。`--localize-every` で navigate と同じ localizer ループ。
- トレース PNG: stop を源別に色分け(keyframe=シアン / language=オレンジ / appeared=赤 / disappeared=青 / xy=白)、計画パス + 走行軌跡。
- MCP ツール `patrol`(from_changes に detect_changes の output_json をそのまま渡せる、と docstring に明記 — エージェントの 2 ツールチェーンを誘導)。

検証: テスト 10 本(xy パース / 等間隔選択 / changes 射影と kind フィルタ / limit 大クラスタ優先 / 到達不能スキップ / capture_fn / return-to-start / トレース)→ 全 1228 グリーン。実走は e2e_bag_live3 の round5 vs 4 changes で 119 クラスタ完走(22 分、78 到達)と `--max-stops 8` デモ(97 秒、7/9 到達、視点 8 枚 + GIF)。

次の弾(未着手): rerun.io 連携 / シーンインベントリ / explore v2(live mapping 接続)/ 保留残 3 案(splat-grab/paste / click-to-go / Isaac 深化)。**MCP は 9 ツール**になり、「detect_changes → patrol」「query_map → navigate」のチェーンをエージェントが自律で組める構成が完成。


## 24. アクティブマッピング — 地図はロボットが決めた方向に育つ(explore v2、2026-06-11 実装完了)

> **状況: 2026-06-11 完了**(実装 + テスト 7 本 + KITTI 実走 + GIF。§1.12 参照)

**ゴール**: §21 の explore は静的グリッド上で「観測領域」が育つだけだった。v2 は live mapping のラウンドループに接続し、**地図データそのものがロボットの選んだ方向に伸びる**: 現在の地図のマップフロンティア(走行可能空間と unknown の境界)を検出 → 効用最大のフロンティアへ走行 → そこで「撮影」(リプレイでは録画ドライブの次バッチを給餌)→ 再構築ラウンドで地図がそちらへ拡張 → 新フロンティア出現、の完全ループ。

設計(`src/gs_sim2real/robotics/active_mapping.py`):

- **セッションゲージ不変条件**: round の raw PLY / images.txt は round 固有ゲージなので、`gauge_transform.json` の Sim3 を位置・キーフレーム中心・qvec(`q' = q ⊗ quat(Rᵀ)`)に適用してから全幾何を計算。round をまたぐロボット位置・成長判定が座標比較可能になる。
- **マップフロンティアの罠**: `inflate_obstacles` は unknown も膨張させるため「到達可能 ∧ unknown 隣接」は恒真で空。フロンティア = 「free ∧ unknown 隣接 ∧ 到達圏の near_steps 近傍」に定義し、ゴールへの接近は planner の nearest_free_cell スナップに委ねる。
- **成長判定**: ラウンドは全履歴をストライド再サンプルするので mapped centers は追記列ではない。判定は「次 round の最近傍キーフレーム距離 ≤ growth_tolerance × camera_height」。届かなければそのフロンティアを exhausted として放棄(リプレイに映像が残っていない方向の検出を兼ねる)。
- ドライバは LiveMappingSession を `build_pending_round()`(新設の同期ビルド、本体変更 10 行)で直接駆動。`grid_loader` 注入で CPU テスト可能。
- 正直な枠組み(docstring / docs に明記): 撮影源は録画リプレイであり実カメラ制御ではない。ロボットが本当に決めるのは「どのフロンティアを追うか」と「育ったかの検証」。実ロボットでは同じゴールがプラットフォーム操縦に置き換わる。

検証: テスト 7 本(フロンティア検出 / 成長・枯渇 / フレーム枯渇 / max-rounds / bootstrap 失敗 / qvec 回転)→ 全 1235 グリーン。KITTI 0056 実走(VGGT、計 13 分): bootstrap 15 フレーム 2.09M → 6 自律ラウンドで 3.05M gaussians、全フロンティア grew=True(距離 0.05〜0.14、閾値 0.08〜0.20)、stop=frames-exhausted(59 フレーム消費完了)。`active_mapping_log.json` に全意思決定を記録。GIF は round ごとの地図伸長 + ロボット軌跡 + 追跡フロンティア。

残ネタ: splat-grab/paste(次)→ Isaac 深化 → rerun.io / シーンインベントリ / click-to-go。


## 25. splat-grab / splat-paste — 言語でオブジェクトを地図間カット&ペースト(2026-06-11 実装完了)

> **状況: 2026-06-11 完了**(実装 + テスト 8 本 + 実走 + 素材。§1.13 参照)

**ゴール**: splat-clean(消す)の対になる「つまむ・置く」。実スキャンされたオブジェクトをシナリオ作成資産にする — 「map A の車をつまんで map B の駐車場に置く」「同じ車を 3 台並べてテストシーンを作る」。Physical AI ベンチマークのシーン作成と直結する。

設計(`src/gs_sim2real/robotics/splat_grab.py`、全て既存機構の流用):

- **grab** = clean_map と同一の言語選択パイプライン(query_map スコア → removal_mask)で、mask を**捨てずに残す**。`--best-cluster` 既定: プロンプトが動体(車)だとドライブ全長の出現が全部マッチするため、ベストヒット最近傍の連結ボクセル成分のみ採用(実走で 219k→21.8k に絞れた)。出力は object .ply + **sidecar .json**(camera_height / up / centroid / bottom = ゲージメタデータ)。
- **paste** = sidecar を読み、Sim3 を構成して merge_raw_gaussians(merge-maps の機構)に流す: s = 目標 camera_height / 元 camera_height(**ゲージ自動スケール**、`--scale` で上書き)、R = 目標 up への整列 × `--yaw`、t = オブジェクト底面中心(bottom-center アンカー)が `--at x,y` の目標地面に着地。`--at` は query-map の goal_xy / navigate --goal と同じグリッド平面座標 — 「query で場所を探して paste」がそのまま繋がる。
- placement_sim3 を純関数に分離(単体テスト可能)。プレビューは対象=グレー / 貼付オブジェクト=オレンジの俯瞰。

検証: テスト 8 本(mask 保持 + sidecar / 空マッチ / Rodrigues 回転(平行・反平行)/ placement_sim3 の接地とスケール / sidecar 欠落 / best-cluster 選択 / MCP argv 2 本)→ 全 1244 グリーン。実走: KITTI "car" grab 46 秒 → ゲージ差 1.8 倍の別セッションへ paste 2 秒、1 台の車が目標路上に接地。

残: Isaac 深化(次)→ rerun.io / シーンインベントリ / click-to-go。


## 26. Isaac ルートレイヤ `export-isaac-route` — ロボットの結果を USD に焼く(2026-06-12 実装完了)

> **状況: 2026-06-12 完了**(実装 + テスト 5 本 + 実走 + pxr 検証。§1.14 参照)

**ゴール**: `export-isaac`(NuRec USDZ)を「シーンだけ」から「シーン + ロボットの計画」へ深化。navigate の経路・ゴール、query-map のヒット、マッピング軌跡を USD レイヤ(BasisCurves / Sphere)として焼き、`--usdz` で `/World/Splat` から USDZ を参照 — **1 ファイル開けば Isaac Sim / usdview に splat シーンとルートが揃う**。

設計(`src/gs_sim2real/robotics/isaac_route.py`):

- **フレーム不変条件**: USDZ は round ゲージの raw PLY から transcode されるため、ルート幾何も round ゲージで書く。`route_geometry` は viewer_overlay.build_overlay のデータ準備を鏡写しにして **SplatFrameMapper(ブラウザ用正規化)を適用しない**。nav 経路の道路高さ追従リフト(`_lift_plane_points`)は流用。
- `write_route_layer`: pxr 遅延 import(無ければ `pip install usd-core` のヒント)。/World デフォルトプリム、`--usdz` 参照、カーブ幅 = camera_height×0.08、customLayerData にゲージ注記。StageUpAxis は**意図的に設定しない**(ゲージの up は任意で、splat とルートは同一フレームなので Isaac 側の補正が両方に等しく効く)。
- CLI `export-isaac-route`(--map/--nav/--query/--usdz/--no-trajectory)+ MCP ツール `export_isaac_route`(navigate → export_isaac_route のチェーンを docstring で誘導)。

検証: テスト 5 本(route_geometry のリフト・件数 / USD ラウンドトリップ(pxr importorskip): プリム型・頂点数・半径・customLayerData・参照 / suffix バリデーション / MCP argv)→ 全 1248 グリーン。実走で合成ステージに OmniNuRecFieldAsset(splat)と Route プリムの同居を確認。**正直な限界**: usd-core での読み戻し検証まで。Isaac Sim 本体でのレンダリング確認・メトリックスケール化はユーザー側。

**順送り 3 案(explore v2 / splat-grab/paste / Isaac 深化)コンプリート。** 残ネタ: rerun.io 連携 / シーンインベントリ / click-to-go。


## 27. シーンインベントリ `inventory` — 地図の国勢調査(2026-06-12 実装完了)

> **状況: 2026-06-12 完了**(実装 + テスト 5 本 + KITTI 実走。§1.15 参照)

**ゴール**: 「この地図に何が、どこに、何個あるか」への一括回答。query-map(単発)を語彙総当たりに拡張し、カテゴリ別の件数・位置・サイズ・スコアを `inventory.json` + 注釈付き俯瞰マップ PNG + Markdown レポートに集約する。

設計(`src/gs_sim2real/robotics/inventory.py`、thin):

- 中身は **query_map のループ**。効率の肝は `heatmap_fn` 注入 — `clipseg_heatmap_fn(device)` を 1 回だけ構築して全プロンプトで共有(モデル再ロードなし。テストでセンチネル同一性を検証)。
- デフォルト語彙は屋外 8 種(car/tree/building/traffic sign/pole/fence/bush/person)、`--vocab "a;b"` / `--vocab-file` で差し替え。カテゴリはクラスタ数降順、未検出も "not found" として正直に列挙。
- プレビューはカテゴリ色分け(8 色パレット巡回)の塗り円 + 番号 + 凡例。各ヒットの `goal_xy` は navigate/patrol の座標系そのまま — 「インベントリ → 巡回」「インベントリ → splat-grab」が直結。
- MCP ツール `inventory`(ヒットはカテゴリごと 3 件 cap で要約)。

検証: テスト 5 本(集計順序 / heatmap_fn 共有 / 空語彙 / Markdown / プレビュー)→ 全 1253 グリーン。KITTI 実走 74 クラスタ / 2 分 20 秒。「国勢調査であって ground truth ではない」(閾値・draft 品質依存)を docs/JSON note に明記。

残: rerun.io 連携(次)→ click-to-go。


## 28. rerun.io 連携 `rerun-replay` — セッションを rerun タイムラインへ(2026-06-12 実装完了)

> **状況: 2026-06-12 完了**(実装 + テスト 7 本 + 実走 + rrd stats 検証。§1.16 参照)

**ゴール**: robotics コミュニティで標準化しつつある rerun ビューワに「地図が育つ」を載せる。`round` タイムラインをスクラブすると、セッションゲージの色付き点群(ガウシアン中心)が round ごとに成長し、軌跡が伸び、車載カメラ画像が更新され、ループ候補エッジと nav 経路が重なる。`.rrd` は共有可能 — rerun examples / awesome 系リストへの導線になる。

設計(`src/gs_sim2real/robotics/rerun_bridge.py`):

- **組み立てとロギングを分離**: `session_timeline`(純データ、round ごとの positions/colors/centers/画像パス + loop_edges/nav_points — セッションゲージ整合・サブサンプリング込み)と `log_session`(rr 注入可能)。テストは rerun-sdk なしで両方カバー(fake rr レコーダ)。
- **SDK バージョン互換**: `set_time_sequence`(〜0.26)と `set_time(sequence=)`(新 API)を hasattr で分岐する `_set_round` ヘルパ。ローカル 0.26 / 最新 0.33 の両対応。
- optional extra `rerun = ["rerun-sdk"]`。正直な注記: rerun に 3DGS ネイティブレンダラはないため点群表現(docs / CLI 出力に明記)。
- CLI `rerun-replay`(--save 既定 `<map>/rerun/session.rrd` / --spawn / --nav / --max-points)+ MCP ツール `rerun_replay`(headless なので --spawn は出さない)。

検証: テスト 7 本(タイムライン組み立て・サブサンプル cap・色フォールバック・ループ範囲外スキップ / fake rr のエンティティパスと per-round set_time / 互換分岐 / MCP argv)→ 全 1260 グリーン。実走 8.6 秒で 18MB rrd、`rerun rrd stats` で構成確認。

残: click-to-go(最終)。


## 29. click-to-go — ブラウザでダブルクリック、ロボットが走る(2026-06-12 実装完了)

> **状況: 2026-06-12 完了**(実装 + テスト 5 本 + 実走 + headless Chrome 描画確認。§1.17 参照)

**ゴール**: overlay パイプラインの対話化。Pages のビューワに `?clickgo=<endpoint>` を足し、**地図上をダブルクリックするとロボットがそこへ走り、数秒後に走行経路がスプラットに重なる**。一番伝わるデモ素材。

設計:

- **フロント(`docs/splat-viewer/main.js`、Claude 直書き)**: ダブルクリック → 現フレーム viewProj の逆行列でマウスを 2 深度アンプロジェクト → スプラットフレームのレイを `POST <clickgo>/goal` → 返ってきた overlay URL を再フェッチして既存描画機構(ensureOverlayCanvas にリファクタ)で即描画。右上ステータスチップ(待機 / navigating / reached in N steps)。既存の射影規約(projectOverlayPoint)と同じ列優先行列演算。
- **サーバ(`robotics/click_to_go.py`)**: フレーム連鎖 = ビューワ正規化の逆写像(SplatFrameMapper 逆変換)→ round ゲージ → 地面平面とレイ交差 → グリッド平面 xy。navigate + export-overlay は **サブプロセス CLI の thin 配線**(mcp_server と同じ _run_cli)。CORS + no-cache でセッションディレクトリを配信、POST /goal はロックで直列化(409 busy)。
- 正直な挙動: 回廊外クリックは planner が最寄り free セルへスナップし「did not reach」で正直に返る(docs に明記)。座標はゲージ単位。

検証: テスト 5 本(レイ→ゴールの往復(合成 Sim3+正規化)/ 平行・後方レイの 422 / CLI argv 2 連発 / HTTP e2e(port 0、fake runner、CORS/400/422))→ 全 1265 グリーン。実セッション実走でゴール復元誤差ゼロ・350 步到達、headless Chrome(SwiftShader)で「スプラット + クリック UI + 経路オーバレイ」の合成描画をスクリーンショット確認。

**残ネタ 3 案(インベントリ / rerun.io / click-to-go)コンプリート。** ブレスト由来のネタは全消化 — 次はユーザーと新ブレストから。
