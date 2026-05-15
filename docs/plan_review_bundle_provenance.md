# Production Review Bundle Provenance / Event-Aligned Stratification 設計メモ

更新日: 2026-05-15

GPT pro consultation (2026-05-15) の結論を受けて、`docs/plan_outdoor_gs.md §17` の Sprint 1〜2 を実装するための contract 差分メモ。Sprint 3〜4 は本書では概要のみ。

## 0. 背景と狙い

現状の `RoutePolicyScenarioCIReviewArtifact` は scenario CI gate（validation / activation / shard merge / history / correlation）には足りているが、Pages `/reviews/` に並んだ bundle が「synthetic な smoke fixture 由来」なのか「実 production benchmark run の artifact」なのかが、`metadata.sampleNotice` という free-form text でしか分からない。

Sprint 1 (PR A) でその区別を first-class field に昇格させ、Sprint 2 (PR B) で correlation gate を event-aligned window で評価できるようにする。両方とも「Physical AI policy evaluation が CI artifact として継続的に出る」という方向の前提工事。

## 1. Sprint 1 — production-review-bundle-manifest

### 1.1 現状ギャップ

`RoutePolicyScenarioCIReviewArtifact` (`src/gs_sim2real/sim/policy_scenario_ci_review.py:124`) の field と GPT pro が要求した 11 個の provenance との突合:

| GPT pro 要求 | 現状 |
| --- | --- |
| `git commit` | なし（自由 metadata 欄しか置き場が無い） |
| `scene id` | なし |
| `scenario set id` | なし（`manifest_id` のみ） |
| `matrix hash` | なし |
| `shard count` | あり（`shard_count` derived property） |
| `policy version` | なし |
| `env contract version` | なし |
| `correlation threshold profile` (名前) | なし（thresholds object 自体は serialise されている） |
| `synthetic` / `production` label | **soft hint のみ** (`metadata.sampleBundle` / `metadata.sampleNotice`) |
| `asset source` | なし |
| `generated timestamp` | なし |

Pages index (`scripts/build_pages_reviews_index.py:39 ReviewIndexEntry`) も:

- `kind` 列なし、production/synthetic は区別不能。
- `generated_at` / `scene_id` 列なし。

### 1.2 提案する contract 差分

#### 1.2.1 新規 dataclass

```python
ROUTE_POLICY_SCENARIO_CI_REVIEW_PROVENANCE_VERSION = "gs-mapper-route-policy-scenario-ci-review-provenance/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewProvenance:
    """First-class provenance attached to a scenario CI review artifact."""

    kind: str  # "synthetic" | "production"
    generated_at: str  # ISO 8601 UTC, e.g. "2026-05-15T12:34:56Z"
    git_commit: str | None = None
    scene_id: str | None = None
    scenario_set_id: str | None = None
    matrix_hash: str | None = None
    policy_version: str | None = None
    env_contract_version: str | None = None
    correlation_threshold_profile: str | None = None
    asset_source: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_REVIEW_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        if self.kind not in ("synthetic", "production"):
            raise ValueError(f"kind must be 'synthetic' or 'production', got {self.kind!r}")
        # ISO 8601 syntactic check; full parse is out of scope here.
        if not self.generated_at:
            raise ValueError("generated_at must not be empty")
```

`RoutePolicyScenarioCIReviewArtifact` に optional な `provenance: RoutePolicyScenarioCIReviewProvenance | None = None` を追加。

#### 1.2.2 JSON 互換性ルール

- `provenance is None` のとき、`to_dict()` は **provenance キーを payload に含めない**（既存 v1 JSON とバイト等価）。
- `provenance is not None` のとき、`payload["provenance"] = provenance.to_dict()` を追加。`recordType` / `version` は不変（既存 loader の `_record_type` チェックは通る）。
- `route_policy_scenario_ci_review_from_dict` は `provenance` キーが無ければ `provenance=None` でロード（後方互換）。
- 既存 `metadata.sampleBundle` / `metadata.sampleNotice` / `metadata.sampleSource` は **削除しない**。レンダラー側は `provenance.kind == "synthetic"` を優先しつつ、未指定 artifact では既存 metadata fallback を使う（既存 `docs/reviews/smoke-route-policy-ci/review.json` を即 regenerate しなくても破綻しない）。

#### 1.2.3 CLI 差分

`gs-mapper route-policy-scenario-ci-review` に追加するフラグ:

| Flag | 必須 | デフォルト挙動 | 備考 |
| --- | --- | --- | --- |
| `--kind {synthetic,production}` | optional | provenance 未生成 (≒ 後方互換) | 後述の他 provenance フラグが 1 つでも指定されたら kind は必須に昇格 |
| `--scene-id` | optional | None | scenes-list.json の id を推奨 |
| `--scenario-set-id` | optional | None | shard plan の `shard-plan-id` 由来 |
| `--policy-version` | optional | None | git tag / semver |
| `--env-contract-version` | optional | None | `gs_sim2real.sim.contract` module hash や git tag |
| `--matrix-hash` | optional | None | matrix expansion JSON の sha256 を呼び出し側で計算 |
| `--correlation-threshold-profile` | optional | None | thresholds object とは別の「profile 名」ラベル |
| `--asset-source` | optional | None | `bag6` / `mcd-ntu-day02` 等 |
| `--git-commit` | optional | `git rev-parse HEAD` を best-effort で実行、失敗時 None | repo 外で生成するときは明示 |
| `--generated-at` | optional | `datetime.now(UTC).isoformat()` | 再現性が要るときは明示 |

production run の標準呼び出し例:

```bash
gs-mapper route-policy-scenario-ci-review \
  --kind production \
  --scene-id outdoor-demo \
  --scenario-set-id outdoor-demo-shard-merge \
  --policy-version v0.2.0 \
  --env-contract-version <hash> \
  --matrix-hash <sha256> \
  --asset-source bag6 \
  --correlation-threshold-profile outdoor-demo-default \
  --shard-merge runs/scenarios/ci/shard-merge.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --activation-report runs/scenarios/ci-workflow-activation.json \
  --bundle-dir docs/reviews/<run-id>/ \
  --fail-on-review
```

#### 1.2.4 Pages index 差分

`ReviewIndexEntry` に追加:

```python
kind: str = "synthetic"  # 後方互換のため synthetic を default
generated_at: str | None = None
scene_id: str | None = None
```

`collect_review_entries` は `payload.get("provenance") or {}` から `kind` / `generated_at` / `scene_id` を読む。provenance が無ければ `metadata.sampleBundle is True` を見て synthetic / 不明扱い（**production には fallback しない**）。

`render_reviews_index_html` / `render_reviews_index_markdown` に追加する列:

- `Kind` (pill: `production` 緑 / `synthetic` 黄 / `unknown` 灰)
- `Scene` (省略可、bundle 内の最初の scene_id)
- `Generated` (ISO 8601 を YYYY-MM-DD 表示)

`index.json` の schema は `SCHEMA_VERSION` を `v2` に bump。v1 ↔ v2 差分は purely additive（新 field を追加するだけ）なので、既存 tests が拾えるよう loader を一段だけ後方互換にする。

#### 1.2.5 テスト / 検証

| Test | 内容 |
| --- | --- |
| `tests/test_physical_ai_policy_benchmark.py` | review artifact roundtrip に `provenance=None` / `provenance=production` の 2 ケース |
| `tests/test_pages_reviews_index.py` | kind / generated_at / scene_id 表示、provenance 未指定 bundle が unknown 扱いになる確認 |
| `tests/test_pages_assets.py` | `docs/reviews/index.html` に `Kind` 列が含まれる smoke 検査 |
| `tests/test_cli.py` | `--kind` / `--scene-id` 等のフラグが review JSON に反映されること、`--kind production` で他 provenance を省略すると warning が出ることの確認 |
| `scripts/build_pages_sample_review_bundle.py` | `provenance=RoutePolicyScenarioCIReviewProvenance(kind="synthetic", ...)` を埋めて regenerate、commit に `docs/reviews/smoke-route-policy-ci/review.json` の diff を含める |

### 1.3 PR の落とし所

**PR A (production-review-bundle-manifest)** は contract / Pages 骨格に閉じる:

1. `RoutePolicyScenarioCIReviewProvenance` dataclass + `RoutePolicyScenarioCIReviewArtifact.provenance` field + roundtrip テスト。
2. CLI に `--kind` 以下のフラグを追加。
3. Pages index に kind / scene_id / generated_at 列を追加（schema v2）。
4. sample bundle (`docs/reviews/smoke-route-policy-ci/`) を `kind=synthetic` に更新。
5. `docs/plan_outdoor_gs.md §10.2` / `§12.1` / `§17` を実装後の状態に揃える。

**PR A2 (initial-production-review-bundle)** を別 PR に切る:

- 実 production scenario run（おそらく `outdoor-demo` 系）から `route-policy-scenario-ci-review --kind production --bundle-dir docs/reviews/<run-id>/` で bundle を生成。
- `scripts/build_pages_reviews_index.py` を流して `docs/reviews/index.{html,json}` を更新。
- README / Pages landing から「real production review」への導線を追加。

これで PR A が「synthetic→production の切替を可能にする骨格」、PR A2 が「最初の production bundle を公開する」と役割が分離する。

## 2. Sprint 2 — event-aligned stratification

### 2.1 現状ギャップ

`RealVsSimCorrelationThresholds.pair_distribution_strata_mode` (`src/gs_sim2real/robotics/rosbag_correlation.py:53`) は現在:

```python
_PAIR_DISTRIBUTION_STRATA_MODES: frozenset[str] = frozenset({"equal-duration", "equal-pair-count"})
```

`equal-duration` (#131) と `equal-pair-count` (#133) の 2 値のみ。event-aligned mode が無く、外部 event timestamp の入力点もない。

### 2.2 提案する schema

#### 2.2.1 新規 dataclass

```python
@dataclass(frozen=True, slots=True)
class CorrelationEventWindow:
    """One scenario-phase / event-aligned bag-time window."""

    name: str
    start_time: float  # bag time seconds
    end_time: float    # bag time seconds, must be > start_time
    tags: tuple[str, ...] = ()
    source: str = "external"  # "external" | "policy_trace"
```

**重要**: `source` を Sprint 2 の段階で入れておく。Sprint 3 で policy が吐く event trace は schema を変えずにそのまま乗る。

#### 2.2.2 入力フォーマット

JSON:

```json
{
  "recordType": "gs-mapper-correlation-event-windows/v1",
  "bagSourceTopic": "/odom/imu",
  "windows": [
    {"name": "approach",       "start_time": 0.0,  "end_time": 8.5,  "tags": ["straight"]},
    {"name": "turn_left",      "start_time": 8.5,  "end_time": 12.2, "tags": ["corner"], "source": "policy_trace"},
    {"name": "obstacle_avoid", "start_time": 12.2, "end_time": 18.7, "tags": ["dynamic"]}
  ]
}
```

CSV (alternative): `name,start_time,end_time,tags,source` の 5 カラム、`tags` は `;` 区切り。

### 2.3 Threshold / mode の fallback chain

`_PAIR_DISTRIBUTION_STRATA_MODES` に `"event-aligned"` を追加。`RealVsSimCorrelationThresholds` に optional な `event_windows_path: str | None` を追加（mode が event-aligned のときだけ意味を持つ）。

fallback 優先順位:

1. `pair_distribution_strata_mode == "event-aligned"` かつ event windows が読めて 1 window 以上 → event-aligned で評価。
2. event-aligned 指定だが windows が読めない / 0 window → `equal-pair-count` に **explicit fallback**。silent fallback はしない:
   - review bundle metadata に `correlationStratificationFallback = {"requested": "event-aligned", "applied": "equal-pair-count", "reason": "..."}` を立てる。
   - CLI 側で warning を stderr に出す。
3. mode 未指定 → 既存挙動 (`equal-duration`)。

### 2.4 Window stats への event 情報伝播

`RealVsSimCorrelationWindowStats` に optional な 3 field を追加（v1 互換のため未指定なら JSON 出力しない）:

```python
event_name: str | None = None
event_tags: tuple[str, ...] = ()
event_source: str | None = None
```

review bundle Markdown / HTML は event-aligned mode のときだけ "Event" 列を追加。

### 2.5 CLI 差分

`gs-mapper route-policy-scenario-ci-review` に追加:

| Flag | 説明 |
| --- | --- |
| `--correlation-event-windows <path>` | 上記 JSON / CSV パス |
| `--correlation-pair-distribution-strata-mode event-aligned` | 既存 flag の値追加 |

### 2.6 PR の落とし所

PR B 単独で:

1. `CorrelationEventWindow` dataclass + JSON / CSV loader。
2. `RealVsSimCorrelationThresholds.pair_distribution_strata_mode` の `event-aligned` 受け入れ + `event_windows_path` field。
3. `compute_per_window_correlation_stats` に event-aligned 経路追加。
4. fallback chain（explicit fallback + metadata 記録 + stderr warning）。
5. CLI flag 追加。
6. review bundle Markdown / HTML に event name 列追加（event-aligned 時のみ）。
7. test: equal-duration / equal-pair-count / event-aligned (3 windows) / event 0 windows fallback / 不正な event JSON の 5 ケース。

## 3. Sprint 3〜4 概要（PR は別途）

### 3.1 Sprint 3: policy trace events

route / imitation policy の rollout から `goal_reached` / `near_obstacle_slowdown` / `collision` / `near_miss` / `route_deviation` 等の event を吐き、Sprint 2 で入れた `CorrelationEventWindow.source = "policy_trace"` 経路にそのまま流す。

Sprint 2 で schema を先に固めているので、PR C 段階では event 検出ロジックと bag time 対応付けだけ実装すれば良い。

### 3.2 Sprint 4: multi-agent Tier 3

scenario contract に追加する dimension:

```json
{
  "agents": [
    {"id": "ego", "role": "ego", "policy": "route_policy", "route_id": "route_a"},
    {"id": "peer_001", "role": "dynamic_obstacle", "policy": "maintain_separation", "spawn": "left_crossing"}
  ],
  "population": {"preset": "pi3_style_dense", "count": 32, "seed": 123},
  "interaction_metrics": [
    "min_peer_separation_meters",
    "time_to_collision_seconds",
    "collision_count",
    "near_miss_count"
  ]
}
```

着手前に既存 `DynamicObstacleTimeline` / `ObstaclePolicy` protocol との関係を 1 度図に起こす。scenario set / matrix / shard / merge の既存 contract を壊さないため、multi-agent は scenario の追加 dimension として扱う。

scenario matrix の段階拡張:

1. 2-agent deterministic crossing
2. 4-agent route conflict
3. 16-agent seeded population
4. 32+ agent Pi3-style dense

各段階で review bundle / shard merge gate が安定することを確認してから次の規模へ。

## 4. CI 自動化（Sprint 1 完了後に有効化）

| CI | 頻度 | 目的 | 起点 |
| --- | --- | --- | --- |
| PR smoke | every PR | contract drift / unit regression 検出 | 既存 `pytest tests/ -q --ignore=tests/e2e` |
| Nightly production review | scheduled | `outdoor-demo` 系 production scene で scenario CI を回し `docs/reviews/<run-id>-<date>/` を生成（commit + push） | PR A2 後 |
| Manual promotion | workflow_dispatch | Pages に載せたい review bundle を picker から選んで promote | PR A2 後 |

Sprint 1 完了前に nightly を走らせると synthetic と production を区別する手段が無いので、**必ず PR A → PR A2 → CI 自動化** の順で進める。Sprint 1 の `provenance.extra["runTrigger"]` に `nightly` / `manual` / `pr` を入れれば、`kind=production` 内でも nightly / manual を区別できる。

## 5. 関連ファイル参照

| 役割 | パス |
| --- | --- |
| 現行 review artifact dataclass | `src/gs_sim2real/sim/policy_scenario_ci_review.py` |
| 現行 correlation thresholds | `src/gs_sim2real/robotics/rosbag_correlation.py` |
| Pages reviews index | `scripts/build_pages_reviews_index.py` |
| 既存 sample bundle generator | `scripts/build_pages_sample_review_bundle.py` |
| 既存 scenario CI smoke | `scripts/smoke_route_policy_scenario_ci.py` |
| Pages reviews index test | `tests/test_pages_reviews_index.py` |
| Pages assets / production picker test | `tests/test_pages_assets.py` |
| CLI test | `tests/test_cli.py` |
| Physical AI benchmark test | `tests/test_physical_ai_policy_benchmark.py` |
