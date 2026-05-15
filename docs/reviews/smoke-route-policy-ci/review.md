# Route Policy Scenario CI Review: smoke-route-policy-ci-review
- Status: PASS
- Workflow: smoke-route-policy-ci-workflow
- Manifest: smoke-route-policy-ci-manifest
- Merge: smoke-route-policy-ci-merge (PASS)
- Validation: smoke-route-policy-ci-validation (PASS)
- Activation: smoke-route-policy-ci-activation (ACTIVE)
- Active workflow: sample-artifacts/workflows/smoke-route-policy-ci.yml
- Shards: 2
- Scenarios: 2
- Reports: 2

> Synthetic smoke fixture generated from scripts/smoke_route_policy_scenario_ci.py; it proves the scenario-CI review bundle contract but is not a production benchmark run.

## Run Provenance

- Kind: synthetic
- Generated at: 2026-05-15T00:00:00+00:00
- Scene id: `smoke-route-policy-ci`
- Asset source: `scripts/smoke_route_policy_scenario_ci.py`
- runTrigger: `manual`
- sampleBundle: `true`

| Shard | Pass | Scenarios | Reports | Run |
| --- | --- | ---: | ---: | --- |
| smoke-route-policy-ci-matrix-direct-shard-001 | yes | 1 | 1 | sample-artifacts/runs/smoke-route-policy-ci-matrix-direct-shard-001.json |
| smoke-route-policy-ci-matrix-direct-shard-002 | yes | 1 | 1 | sample-artifacts/runs/smoke-route-policy-ci-matrix-direct-shard-002.json |

## Adopted Workflow

- Adoption: smoke-route-policy-ci-adoption (ADOPTED)
- Trigger mode: pull-request
- Adopted active path: sample-artifacts/workflows/smoke-route-policy-ci-adopted.yml
- Adopted source path: sample-artifacts/ci-workflow-adopted.generated.yml
- Push branches: n/a
- Pull request branches: main

```diff
--- manual
+++ adopted
@@ -4,6 +4,9 @@
 
 on:
   workflow_dispatch: {}
+  pull_request:
+    branches:
+      - 'main'
 
 jobs:
   route-policy-scenario-shards:
```
