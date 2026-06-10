# Experiments

Updated: 2026-04-21T12:27:09.436Z

This repository treats design as an evolving comparison space.
Stable code stays in `core`; competing implementations stay in `experiments` until comparison makes part of them worth keeping.

This page is the public index. Full generated comparison tables live in
[docs/experiments.generated.md](experiments.generated.md); accepted/deferred
decisions live in [docs/decisions.md](decisions.md); stable production
surfaces live in [docs/interfaces.md](interfaces.md).

## Current Seams

| Seam | Stable Decision | Deferred Strategies | Detail |
| --- | --- | --- | --- |
| Localization Alignment | Stable code uses `gs_sim2real.core.localization_alignment.align_pose_samples()` as the only contract that production callers should depend on. | 2 deferred | [tables](experiments.generated.md#localization-alignment) |
| Render Backend Selection | Stable code uses `select_render_backend(RenderBackendRequest(...))` as the only backend-selection contract. | 2 deferred | [tables](experiments.generated.md#render-backend-selection) |
| Localization Estimate Import | Stable code uses `import_localization_estimate_document(LocalizationEstimateImportRequest(...))` as the importer contract. | 2 deferred | [tables](experiments.generated.md#localization-estimate-import) |
| Localization Review Bundle Import | Stable web code uses `importLocalizationReviewBundleDocument(rawDocument, policy?)` as the only review-bundle import contract. | 2 deferred | [tables](experiments.generated.md#localization-review-bundle-import) |
| Query Cancellation Policy | Stable transport code uses `resolve_query_cancellation_decision(QueryCancellationRequest(...))` as the only cancellation surface. | 2 deferred | [tables](experiments.generated.md#query-cancellation-policy) |
| Query Coalescing Policy | Stable queue code uses `resolve_query_coalescing_decision(QueryCoalescingRequest(...))` as the only dedupe/coalescing surface. | 2 deferred | [tables](experiments.generated.md#query-coalescing-policy) |
| Query Error Mapping | Stable transport code uses `resolve_query_error_mapping(QueryErrorMappingRequest(...))` as the only error-mapping surface. | 2 deferred | [tables](experiments.generated.md#query-error-mapping) |
| Query Transport Selection | Stable code uses `select_query_transport(QueryTransportRequest(...))` as the only transport-selection contract. | 2 deferred | [tables](experiments.generated.md#query-transport-selection) |
| Query Request Import | Stable code uses `import_query_request(QueryRequestImportRequest(...))` as the only query payload import contract. | 2 deferred | [tables](experiments.generated.md#query-request-import) |
| Query Queue Policy | Stable server code uses `admit_query_queue_item(...)` and `dispatch_query_queue_item(...)` as the only queue policy surface. | 2 deferred | [tables](experiments.generated.md#query-queue-policy) |
| Query Source Identity | Stable transport code uses `resolve_query_source_identity(QuerySourceIdentityRequest(...))` as the only source-identity surface. | 2 deferred | [tables](experiments.generated.md#query-source-identity) |
| Query Timeout Policy | Stable server/client code uses `resolve_query_timeout_plan(QueryTimeoutPolicyRequest(...))` as the only timeout-policy surface. | 2 deferred | [tables](experiments.generated.md#query-timeout-policy) |
| Query Response Build | Stable server code uses `build_render_result_response_document(...)`, `build_query_ready_response_document(...)`, and `build_query_error_response_document(...)` as the only response-build surface. | 2 deferred | [tables](experiments.generated.md#query-response-build) |
| Live Localization Stream Import | Stable web code uses `importLiveLocalizationStreamMessage(previousEstimate, rawMessage, options, policy?)` as the only live-stream import contract. | 2 deferred | [tables](experiments.generated.md#live-localization-stream-import) |
| Route Capture Bundle Import | Stable code uses `import_route_capture_bundle(RouteCaptureBundleImportRequest(...))` as the only ground-truth bundle import contract. | 2 deferred | [tables](experiments.generated.md#route-capture-bundle-import) |
| Sim2Real Websocket Protocol | Stable web code uses `importSim2realWebsocketMessage(rawMessage, policy?)` as the only websocket response import contract. | 2 deferred | [tables](experiments.generated.md#sim2real-websocket-protocol) |

## Localization Alignment

Stable: Stable code uses `gs_sim2real.core.localization_alignment.align_pose_samples()` as the only contract that production callers should depend on.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#localization-alignment](experiments.generated.md#localization-alignment)

## Render Backend Selection

Stable: Stable code uses `select_render_backend(RenderBackendRequest(...))` as the only backend-selection contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#render-backend-selection](experiments.generated.md#render-backend-selection)

## Localization Estimate Import

Stable: Stable code uses `import_localization_estimate_document(LocalizationEstimateImportRequest(...))` as the importer contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#localization-estimate-import](experiments.generated.md#localization-estimate-import)

## Localization Review Bundle Import

Stable: Stable web code uses `importLocalizationReviewBundleDocument(rawDocument, policy?)` as the only review-bundle import contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#localization-review-bundle-import](experiments.generated.md#localization-review-bundle-import)

## Query Cancellation Policy

Stable: Stable transport code uses `resolve_query_cancellation_decision(QueryCancellationRequest(...))` as the only cancellation surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-cancellation-policy](experiments.generated.md#query-cancellation-policy)

## Query Coalescing Policy

Stable: Stable queue code uses `resolve_query_coalescing_decision(QueryCoalescingRequest(...))` as the only dedupe/coalescing surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-coalescing-policy](experiments.generated.md#query-coalescing-policy)

## Query Error Mapping

Stable: Stable transport code uses `resolve_query_error_mapping(QueryErrorMappingRequest(...))` as the only error-mapping surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-error-mapping](experiments.generated.md#query-error-mapping)

## Query Transport Selection

Stable: Stable code uses `select_query_transport(QueryTransportRequest(...))` as the only transport-selection contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-transport-selection](experiments.generated.md#query-transport-selection)

## Query Request Import

Stable: Stable code uses `import_query_request(QueryRequestImportRequest(...))` as the only query payload import contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-request-import](experiments.generated.md#query-request-import)

## Query Queue Policy

Stable: Stable server code uses `admit_query_queue_item(...)` and `dispatch_query_queue_item(...)` as the only queue policy surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-queue-policy](experiments.generated.md#query-queue-policy)

## Query Source Identity

Stable: Stable transport code uses `resolve_query_source_identity(QuerySourceIdentityRequest(...))` as the only source-identity surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-source-identity](experiments.generated.md#query-source-identity)

## Query Timeout Policy

Stable: Stable server/client code uses `resolve_query_timeout_plan(QueryTimeoutPolicyRequest(...))` as the only timeout-policy surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-timeout-policy](experiments.generated.md#query-timeout-policy)

## Query Response Build

Stable: Stable server code uses `build_render_result_response_document(...)`, `build_query_ready_response_document(...)`, and `build_query_error_response_document(...)` as the only response-build surface.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#query-response-build](experiments.generated.md#query-response-build)

## Live Localization Stream Import

Stable: Stable web code uses `importLiveLocalizationStreamMessage(previousEstimate, rawMessage, options, policy?)` as the only live-stream import contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#live-localization-stream-import](experiments.generated.md#live-localization-stream-import)

## Route Capture Bundle Import

Stable: Stable code uses `import_route_capture_bundle(RouteCaptureBundleImportRequest(...))` as the only ground-truth bundle import contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#route-capture-bundle-import](experiments.generated.md#route-capture-bundle-import)

## Sim2Real Websocket Protocol

Stable: Stable web code uses `importSim2realWebsocketMessage(rawMessage, policy?)` as the only websocket response import contract.

Deferred strategies: 2

Detailed comparison: [experiments.generated.md#sim2real-websocket-protocol](experiments.generated.md#sim2real-websocket-protocol)

## Regeneration

Run one experiment lab with `--write-docs` to refresh this index,
`experiments.generated.md`, `decisions.md`, and `interfaces.md`:

```bash
3dgs-robotics experiment render-backend-selection --write-docs --output outputs/render-backend-selection-experiment-report.json
```

Use `3dgs-robotics experiment --help` for the full lab list.
