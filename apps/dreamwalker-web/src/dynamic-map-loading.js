import {
  dreamwalkerConfig,
  resolveDreamwalkerConfig,
  resolveWorldAssetBundle
} from './app-config.js';
import { normalizeLocalAssetPath } from './studio-health.js';

export const dynamicMapPreloadModes = ['off', 'metadata', 'cache'];

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function normalizeFragmentId(value) {
  return hasNonEmptyString(value) ? value.trim() : '';
}

function pushUniqueFragmentId(targets, fragmentId, knownFragmentIds) {
  const normalizedFragmentId = normalizeFragmentId(fragmentId);

  if (!normalizedFragmentId) {
    return;
  }

  if (knownFragmentIds && !knownFragmentIds.has(normalizedFragmentId)) {
    return;
  }

  if (!targets.includes(normalizedFragmentId)) {
    targets.push(normalizedFragmentId);
  }
}

function classifyAssetUrl(assetUrl) {
  if (!hasNonEmptyString(assetUrl)) {
    return 'missing';
  }

  return normalizeLocalAssetPath(assetUrl) ? 'local' : 'remote';
}

function normalizeCatalogTile(tileLike, index) {
  const tile = tileLike && typeof tileLike === 'object' ? tileLike : {};
  const id = hasNonEmptyString(tile.id) ? tile.id.trim() : `tile-${index + 1}`;
  const splatUrl = hasNonEmptyString(tile.splatUrl) ? tile.splatUrl.trim() : '';
  const status = hasNonEmptyString(tile.status)
    ? tile.status.trim()
    : splatUrl
      ? 'ready'
      : 'missing-splat';

  return {
    ...tile,
    id,
    label: hasNonEmptyString(tile.label) ? tile.label.trim() : id.replaceAll('_', ' '),
    status,
    runStatus: hasNonEmptyString(tile.runStatus) ? tile.runStatus.trim() : 'unknown',
    splatUrl,
    sourceSplat: hasNonEmptyString(tile.sourceSplat) ? tile.sourceSplat.trim() : '',
    publicPath: hasNonEmptyString(tile.publicPath) ? tile.publicPath.trim() : '',
    splatAssetKind: classifyAssetUrl(splatUrl),
    coreBounds: tile.coreBounds && typeof tile.coreBounds === 'object' ? tile.coreBounds : {},
    expandedBounds:
      tile.expandedBounds && typeof tile.expandedBounds === 'object' ? tile.expandedBounds : {},
    tileIndex: tile.tileIndex && typeof tile.tileIndex === 'object' ? tile.tileIndex : {}
  };
}

function isRuntimeReadyTile(tile) {
  return hasNonEmptyString(tile?.splatUrl) && tile.status !== 'missing-splat';
}

function normalizeMapPosition(positionLike) {
  if (Array.isArray(positionLike)) {
    const [x, y, z] = positionLike.map((value) => Number(value));
    if ([x, y, z].every(Number.isFinite)) {
      return { x, y, z };
    }
  }

  if (positionLike && typeof positionLike === 'object') {
    const x = Number(positionLike.x ?? positionLike.position?.[0]);
    const y = Number(positionLike.y ?? positionLike.position?.[1]);
    const z = Number(positionLike.z ?? positionLike.position?.[2]);
    if ([x, y, z].every(Number.isFinite)) {
      return { x, y, z };
    }
  }

  return null;
}

function tileAxes(tile) {
  const axes = hasNonEmptyString(tile?.axes) ? tile.axes.trim().toLowerCase() : '';
  if (axes.length >= 2) {
    return axes.slice(0, 2).split('');
  }

  const boundKeys = Object.keys(tile?.coreBounds ?? {});
  const discoveredAxes = boundKeys
    .map((key) => key.match(/^min([XYZ])$/)?.[1]?.toLowerCase())
    .filter(Boolean);

  return discoveredAxes.length >= 2 ? discoveredAxes.slice(0, 2) : ['x', 'z'];
}

function tileBoundsForAxes(tile, boundsName) {
  const bounds = tile?.[boundsName] && typeof tile[boundsName] === 'object' ? tile[boundsName] : {};
  const axes = tileAxes(tile);

  return axes.map((axis) => {
    const upperAxis = axis.toUpperCase();
    const min = Number(bounds[`min${upperAxis}`]);
    const max = Number(bounds[`max${upperAxis}`]);

    return {
      axis,
      min,
      max
    };
  });
}

function isPositionInsideTileBounds(tile, positionLike, boundsName) {
  const position = normalizeMapPosition(positionLike);
  if (!position) {
    return false;
  }

  const bounds = tileBoundsForAxes(tile, boundsName);
  if (bounds.some((bound) => !Number.isFinite(bound.min) || !Number.isFinite(bound.max))) {
    return false;
  }

  return bounds.every((bound) => position[bound.axis] >= bound.min && position[bound.axis] <= bound.max);
}

function tileBoundsCenter(tile, boundsName = 'coreBounds') {
  const bounds = tileBoundsForAxes(tile, boundsName);
  if (bounds.some((bound) => !Number.isFinite(bound.min) || !Number.isFinite(bound.max))) {
    return null;
  }

  return Object.fromEntries(
    bounds.map((bound) => [bound.axis, (bound.min + bound.max) / 2])
  );
}

function tileDistanceToPosition(tile, positionLike) {
  const position = normalizeMapPosition(positionLike);
  const center = tileBoundsCenter(tile);
  if (!position || !center) {
    return Number.POSITIVE_INFINITY;
  }

  return Math.sqrt(
    tileAxes(tile).reduce((sum, axis) => {
      const delta = position[axis] - center[axis];
      return sum + delta * delta;
    }, 0)
  );
}

function sortTilesByPositionDistance(tiles, positionLike) {
  const position = normalizeMapPosition(positionLike);
  if (!position) {
    return tiles;
  }

  return [...tiles].sort(
    (left, right) =>
      tileDistanceToPosition(left, position) - tileDistanceToPosition(right, position)
  );
}

function normalizeMapPositions(positionsLike) {
  return Array.isArray(positionsLike)
    ? positionsLike.map((position) => normalizeMapPosition(position)).filter(Boolean)
    : [];
}

function selectDynamicMapTileForPreviewPosition(tiles, positionLike) {
  const position = normalizeMapPosition(positionLike);
  if (!position) {
    return null;
  }

  const coreMatches = tiles.filter((tile) =>
    isPositionInsideTileBounds(tile, position, 'coreBounds')
  );
  if (coreMatches.length > 0) {
    return sortTilesByPositionDistance(coreMatches, position)[0];
  }

  const expandedMatches = tiles.filter((tile) =>
    isPositionInsideTileBounds(tile, position, 'expandedBounds')
  );
  if (expandedMatches.length > 0) {
    return sortTilesByPositionDistance(expandedMatches, position)[0];
  }

  return null;
}

function collectRoutePreviewTileCandidates(tiles, activeTile, routePreviewPositions) {
  const candidates = [];
  const seenTileIds = new Set([activeTile?.id].filter(Boolean));

  for (const position of normalizeMapPositions(routePreviewPositions)) {
    const tile = selectDynamicMapTileForPreviewPosition(tiles, position);
    if (!tile || seenTileIds.has(tile.id)) {
      continue;
    }

    seenTileIds.add(tile.id);
    candidates.push(tile);
  }

  return candidates;
}

function tileIndexForAxes(tile, axes) {
  const tileIndex = tile?.tileIndex && typeof tile.tileIndex === 'object' ? tile.tileIndex : {};
  const values = axes.map((axis) => Number(tileIndex[axis]));

  return values.every(Number.isFinite) ? values : null;
}

function tileIndexDistance(candidateTile, activeTile) {
  const axes = tileAxes(activeTile);
  const activeIndex = tileIndexForAxes(activeTile, axes);
  const candidateIndex = tileIndexForAxes(candidateTile, axes);

  if (!activeIndex || !candidateIndex) {
    return null;
  }

  const deltas = activeIndex.map((value, index) => Math.abs(candidateIndex[index] - value));
  const chebyshev = Math.max(...deltas);
  const manhattan = deltas.reduce((sum, value) => sum + value, 0);

  return {
    chebyshev,
    manhattan
  };
}

function sortTilePreloadCandidates(tiles, activeTile, positionLike) {
  return [...tiles].sort((left, right) => {
    const leftIndexDistance = tileIndexDistance(left, activeTile);
    const rightIndexDistance = tileIndexDistance(right, activeTile);
    const leftHasIndex = leftIndexDistance ? 0 : 1;
    const rightHasIndex = rightIndexDistance ? 0 : 1;

    if (leftHasIndex !== rightHasIndex) {
      return leftHasIndex - rightHasIndex;
    }

    if (leftIndexDistance && rightIndexDistance) {
      const chebyshevDelta = leftIndexDistance.chebyshev - rightIndexDistance.chebyshev;
      if (chebyshevDelta !== 0) {
        return chebyshevDelta;
      }

      const manhattanDelta = leftIndexDistance.manhattan - rightIndexDistance.manhattan;
      if (manhattanDelta !== 0) {
        return manhattanDelta;
      }
    }

    const distanceDelta =
      tileDistanceToPosition(left, positionLike) - tileDistanceToPosition(right, positionLike);
    if (Number.isFinite(distanceDelta) && distanceDelta !== 0) {
      return distanceDelta;
    }

    return left.id.localeCompare(right.id);
  });
}

function buildTileCatalogLoadEntry(tile, baseEntry, catalog, role) {
  const tileLabel = hasNonEmptyString(tile.label) ? tile.label : tile.id;
  const assetLabel = `${catalog.label} / ${tileLabel}`;
  const assetBundle = {
    ...baseEntry.assetBundle,
    assetLabel,
    worldNote: `large-scale 3DGS tile ${tile.id}`,
    manifestLabel: catalog.label,
    splatUrl: tile.splatUrl,
    splatSource: 'tile-catalog',
    usesDemoFallback: false,
    hasConfiguredSplat: true,
    tileCatalog: {
      sceneId: catalog.sceneId,
      label: catalog.label,
      tileId: tile.id,
      tileStatus: tile.status,
      readyTileCount: catalog.summary.readyTileCount,
      tileCount: catalog.summary.tileCount
    },
    sourceTile: tile
  };

  return {
    ...baseEntry,
    role,
    assetLabel,
    splatUrl: tile.splatUrl,
    splatSource: 'tile-catalog',
    splatAssetKind: classifyAssetUrl(tile.splatUrl),
    usesDemoFallback: false,
    hasConfiguredSplat: true,
    tileId: tile.id,
    tileLabel,
    tileStatus: tile.status,
    tileCatalogSceneId: catalog.sceneId,
    tileCatalogLabel: catalog.label,
    preloadCacheKey: `${catalog.sceneId}:${tile.id}:${tile.splatUrl}`,
    sourceTile: tile,
    assetBundle
  };
}

function isAbortError(error) {
  return error?.name === 'AbortError';
}

function normalizeFetchResponseStatus(response) {
  return {
    ok: Boolean(response?.ok),
    status: Number(response?.status ?? 0),
    statusText: hasNonEmptyString(response?.statusText) ? response.statusText.trim() : ''
  };
}

function buildFetchErrorMessage(response) {
  const normalized = normalizeFetchResponseStatus(response);
  const statusText = normalized.statusText ? ` ${normalized.statusText}` : '';
  return `HTTP ${normalized.status}${statusText}`.trim();
}

export function normalizeDynamicMapPreloadMode(value, fallback = 'metadata') {
  const normalizedFallback = dynamicMapPreloadModes.includes(fallback) ? fallback : 'metadata';

  if (!hasNonEmptyString(value)) {
    return normalizedFallback;
  }

  const normalized = value.trim().toLowerCase();

  if (['0', 'false', 'off', 'none', 'disabled'].includes(normalized)) {
    return 'off';
  }

  if (['1', 'true', 'cache', 'warm', 'full'].includes(normalized)) {
    return 'cache';
  }

  if (['head', 'metadata', 'probe', 'check'].includes(normalized)) {
    return 'metadata';
  }

  return normalizedFallback;
}

export function normalizeDynamicMapTileCatalog(catalogLike) {
  const catalog = catalogLike && typeof catalogLike === 'object' ? catalogLike : {};
  const tiles = Array.isArray(catalog.tiles)
    ? catalog.tiles.map((tile, index) => normalizeCatalogTile(tile, index))
    : [];
  const readyTiles = tiles.filter(isRuntimeReadyTile);

  return {
    version: Number(catalog.version ?? 1),
    type: hasNonEmptyString(catalog.type)
      ? catalog.type.trim()
      : 'large-scale-3dgs-tile-catalog',
    sceneId: hasNonEmptyString(catalog.sceneId)
      ? catalog.sceneId.trim()
      : 'large-scale-3dgs',
    label: hasNonEmptyString(catalog.label)
      ? catalog.label.trim()
      : 'Large-scale 3DGS',
    planPath: hasNonEmptyString(catalog.planPath) ? catalog.planPath.trim() : '',
    runReportPath: hasNonEmptyString(catalog.runReportPath) ? catalog.runReportPath.trim() : '',
    tiling: catalog.tiling && typeof catalog.tiling === 'object' ? catalog.tiling : {},
    summary: {
      tileCount: tiles.length,
      readyTileCount: readyTiles.length,
      missingSplatTileCount: tiles.filter((tile) => tile.status === 'missing-splat').length
    },
    tiles
  };
}

export function selectDynamicMapTile(catalogLike, options = {}) {
  const catalog = normalizeDynamicMapTileCatalog(catalogLike);
  const requestedTileId = normalizeFragmentId(options.tileId);
  const currentTileId = normalizeFragmentId(options.currentTileId);
  const position = normalizeMapPosition(options.position);
  const readyTiles = catalog.tiles.filter(isRuntimeReadyTile);

  if (requestedTileId) {
    const requestedTile = readyTiles.find((tile) => tile.id === requestedTileId);
    if (requestedTile && isRuntimeReadyTile(requestedTile)) {
      return requestedTile;
    }
  }

  if (position) {
    const currentTile = readyTiles.find((tile) => tile.id === currentTileId);
    if (
      currentTile &&
      isPositionInsideTileBounds(currentTile, position, 'expandedBounds')
    ) {
      return currentTile;
    }

    const coreMatch = readyTiles.find((tile) =>
      isPositionInsideTileBounds(tile, position, 'coreBounds')
    );
    if (coreMatch) {
      return coreMatch;
    }

    const expandedMatches = readyTiles.filter((tile) =>
      isPositionInsideTileBounds(tile, position, 'expandedBounds')
    );
    if (expandedMatches.length > 0) {
      return sortTilesByPositionDistance(expandedMatches, position)[0];
    }
  }

  return readyTiles[0] ?? null;
}

export function collectDynamicMapTargetFragmentIds(activeConfig, options = {}) {
  const knownFragmentIds = new Set(
    Object.keys(options.fragments ?? dreamwalkerConfig.fragments)
  );
  const targets = [];

  pushUniqueFragmentId(targets, activeConfig?.fragmentId, knownFragmentIds);

  if (options.includeGateTarget !== false) {
    pushUniqueFragmentId(targets, activeConfig?.gate?.targetFragmentId, knownFragmentIds);
  }

  for (const fragmentId of options.extraFragmentIds ?? []) {
    pushUniqueFragmentId(targets, fragmentId, knownFragmentIds);
  }

  return targets;
}

export function buildDynamicMapLoadEntry(fragmentId, role, assetManifest) {
  const config = resolveDreamwalkerConfig(fragmentId);
  const assetBundle = resolveWorldAssetBundle(config, assetManifest);

  return {
    role,
    fragmentId: config.fragmentId,
    fragmentLabel: config.fragmentLabel,
    assetLabel: assetBundle.assetLabel,
    splatUrl: assetBundle.splatUrl,
    colliderMeshUrl: assetBundle.colliderMeshUrl,
    splatSource: assetBundle.splatSource,
    colliderSource: assetBundle.colliderSource,
    splatAssetKind: classifyAssetUrl(assetBundle.splatUrl),
    colliderAssetKind: classifyAssetUrl(assetBundle.colliderMeshUrl),
    usesDemoFallback: assetBundle.usesDemoFallback,
    hasConfiguredSplat: assetBundle.hasConfiguredSplat,
    hasColliderMesh: assetBundle.hasColliderMesh,
    assetBundle
  };
}

export function buildDynamicMapLoadPlan(activeConfig, assetManifest, options = {}) {
  const targetFragmentIds = collectDynamicMapTargetFragmentIds(activeConfig, options);
  const entries = targetFragmentIds.map((fragmentId, index) =>
    buildDynamicMapLoadEntry(fragmentId, index === 0 ? 'active' : 'preload-candidate', assetManifest)
  );
  let active = entries[0] ?? null;
  const tileCatalog = normalizeDynamicMapTileCatalog(options.tileCatalog);
  const activeTile = active
    ? selectDynamicMapTile(tileCatalog, {
        currentTileId: options.currentTileId,
        position: options.position,
        tileId: options.tileId
      })
    : null;
  const maxTilePreloadCandidates = Math.max(
    0,
    Number.isFinite(options.maxTilePreloadCandidates)
      ? Number(options.maxTilePreloadCandidates)
      : 4
  );
  const maxResidentTiles = Math.max(
    1,
    Number.isFinite(options.maxResidentTiles)
      ? Number(options.maxResidentTiles)
      : maxTilePreloadCandidates + 1
  );

  if (active && activeTile) {
    active = buildTileCatalogLoadEntry(activeTile, active, tileCatalog, 'active-tile');
    entries[0] = active;
  }

  const preloadCandidates = entries.slice(1);
  const readyTilePreloadCandidates =
    active && activeTile
      ? tileCatalog.tiles.filter((tile) => tile.id !== activeTile.id && isRuntimeReadyTile(tile))
      : [];
  const routePreviewTilePreloadCandidates =
    active && activeTile
      ? collectRoutePreviewTileCandidates(
          readyTilePreloadCandidates,
          activeTile,
          options.routePreviewPositions
        )
      : [];
  const routePreviewTileIds = new Set(
    routePreviewTilePreloadCandidates.map((tile) => tile.id)
  );
  const sortedTilePreloadCandidateTiles =
    active && activeTile
      ? [
          ...routePreviewTilePreloadCandidates,
          ...sortTilePreloadCandidates(
            readyTilePreloadCandidates.filter((tile) => !routePreviewTileIds.has(tile.id)),
            activeTile,
            options.position
          )
        ]
      : [];
  const maxResidentPreloadCandidates = Math.max(0, maxResidentTiles - (activeTile ? 1 : 0));
  const effectiveTilePreloadLimit = Math.min(
    maxTilePreloadCandidates,
    maxResidentPreloadCandidates
  );
  const tilePreloadCandidates =
    active && activeTile
      ? sortedTilePreloadCandidateTiles
          .slice(0, effectiveTilePreloadLimit)
          .map((tile) =>
            buildTileCatalogLoadEntry(
              tile,
              active,
              tileCatalog,
              routePreviewTileIds.has(tile.id) ? 'route-preload-tile' : 'preload-tile'
            )
          )
      : [];
  const residentTileIds = new Set(
    [
      active && activeTile ? activeTile.id : '',
      ...tilePreloadCandidates.map((entry) => entry.tileId)
    ].filter(Boolean)
  );
  const tileResidentCandidates = active && activeTile ? [active, ...tilePreloadCandidates] : [];
  const tileEvictionCandidates =
    active && activeTile
      ? sortedTilePreloadCandidateTiles
          .filter((tile) => !residentTileIds.has(tile.id))
          .map((tile) => buildTileCatalogLoadEntry(tile, active, tileCatalog, 'evicted-tile'))
      : [];

  return {
    strategy: activeTile ? 'large-scale-3dgs-tile-catalog' : 'active-fragment-on-demand',
    sourceLabel: activeTile
      ? tileCatalog.label
      : hasNonEmptyString(assetManifest?.label)
        ? assetManifest.label.trim()
        : '',
    active,
    activeTile,
    maxResidentTiles,
    tileCatalog: tileCatalog.tiles.length ? tileCatalog : null,
    tileResidentCandidates,
    tileEvictionCandidates,
    tilePreloadCandidates,
    preloadCandidates,
    entries,
    runtimeKey: active
      ? active.tileId
        ? `${active.fragmentId}:${active.tileId}:${active.splatUrl}:${active.colliderMeshUrl}`
        : `${active.fragmentId}:${active.splatUrl}:${active.colliderMeshUrl}`
      : ''
  };
}

export function collectDynamicMapPreloadAssets(entry, options = {}) {
  const assets = [];

  if (!entry || typeof entry !== 'object') {
    return assets;
  }

  if (hasNonEmptyString(entry.splatUrl)) {
    assets.push({
      kind: 'splat',
      url: entry.splatUrl.trim()
    });
  }

  if (options.includeCollider && hasNonEmptyString(entry.colliderMeshUrl)) {
    assets.push({
      kind: 'collider',
      url: entry.colliderMeshUrl.trim()
    });
  }

  return assets;
}

export async function preloadDynamicMapAsset(asset, options = {}) {
  const mode = normalizeDynamicMapPreloadMode(options.mode);
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;

  if (mode === 'off') {
    return {
      kind: asset?.kind ?? 'asset',
      url: asset?.url ?? '',
      status: 'skipped',
      detail: 'map preload は無効です。'
    };
  }

  if (!asset || !hasNonEmptyString(asset.url)) {
    return {
      kind: asset?.kind ?? 'asset',
      url: '',
      status: 'skipped',
      detail: 'preload asset URL がありません。'
    };
  }

  if (typeof fetchImpl !== 'function') {
    return {
      kind: asset.kind,
      url: asset.url,
      status: 'skipped',
      detail: 'fetch が使えないため preload を省略しました。'
    };
  }

  const method = mode === 'cache' ? 'GET' : 'HEAD';

  try {
    const response = await fetchImpl(asset.url, {
      cache: mode === 'cache' ? 'force-cache' : 'no-store',
      headers: {
        Accept: '*/*'
      },
      method,
      signal: options.signal
    });

    if (!response?.ok) {
      throw new Error(buildFetchErrorMessage(response));
    }

    return {
      kind: asset.kind,
      url: asset.url,
      status: 'ready',
      method,
      detail:
        mode === 'cache'
          ? `${asset.kind} を HTTP cache へ warm しました。`
          : `${asset.kind} metadata を確認しました。`
    };
  } catch (error) {
    if (isAbortError(error)) {
      return {
        kind: asset.kind,
        url: asset.url,
        status: 'aborted',
        method,
        detail: `${asset.kind} preload を中断しました。`
      };
    }

    if (mode === 'metadata' && method === 'HEAD') {
      try {
        const response = await fetchImpl(asset.url, {
          cache: 'no-store',
          headers: {
            Accept: '*/*',
            Range: 'bytes=0-0'
          },
          method: 'GET',
          signal: options.signal
        });

        if (!response?.ok) {
          throw new Error(buildFetchErrorMessage(response));
        }

        return {
          kind: asset.kind,
          url: asset.url,
          status: 'ready',
          method: 'GET',
          detail: `${asset.kind} metadata を range fallback で確認しました。`
        };
      } catch (fallbackError) {
        if (isAbortError(fallbackError)) {
          return {
            kind: asset.kind,
            url: asset.url,
            status: 'aborted',
            method: 'GET',
            detail: `${asset.kind} preload を中断しました。`
          };
        }

        return {
          kind: asset.kind,
          url: asset.url,
          status: 'error',
          method: 'GET',
          detail:
            fallbackError instanceof Error
              ? fallbackError.message
              : String(fallbackError)
        };
      }
    }

    return {
      kind: asset.kind,
      url: asset.url,
      status: 'error',
      method,
      detail: error instanceof Error ? error.message : String(error)
    };
  }
}

export async function preloadDynamicMapEntry(entry, options = {}) {
  const mode = normalizeDynamicMapPreloadMode(options.mode);

  if (!entry) {
    return {
      status: 'skipped',
      label: 'No Candidate',
      detail: '事前読み込み候補の fragment がありません。',
      assets: []
    };
  }

  if (mode === 'off') {
    return {
      status: 'skipped',
      label: 'Preload Off',
      detail: 'map preload は無効です。',
      fragmentId: entry.fragmentId,
      fragmentLabel: entry.fragmentLabel,
      assets: []
    };
  }

  if (entry.usesDemoFallback && options.includeDemoFallback !== true) {
    return {
      status: 'skipped',
      label: 'Demo Skipped',
      detail: 'demo fallback map は事前読み込みしません。',
      fragmentId: entry.fragmentId,
      fragmentLabel: entry.fragmentLabel,
      assets: []
    };
  }

  const assets = collectDynamicMapPreloadAssets(entry, {
    includeCollider: Boolean(options.includeCollider)
  });

  if (assets.length === 0) {
    return {
      status: 'skipped',
      label: 'No Asset',
      detail: '事前読み込みできる map asset がありません。',
      fragmentId: entry.fragmentId,
      fragmentLabel: entry.fragmentLabel,
      assets: []
    };
  }

  const results = await Promise.all(
    assets.map((asset) =>
      preloadDynamicMapAsset(asset, {
        fetchImpl: options.fetchImpl,
        mode,
        signal: options.signal
      })
    )
  );

  if (results.some((result) => result.status === 'aborted')) {
    return {
      status: 'aborted',
      label: 'Preload Aborted',
      detail: `${entry.fragmentLabel} preload を中断しました。`,
      fragmentId: entry.fragmentId,
      fragmentLabel: entry.fragmentLabel,
      assets: results
    };
  }

  if (results.some((result) => result.status === 'error')) {
    return {
      status: 'warning',
      label: 'Preload Warning',
      detail: `${entry.fragmentLabel} preload に失敗した asset があります。`,
      fragmentId: entry.fragmentId,
      fragmentLabel: entry.fragmentLabel,
      assets: results
    };
  }

  return {
    status: 'ready',
    label: mode === 'cache' ? 'Cache Warmed' : 'Preload Ready',
    detail:
      mode === 'cache'
        ? `${entry.fragmentLabel} map を cache warm しました。`
        : `${entry.fragmentLabel} map metadata を確認しました。`,
    fragmentId: entry.fragmentId,
    fragmentLabel: entry.fragmentLabel,
    assets: results
  };
}
