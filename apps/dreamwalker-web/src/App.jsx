import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  dreamwalkerConfig,
  resolveDreamwalkerConfig,
  resolveWorldAssetBundle
} from './app-config.js';
import {
  buildWorldAssetHealth,
  normalizeLocalAssetPath,
  resolveBundleWorldHealth
} from './studio-health.js';
import {
  buildDynamicMapLoadPlan,
  normalizeDynamicMapPreloadMode,
  preloadDynamicMapEntry
} from './dynamic-map-loading.js';
import {
  buildRelayEndpoint,
  loadOverlayState,
  overlayRelayDefaultUrl,
  overlayStateKey,
  parseOverlayRelayConfigFromSearch
} from './overlay-shared.js';
import {
  buildCameraFrameMessage,
  buildDepthFrameMessage,
  parseRobotBridgeMessage,
  parseRobotBridgeConfigFromSearch,
  robotBridgeBrowserSource,
  robotBridgeDefaultUrl,
  stringifyRobotBridgeMessage
} from './robotics-bridge.js';
import { parseSim2realConfigFromSearch } from './sim2real-query.js';
import {
  buildSemanticZoneMap,
  buildSemanticZoneProjectionPoints,
  buildSemanticZoneSurfacePoints,
  findSemanticZoneHits,
  serializeSemanticZoneMap,
  summarizeSemanticZoneHits
} from './semantic-zones.js';
import {
  normalizeOverlayMemoItems,
  ObsOverlayView,
  OverlayStage,
  resolveOverlayBrandingForScene
} from './overlay-ui.jsx';
import Sim2RealPanel from './Sim2RealPanel.jsx';

const captureKey = 'k';
const guideKey = 'g';
const homeKey = 'h';
const interactKey = 'f';
const liveKey = 'l';
const overlayPresetKeys = ['7', '8', '9'];
const photoKey = 'p';
const robotKey = 'r';
const streamSceneKeys = ['4', '5', '6'];
const clearRouteKey = 'c';
const waypointKey = 'v';
const walkKey = 'x';
const overlayPresetStorageKey = 'dreamwalker-live-overlay-preset';
const assetWorkspaceStorageKey = 'dreamwalker-live-asset-workspace';
const sceneWorkspaceStorageKey = 'dreamwalker-live-scene-workspace';
const semanticZoneWorkspaceStorageKey = 'dreamwalker-live-semantic-zone-workspace';
const studioBundleShelfStorageKey = 'dreamwalker-live-studio-bundle-shelf';
const robotRouteShelfStorageKey = 'dreamwalker-live-robot-route-shelf';
const robotMissionDraftBundleShelfStorageKey = 'dreamwalker-live-robot-mission-draft-bundle-shelf';
const shardStorageKeyPrefix = 'dreamwalker-live-collected-shards';
const robotRouteProtocolId = 'dreamwalker-robot-route/v1';
const robotMissionProtocolId = 'dreamwalker-robot-mission/v1';
const robotMissionArtifactPackProtocolId = 'dreamwalker-robot-mission-artifact-pack/v1';
const robotMissionStartupModes = ['explore', 'live', 'photo', 'robot'];
const DreamwalkerScene = lazy(() => import('./DreamwalkerScene.jsx'));

function parseFragmentIdFromHash() {
  if (typeof window === 'undefined') {
    return dreamwalkerConfig.defaultFragmentId;
  }

  const rawHash = window.location.hash.replace(/^#/, '').trim();
  if (!rawHash) {
    return dreamwalkerConfig.defaultFragmentId;
  }

  if (rawHash.startsWith('fragment=')) {
    return rawHash.slice('fragment='.length) || dreamwalkerConfig.defaultFragmentId;
  }

  return rawHash;
}

function buildShardStorageKey(fragmentId) {
  return `${shardStorageKeyPrefix}:${fragmentId}`;
}

function parseOverlayModeFromSearch() {
  if (typeof window === 'undefined') {
    return false;
  }

  const searchParams = new URLSearchParams(window.location.search);
  return searchParams.get('overlay') === '1';
}

function parseAssetManifestUrlFromSearch(defaultUrl, queryParam) {
  if (typeof window === 'undefined') {
    return defaultUrl;
  }

  const searchParams = new URLSearchParams(window.location.search);
  const override = searchParams.get(queryParam)?.trim();
  return override || defaultUrl;
}

function parseDynamicMapTilePreloadModeFromSearch() {
  if (typeof window === 'undefined') {
    return 'metadata';
  }

  const searchParams = new URLSearchParams(window.location.search);
  return normalizeDynamicMapPreloadMode(searchParams.get('tilePreload'), 'metadata');
}

function parseDynamicMapTilePreloadLimitFromSearch() {
  if (typeof window === 'undefined') {
    return 4;
  }

  const searchParams = new URLSearchParams(window.location.search);
  const rawValue = searchParams.get(dreamwalkerConfig.tileCatalog.preloadLimitQueryParam);
  if (rawValue === null || rawValue.trim() === '') {
    return 4;
  }

  const value = Number(rawValue);
  if (!Number.isFinite(value)) {
    return 4;
  }

  return Math.max(0, Math.min(32, Math.floor(value)));
}

function parseDynamicMapTileResidentLimitFromSearch(defaultLimit) {
  const fallbackLimit = Number.isFinite(defaultLimit) ? Math.max(1, Math.floor(defaultLimit)) : 5;
  if (typeof window === 'undefined') {
    return fallbackLimit;
  }

  const searchParams = new URLSearchParams(window.location.search);
  const rawValue = searchParams.get(dreamwalkerConfig.tileCatalog.residentLimitQueryParam);
  if (rawValue === null || rawValue.trim() === '') {
    return fallbackLimit;
  }

  const value = Number(rawValue);
  if (!Number.isFinite(value)) {
    return fallbackLimit;
  }

  return Math.max(1, Math.min(64, Math.floor(value)));
}

function parseBooleanSearchFlag(value, truthyValues = ['1', 'true', 'yes', 'on']) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  return truthyValues.includes(normalized);
}

function parseDynamicMapDiagnosticsEnabledFromSearch() {
  if (typeof window === 'undefined') {
    return false;
  }

  const searchParams = new URLSearchParams(window.location.search);
  return parseBooleanSearchFlag(searchParams.get('dynamicMapDiagnostics'));
}

function serializeDiagnosticMapPosition(positionLike) {
  const x = Number(positionLike?.x ?? positionLike?.[0]);
  const y = Number(positionLike?.y ?? positionLike?.[1] ?? 0);
  const z = Number(positionLike?.z ?? positionLike?.[2]);

  if (![x, y, z].every(Number.isFinite)) {
    return null;
  }

  return {
    x: Number(x.toFixed(2)),
    y: Number(y.toFixed(2)),
    z: Number(z.toFixed(2))
  };
}

function serializeDynamicMapDiagnosticsTile(tile) {
  if (!tile || typeof tile !== 'object') {
    return null;
  }

  return {
    id: tile.id ?? '',
    label: tile.label ?? tile.id ?? '',
    status: tile.status ?? '',
    splatUrl: tile.splatUrl ?? '',
    tileIndex: tile.tileIndex && typeof tile.tileIndex === 'object' ? { ...tile.tileIndex } : {},
    coreBounds: tile.coreBounds && typeof tile.coreBounds === 'object' ? { ...tile.coreBounds } : null,
    expandedBounds:
      tile.expandedBounds && typeof tile.expandedBounds === 'object'
        ? { ...tile.expandedBounds }
        : null
  };
}

function serializeDynamicMapDiagnosticsEntry(entry, index) {
  return {
    index,
    role: entry.role ?? '',
    fragmentId: entry.fragmentId ?? '',
    tileId: entry.tileId ?? '',
    tileLabel: entry.tileLabel ?? entry.fragmentLabel ?? '',
    tileStatus: entry.tileStatus ?? '',
    splatUrl: entry.splatUrl ?? '',
    colliderMeshUrl: entry.colliderMeshUrl ?? '',
    preloadCacheKey: entry.preloadCacheKey ?? '',
    tileCatalogSceneId: entry.tileCatalogSceneId ?? '',
    sourceTile: serializeDynamicMapDiagnosticsTile(entry.sourceTile)
  };
}

function parseRobotRoutePlaybackConfigFromSearch() {
  const defaultConfig = {
    enabled: false,
    intervalMs: 700,
    loop: false
  };

  if (typeof window === 'undefined') {
    return defaultConfig;
  }

  const searchParams = new URLSearchParams(window.location.search);
  const enabled = parseBooleanSearchFlag(
    searchParams.get(dreamwalkerConfig.robotRoute.playbackQueryParam),
    ['1', 'true', 'yes', 'on', 'play', 'auto']
  );
  const rawInterval = Number(
    searchParams.get(dreamwalkerConfig.robotRoute.playbackIntervalQueryParam)
  );
  const intervalMs = Number.isFinite(rawInterval)
    ? Math.max(250, Math.min(10_000, Math.floor(rawInterval)))
    : defaultConfig.intervalMs;

  return {
    enabled,
    intervalMs,
    loop: parseBooleanSearchFlag(
      searchParams.get(dreamwalkerConfig.robotRoute.playbackLoopQueryParam)
    )
  };
}

const dynamicMapTilePreloadResultCache = new Map();

function buildDynamicMapTilePreloadCacheKey({
  catalogUrl = '',
  colliderMeshUrl = '',
  mode,
  preloadCacheKey = '',
  sceneId = '',
  splatUrl = '',
  tileId = ''
}) {
  return [
    mode,
    catalogUrl,
    preloadCacheKey,
    sceneId,
    tileId,
    splatUrl,
    colliderMeshUrl
  ].join('|');
}

function parseRobotFrameStreamEnabledFromSearch() {
  if (typeof window === 'undefined') {
    return false;
  }

  const searchParams = new URLSearchParams(window.location.search);
  const streamParam = searchParams.get('robotFrameStream')?.trim().toLowerCase() ?? '';

  return streamParam === '1' || streamParam === 'true';
}

function parseRobotDepthStreamEnabledFromSearch() {
  if (typeof window === 'undefined') {
    return false;
  }

  const searchParams = new URLSearchParams(window.location.search);
  const streamParam = searchParams.get('robotDepthStream')?.trim().toLowerCase() ?? '';

  return streamParam === '1' || streamParam === 'true';
}

function loadOverlayPresetId() {
  if (typeof window === 'undefined') {
    return dreamwalkerConfig.overlayPresets[0]?.id ?? 'lower-third';
  }

  try {
    const raw = window.localStorage.getItem(overlayPresetStorageKey);
    return dreamwalkerConfig.overlayPresets.some((preset) => preset.id === raw)
      ? raw
      : dreamwalkerConfig.overlayPresets[0]?.id ?? 'lower-third';
  } catch {
    return dreamwalkerConfig.overlayPresets[0]?.id ?? 'lower-third';
  }
}

function loadCollectedShards(storageKey) {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((value) => typeof value === 'string') : [];
  } catch {
    return [];
  }
}

function normalizeStringList(items) {
  return Array.isArray(items)
    ? items.filter((item) => typeof item === 'string' && item.trim())
    : [];
}

function buildAssetManifestTemplate() {
  const fragments = Object.fromEntries(
    Object.entries(dreamwalkerConfig.fragments).map(([fragmentId, fragment]) => [
      fragmentId,
      {
        label: fragment.assetBundle?.label ?? `${fragment.fragmentLabel} World`,
        splatUrl: '',
        colliderMeshUrl: '',
        expectedSplatUrl: fragment.assetBundle?.expectedSplatUrl ?? '',
        expectedColliderMeshUrl: fragment.assetBundle?.expectedColliderMeshUrl ?? '',
        worldNote: fragment.assetBundle?.worldNote ?? ''
      }
    ])
  );

  return {
    version: 1,
    label: 'Local DreamWalker Asset Manifest',
    note:
      'splatUrl と colliderMeshUrl に実ファイルを入れると、fragment ごとに Marble world を差し替えられる。',
    fragments
  };
}

const defaultAssetManifestTemplate = buildAssetManifestTemplate();

function buildSceneWorkspaceTemplate() {
  const fragments = Object.fromEntries(
    Object.entries(dreamwalkerConfig.fragments).map(([fragmentId, fragment]) => [
      fragmentId,
      {
        label: fragment.fragmentLabel ?? fragmentId,
        streamScenes: (fragment.streamScenes ?? []).map((streamScene) => ({
          id: streamScene.id,
          label: streamScene.label ?? '',
          title: streamScene.title ?? '',
          topic: streamScene.topic ?? '',
          presetId: streamScene.presetId ?? '',
          overlayMemo: {
            title: streamScene.overlayMemo?.title ?? '',
            items: normalizeStringList(streamScene.overlayMemo?.items),
            footer: streamScene.overlayMemo?.footer ?? ''
          },
          overlayBrandingOverrides: {
            badge: streamScene.overlayBrandingOverrides?.badge ?? '',
            strapline: streamScene.overlayBrandingOverrides?.strapline ?? '',
            accent: streamScene.overlayBrandingOverrides?.accent ?? '',
            highlight: streamScene.overlayBrandingOverrides?.highlight ?? '',
            glow: streamScene.overlayBrandingOverrides?.glow ?? ''
          }
        }))
      }
    ])
  );

  return {
    version: 1,
    label: 'Local DreamWalker Scene Workspace',
    note:
      'stream scene の title / topic / memo / branding override を localStorage で持つ workspace。',
    fragments
  };
}

const defaultSceneWorkspaceTemplate = buildSceneWorkspaceTemplate();

function normalizeAssetManifest(manifestLike) {
  const manifest =
    manifestLike && typeof manifestLike === 'object' ? manifestLike : {};

  return {
    ...defaultAssetManifestTemplate,
    ...manifest,
    fragments: Object.fromEntries(
      Object.keys(defaultAssetManifestTemplate.fragments).map((fragmentId) => [
        fragmentId,
        {
          ...defaultAssetManifestTemplate.fragments[fragmentId],
          ...(manifest.fragments?.[fragmentId] ?? {})
        }
      ])
    )
  };
}

function normalizeSceneWorkspace(workspaceLike) {
  const workspace =
    workspaceLike && typeof workspaceLike === 'object' ? workspaceLike : {};

  return {
    ...defaultSceneWorkspaceTemplate,
    ...workspace,
    fragments: Object.fromEntries(
      Object.keys(defaultSceneWorkspaceTemplate.fragments).map((fragmentId) => {
        const defaultFragment = defaultSceneWorkspaceTemplate.fragments[fragmentId];
        const workspaceFragment = workspace.fragments?.[fragmentId] ?? {};
        const workspaceScenesById = new Map(
          Array.isArray(workspaceFragment.streamScenes)
            ? workspaceFragment.streamScenes
                .filter((scene) => scene && typeof scene.id === 'string')
                .map((scene) => [scene.id, scene])
            : []
        );

        return [
          fragmentId,
          {
            ...defaultFragment,
            ...workspaceFragment,
            streamScenes: defaultFragment.streamScenes.map((defaultScene) => {
              const workspaceScene = workspaceScenesById.get(defaultScene.id) ?? {};

              return {
                ...defaultScene,
                ...workspaceScene,
                overlayMemo: {
                  ...defaultScene.overlayMemo,
                  ...(workspaceScene.overlayMemo ?? {}),
                  items: Array.isArray(workspaceScene.overlayMemo?.items)
                    ? normalizeStringList(workspaceScene.overlayMemo.items)
                    : defaultScene.overlayMemo.items
                },
                overlayBrandingOverrides: {
                  ...defaultScene.overlayBrandingOverrides,
                  ...(workspaceScene.overlayBrandingOverrides ?? {})
                }
              };
            })
          }
        ];
      })
    )
  };
}

function loadAssetWorkspaceManifest() {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(assetWorkspaceStorageKey);
    return raw ? normalizeAssetManifest(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

function loadSceneWorkspace() {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(sceneWorkspaceStorageKey);
    return raw ? normalizeSceneWorkspace(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

function normalizeSemanticZoneWorkspaceMap(workspaceLike) {
  const workspace = workspaceLike && typeof workspaceLike === 'object' ? workspaceLike : {};

  return Object.fromEntries(
    Object.entries(workspace)
      .filter(([fragmentId, payload]) => typeof fragmentId === 'string' && payload)
      .map(([fragmentId, payload]) => [fragmentId, serializeSemanticZoneMap(buildSemanticZoneMap(payload))])
  );
}

function loadSemanticZoneWorkspaceMap() {
  if (typeof window === 'undefined') {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(semanticZoneWorkspaceStorageKey);
    return raw ? normalizeSemanticZoneWorkspaceMap(JSON.parse(raw)) : {};
  } catch {
    return {};
  }
}

function tryParseAssetManifestJson(rawJson) {
  const parsed = JSON.parse(rawJson);
  return normalizeAssetManifest(parsed);
}

function tryParseSceneWorkspaceJson(rawJson) {
  const parsed = JSON.parse(rawJson);
  return normalizeSceneWorkspace(parsed);
}

function tryParseSemanticZoneJson(rawJson) {
  const parsed = JSON.parse(rawJson);
  return serializeSemanticZoneMap(buildSemanticZoneMap(parsed));
}

function buildDefaultStudioState() {
  const defaultConfig = resolveDreamwalkerConfig(dreamwalkerConfig.defaultFragmentId);

  return {
    fragmentId: defaultConfig.fragmentId,
    streamSceneId: defaultConfig.streamScenes[0]?.id ?? null,
    overlayPresetId:
      defaultConfig.overlayPresets[0]?.id ??
      dreamwalkerConfig.overlayPresets[0]?.id ??
      null,
    filterId:
      defaultConfig.dreamFilters[0]?.id ??
      dreamwalkerConfig.dreamFilters[0]?.id ??
      null,
    ratioId:
      defaultConfig.photoRatios[0]?.id ??
      dreamwalkerConfig.photoRatios[0]?.id ??
      null,
    cameraPresetId:
      defaultConfig.homePresetId ??
      defaultConfig.cameraPresets[0]?.id ??
      null
  };
}

const defaultStudioState = buildDefaultStudioState();

function normalizeStudioBundle(bundleLike) {
  const bundle = bundleLike && typeof bundleLike === 'object' ? bundleLike : {};
  const state = bundle.state && typeof bundle.state === 'object' ? bundle.state : {};
  let robotRoute = null;

  try {
    robotRoute = bundle.robotRoute ? normalizeRobotRoutePayload(bundle.robotRoute) : null;
  } catch {
    robotRoute = null;
  }

  return {
    version: 1,
    label: typeof bundle.label === 'string' && bundle.label.trim()
      ? bundle.label
      : 'Local DreamWalker Studio Bundle',
    note:
      typeof bundle.note === 'string' && bundle.note.trim()
        ? bundle.note
        : 'asset workspace / scene workspace / semantic zone workspace / robot route と現在の stage state を束ねた bundle。',
    assetWorkspace: normalizeAssetManifest(bundle.assetWorkspace),
    sceneWorkspace: normalizeSceneWorkspace(bundle.sceneWorkspace),
    semanticZoneWorkspace: normalizeSemanticZoneWorkspaceMap(bundle.semanticZoneWorkspace),
    robotRoute,
    state: {
      ...defaultStudioState,
      ...state
    }
  };
}

function tryParseStudioBundleJson(rawJson) {
  const parsed = JSON.parse(rawJson);
  return normalizeStudioBundle(parsed);
}

function normalizeStudioBundleShelfEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {};

  return {
    id:
      typeof entry.id === 'string' && entry.id.trim()
        ? entry.id
        : `studio-bundle-${index + 1}`,
    label:
      typeof entry.label === 'string' && entry.label.trim()
        ? entry.label
        : `Studio Bundle ${index + 1}`,
    bundle: normalizeStudioBundle(entry.bundle)
  };
}

function loadStudioBundleShelf() {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(studioBundleShelfStorageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.map((entry, index) => normalizeStudioBundleShelfEntry(entry, index))
      : [];
  } catch {
    return [];
  }
}

function normalizeRobotRouteShelfEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {};

  return {
    id:
      typeof entry.id === 'string' && entry.id.trim()
        ? entry.id
        : `robot-route-${index + 1}`,
    label:
      typeof entry.label === 'string' && entry.label.trim()
        ? entry.label
        : `Robot Route ${index + 1}`,
    route: normalizeRobotRoutePayload(entry.route ?? entry.payload ?? entry)
  };
}

function loadRobotRouteShelf() {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(robotRouteShelfStorageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.map((entry, index) => normalizeRobotRouteShelfEntry(entry, index))
      : [];
  } catch {
    return [];
  }
}

function normalizeRobotMissionDraftBundleShelfEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {};

  return {
    id:
      typeof entry.id === 'string' && entry.id.trim()
        ? entry.id
        : `robot-mission-draft-bundle-${index + 1}`,
    label:
      typeof entry.label === 'string' && entry.label.trim()
        ? entry.label
        : `Robot Mission Draft Bundle ${index + 1}`,
    bundle: normalizeRobotMissionDraftBundle(
      entry.bundle ?? entry.payload ?? entry
    )
  };
}

function loadRobotMissionDraftBundleShelf() {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(robotMissionDraftBundleShelfStorageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed)
      ? parsed.map((entry, index) =>
          normalizeRobotMissionDraftBundleShelfEntry(entry, index)
        )
      : [];
  } catch {
    return [];
  }
}

function normalizeRobotRouteCatalogEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {};

  return {
    id:
      typeof entry.id === 'string' && entry.id.trim()
        ? entry.id
        : `public-route-${index + 1}`,
    label:
      typeof entry.label === 'string' && entry.label.trim()
        ? entry.label
        : `Public Route ${index + 1}`,
    url: typeof entry.url === 'string' ? entry.url.trim() : '',
    description: typeof entry.description === 'string' ? entry.description.trim() : '',
    fragmentId: typeof entry.fragmentId === 'string' ? entry.fragmentId.trim() : '',
    accent: typeof entry.accent === 'string' ? entry.accent.trim() : ''
  };
}

function normalizeRobotRouteCatalog(catalogLike) {
  const catalog = catalogLike && typeof catalogLike === 'object' ? catalogLike : {};

  return {
    version: 1,
    label:
      typeof catalog.label === 'string' && catalog.label.trim()
        ? catalog.label
        : 'DreamWalker Robot Route Catalog',
    note:
      typeof catalog.note === 'string' && catalog.note.trim()
        ? catalog.note
        : '公開用 robot route preset 一覧。',
    routes: Array.isArray(catalog.routes)
      ? catalog.routes.map((entry, index) => normalizeRobotRouteCatalogEntry(entry, index))
      : []
  };
}

function normalizeRobotMissionCatalogEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {};

  return {
    id:
      typeof entry.id === 'string' && entry.id.trim()
        ? entry.id
        : `public-mission-${index + 1}`,
    label:
      typeof entry.label === 'string' && entry.label.trim()
        ? entry.label
        : `Public Mission ${index + 1}`,
    url: typeof entry.url === 'string' ? entry.url.trim() : '',
    description: typeof entry.description === 'string' ? entry.description.trim() : '',
    fragmentId: typeof entry.fragmentId === 'string' ? entry.fragmentId.trim() : '',
    accent: typeof entry.accent === 'string' ? entry.accent.trim() : ''
  };
}

function normalizeRobotMissionCatalog(catalogLike) {
  const catalog = catalogLike && typeof catalogLike === 'object' ? catalogLike : {};

  return {
    version: 1,
    label:
      typeof catalog.label === 'string' && catalog.label.trim()
        ? catalog.label
        : 'DreamWalker Robot Mission Catalog',
    note:
      typeof catalog.note === 'string' && catalog.note.trim()
        ? catalog.note
        : '公開用 robot mission manifest 一覧。',
    missions: Array.isArray(catalog.missions)
      ? catalog.missions.map((entry, index) => normalizeRobotMissionCatalogEntry(entry, index))
      : []
  };
}

function normalizeStudioBundleCatalogEntry(entryLike, index) {
  const entry = entryLike && typeof entryLike === 'object' ? entryLike : {};

  return {
    id:
      typeof entry.id === 'string' && entry.id.trim()
        ? entry.id
        : `public-bundle-${index + 1}`,
    label:
      typeof entry.label === 'string' && entry.label.trim()
        ? entry.label
        : `Public Bundle ${index + 1}`,
    url: typeof entry.url === 'string' ? entry.url.trim() : '',
    description: typeof entry.description === 'string' ? entry.description.trim() : '',
    fragmentId: typeof entry.fragmentId === 'string' ? entry.fragmentId.trim() : '',
    accent: typeof entry.accent === 'string' ? entry.accent.trim() : ''
  };
}

function normalizeStudioBundleCatalog(catalogLike) {
  const catalog = catalogLike && typeof catalogLike === 'object' ? catalogLike : {};

  return {
    version: 1,
    label:
      typeof catalog.label === 'string' && catalog.label.trim()
        ? catalog.label
        : 'DreamWalker Studio Bundle Catalog',
    note:
      typeof catalog.note === 'string' && catalog.note.trim()
        ? catalog.note
        : '公開用 studio bundle file 一覧。',
    bundles: Array.isArray(catalog.bundles)
      ? catalog.bundles.map((entry, index) => normalizeStudioBundleCatalogEntry(entry, index))
      : []
  };
}

async function probeLocalAssetAvailability(assetUrl) {
  const localPath = normalizeLocalAssetPath(assetUrl);

  if (!localPath || typeof fetch === 'undefined') {
    return undefined;
  }

  try {
    const response = await fetch(localPath, {
      method: 'HEAD',
      cache: 'no-store'
    });

    if (!response.ok) {
      return false;
    }

    const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
    const isHtmlFallback =
      contentType.includes('text/html') &&
      !localPath.endsWith('.html') &&
      !localPath.endsWith('.htm');

    return !isHtmlFallback;
  } catch {
    return false;
  }
}

function buildHealthClassName(status) {
  return [
    'health-badge',
    status === 'ready'
      ? 'health-badge-ready'
      : status === 'warning'
        ? 'health-badge-warning'
        : status === 'error'
          ? 'health-badge-error'
          : 'health-badge-neutral'
  ].join(' ');
}

function HealthBadge({ health }) {
  if (!health?.label) {
    return null;
  }

  return (
    <span className={buildHealthClassName(health.status)}>
      {health.label}
    </span>
  );
}

function buildCaptureFileName() {
  const now = new Date();
  const safe = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    '-',
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0')
  ].join('');

  return `dreamwalker-live-${safe}.png`;
}

function buildSceneExportFileName(fragmentId, streamSceneId) {
  const safeFragmentId = fragmentId || 'fragment';
  const safeStreamSceneId = streamSceneId || 'scene';
  return `dreamwalker-live-${safeFragmentId}-${safeStreamSceneId}.json`;
}

function downloadTextFile(fileName, content, type = 'application/json') {
  const blob = new Blob([content], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadCanvasSnapshot() {
  const canvas = document.querySelector('.dreamwalker-stage canvas');
  if (!(canvas instanceof HTMLCanvasElement)) {
    return false;
  }

  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = buildCaptureFileName();
  link.click();
  return true;
}

function getGuideStyle(selectedRatio) {
  if (selectedRatio.id === 'free') {
    return {
      width: '84%',
      height: '84%'
    };
  }

  const ratio = selectedRatio.frameWidth / selectedRatio.frameHeight;
  const viewportRatio = window.innerWidth / window.innerHeight;

  if (ratio > viewportRatio) {
    return {
      width: '84%',
      height: `calc(84vw / ${ratio})`
    };
  }

  return {
    width: `calc(84vh * ${ratio})`,
    height: '84%'
  };
}

function HotspotOverlay({ hotspots, onActivate }) {
  if (hotspots.length === 0) {
    return null;
  }

  return (
    <div className="hotspot-overlay" aria-hidden="false">
      {hotspots.map((hotspot) => (
        <button
          key={hotspot.id}
          className={`hotspot-pin hotspot-${hotspot.kind}`}
          onClick={() => onActivate(hotspot)}
          style={{
            left: `${hotspot.xPercent}%`,
            top: `${hotspot.yPercent}%`,
            borderColor: hotspot.accentColor
          }}
          type="button">
          <span className="hotspot-core" style={{ backgroundColor: hotspot.accentColor }} />
          <span className="hotspot-label">{hotspot.label}</span>
        </button>
      ))}
    </div>
  );
}

function normalizeYawDegrees(value) {
  const normalized = value % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

function getForwardVector(yawDegrees) {
  const radians = (yawDegrees * Math.PI) / 180;
  return {
    x: -Math.sin(radians),
    z: -Math.cos(radians)
  };
}

function yawDegreesFromRouteSegment(currentPosition, nextPosition, fallbackYawDegrees = 0) {
  if (!Array.isArray(currentPosition) || !Array.isArray(nextPosition)) {
    return normalizeYawDegrees(fallbackYawDegrees);
  }

  const dx = Number(nextPosition[0]) - Number(currentPosition[0]);
  const dz = Number(nextPosition[2]) - Number(currentPosition[2]);
  if (!Number.isFinite(dx) || !Number.isFinite(dz) || Math.hypot(dx, dz) < 0.0001) {
    return normalizeYawDegrees(fallbackYawDegrees);
  }

  return normalizeYawDegrees((Math.atan2(-dx, -dz) * 180) / Math.PI);
}

function buildRobotPoseFromConfig(config) {
  const robotics = config.robotics ?? {};
  const spawnPosition = robotics.spawnPosition ?? [0, 0, 0];

  return {
    position: [...spawnPosition],
    yawDegrees: normalizeYawDegrees(robotics.spawnYawDegrees ?? 0)
  };
}

function buildFallbackSemanticZonePayload(config) {
  const proxyCollider = config.walkProxyColliders?.[0] ?? null;

  if (proxyCollider) {
    return {
      frameId: 'dreamwalker_map',
      resolution: 0.5,
      defaultCost: 0,
      bounds: {
        minX: Number((proxyCollider.position[0] - proxyCollider.halfExtents[0]).toFixed(2)),
        maxX: Number((proxyCollider.position[0] + proxyCollider.halfExtents[0]).toFixed(2)),
        minZ: Number((proxyCollider.position[2] - proxyCollider.halfExtents[2]).toFixed(2)),
        maxZ: Number((proxyCollider.position[2] + proxyCollider.halfExtents[2]).toFixed(2))
      },
      zones: []
    };
  }

  const spawnPosition = config.robotics?.spawnPosition ?? [0, 0, 0];
  return {
    frameId: 'dreamwalker_map',
    resolution: 0.5,
    defaultCost: 0,
    bounds: {
      minX: spawnPosition[0] - 6,
      maxX: spawnPosition[0] + 6,
      minZ: spawnPosition[2] - 6,
      maxZ: spawnPosition[2] + 6
    },
    zones: []
  };
}

function buildDefaultSemanticZoneDraft(config, zoneCount) {
  const spawnPosition = config.robotics?.spawnPosition ?? [0, 0, 0];
  const zoneIndex = zoneCount + 1;

  return {
    id: `${config.fragmentId}-zone-${zoneIndex}`,
    label: `Zone ${zoneIndex}`,
    shape: 'rect',
    center: [spawnPosition[0], 0, spawnPosition[2]],
    size: [2.5, 2.5],
    radius: 1.2,
    cost: 25,
    tags: ['nav']
  };
}

function buildDuplicatedSemanticZoneDraft(zone, zoneCount) {
  const safeLabel = zone.label?.trim() || zone.id || `Zone ${zoneCount + 1}`;
  const center = Array.isArray(zone.center) ? [...zone.center] : [0, 0, 0];
  const nextCenter = [center[0] ?? 0, center[1] ?? 0, center[2] ?? 0];
  nextCenter[0] = Number((nextCenter[0] + 0.6).toFixed(2));
  nextCenter[2] = Number((nextCenter[2] + 0.6).toFixed(2));

  return {
    ...zone,
    id: `${zone.id || `zone-${zoneCount + 1}`}-copy-${zoneCount + 1}`,
    label: `${safeLabel} Copy`,
    center: nextCenter,
    tags: Array.isArray(zone.tags) ? [...zone.tags] : []
  };
}

function buildRouteSemanticZones(config, robotTrail, waypoint, existingCount) {
  const routeZoneRadius = config.robotics?.routeZoneRadius ?? 0.85;
  const routeZoneCost = config.robotics?.routeZoneCost ?? 18;
  const routePoints = [...robotTrail];

  if (waypoint?.position) {
    routePoints.push([...waypoint.position]);
  }

  return routePoints.map((position, index) => ({
    id: `${config.fragmentId}-route-zone-${existingCount + index + 1}`,
    label: `Route Zone ${existingCount + index + 1}`,
    shape: 'circle',
    center: [position[0], position[1] ?? 0, position[2]],
    size: [0, 0],
    radius: routeZoneRadius,
    cost: routeZoneCost,
    tags: ['route', 'safe']
  }));
}

function fitSemanticZoneBounds(payload, padding = 1) {
  const zones = Array.isArray(payload?.zones) ? payload.zones : [];
  if (!zones.length) {
    return payload;
  }

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  zones.forEach((zone) => {
    const center = Array.isArray(zone.center) ? zone.center : [0, 0, 0];
    const centerX = Number(center[0] ?? 0);
    const centerZ = Number(center[2] ?? 0);

    if (zone.shape === 'circle') {
      const radius = Math.max(0.1, Number(zone.radius ?? 0.1));
      minX = Math.min(minX, centerX - radius);
      maxX = Math.max(maxX, centerX + radius);
      minZ = Math.min(minZ, centerZ - radius);
      maxZ = Math.max(maxZ, centerZ + radius);
      return;
    }

    const size = Array.isArray(zone.size) ? zone.size : [1, 1];
    const halfX = Math.max(0.05, Number(size[0] ?? 1) / 2);
    const halfZ = Math.max(0.05, Number(size[1] ?? 1) / 2);
    minX = Math.min(minX, centerX - halfX);
    maxX = Math.max(maxX, centerX + halfX);
    minZ = Math.min(minZ, centerZ - halfZ);
    maxZ = Math.max(maxZ, centerZ + halfZ);
  });

  const safePadding = Math.max(0, Number(padding) || 0);

  return {
    ...payload,
    bounds: {
      ...(payload.bounds ?? {}),
      minX: Number((minX - safePadding).toFixed(2)),
      maxX: Number((maxX + safePadding).toFixed(2)),
      minZ: Number((minZ - safePadding).toFixed(2)),
      maxZ: Number((maxZ + safePadding).toFixed(2))
    }
  };
}

function stepRobotPose(currentPose, action, roboticsConfig) {
  const moveStep = roboticsConfig.moveStep ?? 0.8;
  const turnStepDegrees = roboticsConfig.turnStepDegrees ?? 16;
  const nextPose = {
    position: [...currentPose.position],
    yawDegrees: currentPose.yawDegrees
  };

  if (action === 'turn-left') {
    nextPose.yawDegrees = normalizeYawDegrees(currentPose.yawDegrees + turnStepDegrees);
    return nextPose;
  }

  if (action === 'turn-right') {
    nextPose.yawDegrees = normalizeYawDegrees(currentPose.yawDegrees - turnStepDegrees);
    return nextPose;
  }

  const forwardVector = getForwardVector(currentPose.yawDegrees);
  const moveDirection = action === 'backward' ? -1 : 1;
  nextPose.position[0] += forwardVector.x * moveStep * moveDirection;
  nextPose.position[2] += forwardVector.z * moveStep * moveDirection;
  return nextPose;
}

function buildWaypointAhead(robotPose, roboticsConfig) {
  const waypointDistance = roboticsConfig.waypointDistance ?? 2.8;
  const forwardVector = getForwardVector(robotPose.yawDegrees);

  return {
    position: [
      robotPose.position[0] + forwardVector.x * waypointDistance,
      robotPose.position[1],
      robotPose.position[2] + forwardVector.z * waypointDistance
    ]
  };
}

function buildRobotProjectionPoints(robotPose, waypoint, roboticsConfig) {
  const footprintRadius = roboticsConfig.footprintRadius ?? 0.55;
  const forwardVector = getForwardVector(robotPose.yawDegrees);
  const headingDistance = Math.max(footprintRadius * 2.4, 1.35);
  const points = [
    {
      id: 'robot-base',
      kind: 'robot-base',
      label: 'Robot Base',
      accentColor: '#93ffd4',
      position: [
        robotPose.position[0],
        robotPose.position[1] + 0.48,
        robotPose.position[2]
      ]
    },
    {
      id: 'robot-heading',
      kind: 'robot-heading',
      label: `Heading ${Math.round(robotPose.yawDegrees)}°`,
      accentColor: '#85e3e1',
      position: [
        robotPose.position[0] + forwardVector.x * headingDistance,
        robotPose.position[1] + 0.72,
        robotPose.position[2] + forwardVector.z * headingDistance
      ]
    }
  ];

  if (waypoint) {
    points.push({
      id: 'robot-waypoint',
      kind: 'robot-waypoint',
      label: 'Waypoint',
      accentColor: '#f4ca72',
      position: [
        waypoint.position[0],
        waypoint.position[1] + 0.52,
        waypoint.position[2]
      ]
    });
  }

  return points;
}

function buildRobotTrailFromPose(robotPose) {
  return [[...robotPose.position]];
}

function appendRobotTrail(trail, nextPose, roboticsConfig) {
  const nextPosition = [...nextPose.position];
  const lastPosition = trail[trail.length - 1];

  if (lastPosition) {
    const dx = nextPosition[0] - lastPosition[0];
    const dy = nextPosition[1] - lastPosition[1];
    const dz = nextPosition[2] - lastPosition[2];
    const distance = Math.hypot(dx, dy, dz);

    if (distance < 0.05) {
      return trail;
    }
  }

  const trailPointLimit = roboticsConfig.trailPointLimit ?? 18;
  return [...trail, nextPosition].slice(-trailPointLimit);
}

function buildRobotRouteProjectionPoints(robotTrail, waypoint, roboticsConfig) {
  const trailHeight = roboticsConfig.trailHeight ?? 0.16;
  const points = robotTrail.map((position, index) => ({
    id: `robot-route-${index}`,
    kind: 'route-node',
    order: index,
    position: [position[0], position[1] + trailHeight, position[2]]
  }));

  if (waypoint) {
    points.push({
      id: 'robot-route-waypoint',
      kind: 'route-waypoint-anchor',
      order: points.length,
      position: [
        waypoint.position[0],
        waypoint.position[1] + trailHeight,
        waypoint.position[2]
      ]
    });
  }

  return points;
}

function buildBenchmarkRouteProjectionPoints(benchmarkOverlay, roboticsConfig) {
  if (!benchmarkOverlay?.samples?.length) {
    return [];
  }

  const trailHeight = roboticsConfig.trailHeight ?? 0.16;
  const groundTruthHeight = trailHeight + 0.18;
  const estimateHeight = trailHeight + 0.28;

  return benchmarkOverlay.samples.flatMap((sample) => {
    const points = [];

    if (Array.isArray(sample.groundTruthPosition) && sample.groundTruthPosition.length >= 3) {
      points.push({
        id: `benchmark-ground-truth-${sample.index}`,
        kind: 'benchmark-ground-truth-node',
        order: sample.index,
        sampleIndex: sample.index,
        isWorst: benchmarkOverlay.worstSampleIndex === sample.index,
        translationErrorMeters: sample.translationErrorMeters,
        position: [
          sample.groundTruthPosition[0],
          sample.groundTruthPosition[1] + groundTruthHeight,
          sample.groundTruthPosition[2]
        ]
      });
    }

    if (Array.isArray(sample.estimatePosition) && sample.estimatePosition.length >= 3) {
      points.push({
        id: `benchmark-estimate-${sample.index}`,
        kind: 'benchmark-estimate-node',
        order: sample.index,
        sampleIndex: sample.index,
        isWorst: benchmarkOverlay.worstSampleIndex === sample.index,
        translationErrorMeters: sample.translationErrorMeters,
        position: [
          sample.estimatePosition[0],
          sample.estimatePosition[1] + estimateHeight,
          sample.estimatePosition[2]
        ]
      });
    }

    return points;
  });
}

function calculateRobotTrailDistance(robotTrail) {
  if (robotTrail.length < 2) {
    return 0;
  }

  let totalDistance = 0;
  for (let index = 1; index < robotTrail.length; index += 1) {
    const current = robotTrail[index];
    const previous = robotTrail[index - 1];
    totalDistance += Math.hypot(
      current[0] - previous[0],
      current[1] - previous[1],
      current[2] - previous[2]
    );
  }

  return totalDistance;
}

function formatRobotNodeLabel(count) {
  return count === 1 ? '1 node' : `${count} nodes`;
}

function normalizeBridgePosition(positionLike) {
  if (!Array.isArray(positionLike) || positionLike.length < 3) {
    return null;
  }

  const normalizedPosition = positionLike
    .slice(0, 3)
    .map((value) => (typeof value === 'number' ? value : Number(value)));
  return normalizedPosition.every((value) => Number.isFinite(value))
    ? normalizedPosition
    : null;
}

function normalizeBridgePoseMessage(poseLike) {
  if (!poseLike || typeof poseLike !== 'object') {
    return null;
  }

  const normalizedPosition = normalizeBridgePosition(poseLike.position);
  const normalizedYaw = Number(poseLike.yawDegrees);

  if (!normalizedPosition || !Number.isFinite(normalizedYaw)) {
    return null;
  }

  return {
    position: normalizedPosition,
    yawDegrees: normalizeYawDegrees(normalizedYaw)
  };
}

function positionsApproximatelyEqual(left, right, epsilon = 0.0001) {
  return (
    Array.isArray(left) &&
    Array.isArray(right) &&
    left.length >= 3 &&
    right.length >= 3 &&
    Math.abs(left[0] - right[0]) <= epsilon &&
    Math.abs(left[1] - right[1]) <= epsilon &&
    Math.abs(left[2] - right[2]) <= epsilon
  );
}

function normalizeRobotRouteWaypoint(waypointLike) {
  if (!waypointLike) {
    return null;
  }

  const normalizedPosition = normalizeBridgePosition(
    waypointLike.position ?? waypointLike
  );

  return normalizedPosition ? { position: normalizedPosition } : null;
}

function readNonEmptyString(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeRobotMissionStartupMode(value) {
  const normalized = readNonEmptyString(value);

  return ['explore', 'live', 'photo', 'robot'].includes(normalized)
    ? normalized
    : '';
}

function normalizeRobotRouteWorldContext(worldLike, fallbackLike = {}) {
  const world = worldLike && typeof worldLike === 'object' ? worldLike : {};
  const fallback =
    fallbackLike && typeof fallbackLike === 'object' ? fallbackLike : {};

  return {
    fragmentId: readNonEmptyString(world.fragmentId || fallback.fragmentId),
    fragmentLabel: readNonEmptyString(world.fragmentLabel || fallback.fragmentLabel),
    assetLabel: readNonEmptyString(world.assetLabel || fallback.assetLabel),
    manifestLabel: readNonEmptyString(world.manifestLabel || fallback.manifestLabel),
    splatUrl: readNonEmptyString(world.splatUrl || fallback.splatUrl),
    colliderMeshUrl: readNonEmptyString(
      world.colliderMeshUrl || fallback.colliderMeshUrl
    ),
    frameId:
      readNonEmptyString(world.frameId || fallback.frameId) || 'dreamwalker_map',
    zoneMapUrl: readNonEmptyString(world.zoneMapUrl || fallback.zoneMapUrl),
    usesDemoFallback: Boolean(
      Object.prototype.hasOwnProperty.call(world, 'usesDemoFallback')
        ? world.usesDemoFallback
        : fallback.usesDemoFallback
    )
  };
}

function buildRobotRouteCompatibility(routePayloadLike, currentWorldLike) {
  if (!routePayloadLike) {
    return {
      status: 'error',
      label: 'Route Missing',
      detail: 'route payload がありません。'
    };
  }

  let route;
  try {
    route = normalizeRobotRoutePayload(routePayloadLike);
  } catch (error) {
    return {
      status: 'error',
      label: 'Route Invalid',
      detail: error instanceof Error ? error.message : String(error)
    };
  }

  const currentWorld = normalizeRobotRouteWorldContext(currentWorldLike);
  const routeWorld = normalizeRobotRouteWorldContext(route.world, {
    fragmentId: route.fragmentId,
    fragmentLabel: route.fragmentLabel,
    frameId: route.frameId
  });
  const routeFragmentLabel =
    routeWorld.fragmentLabel || route.fragmentLabel || routeWorld.fragmentId || route.fragmentId;
  const currentFragmentLabel =
    currentWorld.fragmentLabel || currentWorld.fragmentId || 'current fragment';
  const routeAssetLabel =
    routeWorld.assetLabel || routeWorld.fragmentLabel || routeWorld.fragmentId || 'unknown world';
  const currentAssetLabel =
    currentWorld.assetLabel || currentWorld.fragmentLabel || currentWorld.fragmentId || 'current world';

  if (
    routeWorld.fragmentId &&
    currentWorld.fragmentId &&
    routeWorld.fragmentId !== currentWorld.fragmentId
  ) {
    return {
      status: 'warning',
      label: 'Fragment Drift',
      detail: `route は ${routeFragmentLabel} 用、current world は ${currentFragmentLabel} です。`
    };
  }

  if (
    routeWorld.frameId &&
    currentWorld.frameId &&
    routeWorld.frameId !== currentWorld.frameId
  ) {
    return {
      status: 'warning',
      label: 'Frame Drift',
      detail: `route frame は ${routeWorld.frameId}、current frame は ${currentWorld.frameId} です。`
    };
  }

  const hasRouteWorldMetadata = Boolean(
    routeWorld.assetLabel ||
      routeWorld.manifestLabel ||
      routeWorld.splatUrl ||
      routeWorld.colliderMeshUrl
  );
  const splatDrift =
    routeWorld.splatUrl &&
    currentWorld.splatUrl &&
    routeWorld.splatUrl !== currentWorld.splatUrl;
  const colliderDrift =
    routeWorld.colliderMeshUrl &&
    currentWorld.colliderMeshUrl &&
    routeWorld.colliderMeshUrl !== currentWorld.colliderMeshUrl;
  const assetLabelDrift =
    routeWorld.assetLabel &&
    currentWorld.assetLabel &&
    routeWorld.assetLabel !== currentWorld.assetLabel;
  const manifestLabelDrift =
    routeWorld.manifestLabel &&
    currentWorld.manifestLabel &&
    routeWorld.manifestLabel !== currentWorld.manifestLabel;

  if (splatDrift || colliderDrift || assetLabelDrift || manifestLabelDrift) {
    return {
      status: 'warning',
      label: 'World Drift',
      detail: `route world は ${routeAssetLabel}、current world は ${currentAssetLabel} です。`
    };
  }

  if (!hasRouteWorldMetadata) {
    return {
      status: 'neutral',
      label: 'Legacy Route',
      detail: 'world metadata が無いため fragment / frame のみ照合しています。'
    };
  }

  return {
    status: 'ready',
    label: 'World Match',
    detail: `${currentAssetLabel} / frame ${currentWorld.frameId || routeWorld.frameId}`
  };
}

function normalizeRobotRoutePayload(payloadLike) {
  const payload = payloadLike && typeof payloadLike === 'object' ? payloadLike : {};
  const pose = normalizeBridgePoseMessage(payload.pose);
  const route = Array.isArray(payload.route)
    ? payload.route.map((position) => normalizeBridgePosition(position)).filter(Boolean)
    : [];
  const waypoint = normalizeRobotRouteWaypoint(payload.waypoint);

  if (!pose && route.length === 0) {
    throw new Error('robot route must contain pose or route');
  }

  const normalizedPose = pose ?? {
    position: [...route[route.length - 1]],
    yawDegrees: 0
  };
  const normalizedRoute = route.length > 0 ? route.map((position) => [...position]) : [[...normalizedPose.position]];

  if (
    !positionsApproximatelyEqual(
      normalizedRoute[normalizedRoute.length - 1],
      normalizedPose.position
    )
  ) {
    normalizedRoute.push([...normalizedPose.position]);
  }

  const normalizedFragmentId = readNonEmptyString(payload.fragmentId);
  const normalizedFragmentLabel = readNonEmptyString(payload.fragmentLabel);
  const normalizedFrameId =
    readNonEmptyString(payload.frameId) || 'dreamwalker_map';
  const normalizedWorld = normalizeRobotRouteWorldContext(payload.world, {
    fragmentId: normalizedFragmentId,
    fragmentLabel: normalizedFragmentLabel,
    frameId: normalizedFrameId
  });

  return {
    version: Number(payload.version ?? 1) || 1,
    protocol:
      typeof payload.protocol === 'string' && payload.protocol.trim()
        ? payload.protocol.trim()
        : robotRouteProtocolId,
    label: typeof payload.label === 'string' ? payload.label.trim() : '',
    description:
      typeof payload.description === 'string' ? payload.description.trim() : '',
    accent: typeof payload.accent === 'string' ? payload.accent.trim() : '',
    fragmentId: normalizedWorld.fragmentId,
    fragmentLabel: normalizedWorld.fragmentLabel,
    frameId: normalizedWorld.frameId,
    world: normalizedWorld,
    pose: {
      position: [...normalizedPose.position],
      yawDegrees: normalizedPose.yawDegrees
    },
    waypoint,
    route: normalizedRoute
  };
}

function tryParseRobotRouteJson(rawJson) {
  const parsed = JSON.parse(rawJson);
  return normalizeRobotRoutePayload(parsed);
}

function normalizeRobotMissionPayload(payloadLike) {
  const payload = payloadLike && typeof payloadLike === 'object' ? payloadLike : {};
  const world = payload.world && typeof payload.world === 'object' ? payload.world : {};

  return {
    version: Number(payload.version ?? 1) || 1,
    protocol:
      typeof payload.protocol === 'string' && payload.protocol.trim()
        ? payload.protocol.trim()
        : robotMissionProtocolId,
    id: readNonEmptyString(payload.id),
    label: readNonEmptyString(payload.label),
    description: readNonEmptyString(payload.description),
    fragmentId: readNonEmptyString(payload.fragmentId),
    fragmentLabel: readNonEmptyString(payload.fragmentLabel),
    accent: readNonEmptyString(payload.accent),
    routeUrl: readNonEmptyString(payload.routeUrl),
    zoneMapUrl: readNonEmptyString(payload.zoneMapUrl),
    launchUrl: readNonEmptyString(payload.launchUrl),
    cameraPresetId: readNonEmptyString(payload.cameraPresetId),
    robotCameraId: readNonEmptyString(payload.robotCameraId),
    streamSceneId: readNonEmptyString(payload.streamSceneId),
    startupMode: normalizeRobotMissionStartupMode(payload.startupMode),
    world: {
      assetLabel: readNonEmptyString(world.assetLabel),
      frameId: readNonEmptyString(world.frameId) || readNonEmptyString(payload.frameId) || 'dreamwalker_map'
    }
  };
}

function buildRobotMissionCompatibility(missionPayloadLike, routePayloadLike, currentWorldLike) {
  if (!missionPayloadLike) {
    return {
      status: 'error',
      label: 'Mission Missing',
      detail: 'mission payload がありません。'
    };
  }

  let mission;
  try {
    mission = normalizeRobotMissionPayload(missionPayloadLike);
  } catch (error) {
    return {
      status: 'error',
      label: 'Mission Invalid',
      detail: error instanceof Error ? error.message : String(error)
    };
  }

  if (!mission.routeUrl) {
    return {
      status: 'error',
      label: 'Mission Invalid',
      detail: 'routeUrl が未設定です。'
    };
  }

  const routeHealth = buildRobotRouteCompatibility(routePayloadLike, currentWorldLike);

  if (routeHealth.status === 'error') {
    return {
      status: 'error',
      label: 'Mission Error',
      detail: routeHealth.detail
    };
  }

  const details = [];
  let status = routeHealth.status;
  let label =
    routeHealth.status === 'ready'
      ? 'Mission Ready'
      : routeHealth.status === 'warning'
        ? 'Mission Warning'
        : 'Mission Loaded';

  let route = null;
  try {
    route = normalizeRobotRoutePayload(routePayloadLike);
  } catch {
    route = null;
  }

  if (route) {
    if (mission.fragmentId && route.fragmentId && mission.fragmentId !== route.fragmentId) {
      status = 'warning';
      label = 'Mission Warning';
      details.push(`mission fragment=${mission.fragmentId} / route fragment=${route.fragmentId}`);
    }

    if (mission.zoneMapUrl && route.world.zoneMapUrl && mission.zoneMapUrl !== route.world.zoneMapUrl) {
      status = 'warning';
      label = 'Mission Warning';
      details.push(`mission zone=${mission.zoneMapUrl} / route zone=${route.world.zoneMapUrl}`);
    }

    if (mission.world.frameId && route.frameId && mission.world.frameId !== route.frameId) {
      status = 'warning';
      label = 'Mission Warning';
      details.push(`mission frame=${mission.world.frameId} / route frame=${route.frameId}`);
    }

    if (
      mission.world.assetLabel &&
      route.world.assetLabel &&
      mission.world.assetLabel !== route.world.assetLabel
    ) {
      status = 'warning';
      label = 'Mission Warning';
      details.push(
        `mission world=${mission.world.assetLabel} / route world=${route.world.assetLabel}`
      );
    }
  }

  if (routeHealth.detail) {
    details.unshift(routeHealth.detail);
  }

  return {
    status,
    label,
    detail: details.filter(Boolean).join(' / ')
  };
}

async function fetchJsonResource(resourceUrl, options = {}) {
  const { signal } = options;
  const response = await fetch(resourceUrl, {
    cache: 'no-store',
    headers: {
      Accept: 'application/json'
    },
    signal
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json();
}

async function loadRobotMissionResource(missionUrl, options = {}) {
  const mission = normalizeRobotMissionPayload(await fetchJsonResource(missionUrl, options));

  if (!mission.routeUrl) {
    throw new Error('mission routeUrl が未設定です');
  }

  const route = normalizeRobotRoutePayload(await fetchJsonResource(mission.routeUrl, options));
  return {
    mission,
    route
  };
}

function normalizeRobotMissionDraftBundle(bundleLike) {
  const bundle = bundleLike && typeof bundleLike === 'object' ? bundleLike : {};
  const missionSource =
    bundle.mission && typeof bundle.mission === 'object' ? bundle.mission : bundle;
  const routeSource =
    bundle.route && typeof bundle.route === 'object'
      ? bundle.route
      : bundle.robotRoute && typeof bundle.robotRoute === 'object'
        ? bundle.robotRoute
        : null;

  if (!routeSource) {
    throw new Error('robot mission draft bundle must contain route');
  }

  const mission = normalizeRobotMissionPayload(missionSource);
  const route = normalizeRobotRoutePayload(routeSource);
  const fragmentId =
    readNonEmptyString(bundle.fragmentId) ||
    mission.fragmentId ||
    route.fragmentId;

  if (!fragmentId) {
    throw new Error('robot mission draft bundle must contain fragmentId');
  }

  const fragmentLabel =
    readNonEmptyString(bundle.fragmentLabel) ||
    mission.fragmentLabel ||
    route.fragmentLabel;

  let zones = null;
  if (bundle.zones) {
    zones = serializeSemanticZoneMap(buildSemanticZoneMap(bundle.zones));
  }

  const normalizedMission = {
    ...mission,
    fragmentId: mission.fragmentId || fragmentId,
    fragmentLabel: mission.fragmentLabel || fragmentLabel
  };
  const normalizedRoute = normalizeRobotRoutePayload({
    ...route,
    fragmentId: route.fragmentId || fragmentId,
    fragmentLabel: route.fragmentLabel || fragmentLabel,
    world: {
      ...route.world,
      fragmentId: route.world?.fragmentId || fragmentId,
      fragmentLabel: route.world?.fragmentLabel || fragmentLabel,
      frameId:
        route.world?.frameId ||
        route.frameId ||
        normalizedMission.world.frameId ||
        'dreamwalker_map',
      zoneMapUrl:
        route.world?.zoneMapUrl ||
        normalizedMission.zoneMapUrl ||
        ''
    }
  });

  return {
    version: Number(bundle.version ?? 1) || 1,
    label:
      readNonEmptyString(bundle.label) ||
      normalizedMission.label ||
      `${fragmentLabel || fragmentId} Draft Bundle`,
    fragmentId,
    fragmentLabel,
    mission: normalizedMission,
    route: normalizedRoute,
    zones
  };
}

function tryParseRobotMissionDraftBundleImport(rawJson) {
  const parsed = JSON.parse(rawJson);

  if (
    parsed &&
    typeof parsed === 'object' &&
    readNonEmptyString(parsed.protocol) === robotMissionArtifactPackProtocolId
  ) {
    const files = Array.isArray(parsed.files) ? parsed.files : [];
    const readArtifactPackFileContent = (kind) => {
      const entry = files.find(
        (candidate) =>
          candidate &&
          typeof candidate === 'object' &&
          readNonEmptyString(candidate.kind) === kind
      );

      if (!entry) {
        return {
          fileName: '',
          content: ''
        };
      }

      const content =
        typeof entry.content === 'string'
          ? entry.content
          : JSON.stringify(entry.content ?? {}, null, 2);

      return {
        fileName: readNonEmptyString(entry.fileName),
        content
      };
    };
    const draftBundleEntry = files.find(
      (entry) =>
        entry &&
        typeof entry === 'object' &&
        readNonEmptyString(entry.kind) === 'draft-bundle'
    );

    if (!draftBundleEntry) {
      throw new Error('robot mission artifact pack must contain draft-bundle content');
    }

    const draftBundleSource =
      typeof draftBundleEntry.content === 'string'
        ? JSON.parse(draftBundleEntry.content)
        : draftBundleEntry.content;

    return {
      bundle: normalizeRobotMissionDraftBundle(draftBundleSource),
      importLabel:
        readNonEmptyString(parsed.label) ||
        readNonEmptyString(draftBundleEntry.fileName) ||
        'Imported Artifact Pack',
      artifactPack: {
        label:
          readNonEmptyString(parsed.label) ||
          readNonEmptyString(draftBundleEntry.fileName) ||
          'Imported Artifact Pack',
        fileCount: files.length,
        preflightSummary: readArtifactPackFileContent('preflight-summary'),
        publishReport: readArtifactPackFileContent('publish-report')
      }
    };
  }

  const bundle = normalizeRobotMissionDraftBundle(parsed);
  return {
    bundle,
    importLabel:
      readNonEmptyString(bundle.label) ||
      readNonEmptyString(bundle.mission.label) ||
      `${bundle.fragmentLabel || bundle.fragmentId} Draft Bundle`
  };
}

function tryParseRobotMissionDraftBundleJson(rawJson) {
  return tryParseRobotMissionDraftBundleImport(rawJson).bundle;
}

function buildMissionSlug(value, fallback = 'robot-mission') {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return normalized || fallback;
}

function extractRobotRouteIdFromUrl(routeUrlLike) {
  const normalizedRouteUrl = readNonEmptyString(routeUrlLike);

  if (!normalizedRouteUrl) {
    return '';
  }

  const routeMatch = normalizedRouteUrl.match(/\/robot-routes\/([^/?#]+)\.json(?:[?#].*)?$/i);
  if (!routeMatch?.[1]) {
    return '';
  }

  return buildMissionSlug(routeMatch[1], '');
}

function buildRobotRouteUrlFromId(routeIdLike, fallbackRouteIdLike = '') {
  const routeId = buildMissionSlug(routeIdLike, '');
  const fallbackRouteId = buildMissionSlug(fallbackRouteIdLike, '');
  const normalizedRouteId = routeId || fallbackRouteId;

  return normalizedRouteId ? `/robot-routes/${normalizedRouteId}.json` : '';
}

function quoteShellValue(value) {
  const normalized = String(value ?? '');
  return /\s/.test(normalized) ? `"${normalized.replace(/"/g, '\\"')}"` : normalized;
}

function buildRobotMissionDraftBundleFileName(bundleLike, fallbackFragmentId = 'residency') {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const fragmentId = bundle.fragmentId || fallbackFragmentId;
  const missionSlug = buildMissionSlug(
    bundle.mission.id ||
      bundle.mission.label ||
      bundle.label,
    'robot-mission-draft-bundle'
  );

  return `dreamwalker-live-${fragmentId}-${missionSlug}-draft-bundle.json`;
}

function buildRobotMissionArtifactPackFileName(bundleLike, fallbackFragmentId = 'residency') {
  const draftBundleFileName = buildRobotMissionDraftBundleFileName(
    bundleLike,
    fallbackFragmentId
  );

  return draftBundleFileName.endsWith('.json')
    ? draftBundleFileName.replace(/\.json$/i, '.artifact-pack.json')
    : `${draftBundleFileName}.artifact-pack.json`;
}

function buildRobotMissionPublishCommandFromBundle(
  bundleLike,
  fileNameLike,
  fallbackConfig = null
) {
  const fileName =
    readNonEmptyString(fileNameLike) ||
    buildRobotMissionArtifactPackFileName(bundleLike);
  const commandLines = [
    'npm run publish:robot-mission --',
    `--bundle ${quoteShellValue(`/absolute/path/to/${fileName}`)}`,
    '--force',
    '--validate'
  ];

  if (!fallbackConfig) {
    return commandLines.join(' \\\n  ');
  }

  const preview = buildPublishedRobotMissionPreviewFromBundle(bundleLike, fallbackConfig);
  const health = buildRobotMissionPreflightHealth(bundleLike, fallbackConfig);

  return [
    `# publish input: /absolute/path/to/${fileName}`,
    `# preflight: ${health.label}`,
    `# detail: ${health.detail || 'none'}`,
    `# target: ${preview.payload.fragmentId || 'none'} / ${preview.payload.world.assetLabel || 'none'} / ${preview.payload.world.frameId || 'none'}`,
    `# zone: ${preview.payload.zoneMapUrl || 'none'}`,
    `# launch: ${preview.payload.launchUrl || 'none'}`,
    commandLines.join(' \\\n  ')
  ].join('\n');
}

function buildRobotMissionValidateCommandFromBundle(
  bundleLike,
  fileNameLike,
  fallbackConfig = null
) {
  const fileName =
    readNonEmptyString(fileNameLike) ||
    buildRobotMissionArtifactPackFileName(bundleLike);
  const commandLines = [
    'npm run validate:robot-bundle --',
    `--bundle ${quoteShellValue(`/absolute/path/to/${fileName}`)}`
  ];

  if (!fallbackConfig) {
    return commandLines.join(' \\\n  ');
  }

  const preview = buildPublishedRobotMissionPreviewFromBundle(bundleLike, fallbackConfig);
  const health = buildRobotMissionPreflightHealth(bundleLike, fallbackConfig);

  return [
    `# validate input: /absolute/path/to/${fileName}`,
    `# preflight: ${health.label}`,
    `# detail: ${health.detail || 'none'}`,
    `# target: ${preview.payload.fragmentId || 'none'} / ${preview.payload.world.assetLabel || 'none'} / ${preview.payload.world.frameId || 'none'}`,
    `# zone: ${preview.payload.zoneMapUrl || 'none'}`,
    `# launch: ${preview.payload.launchUrl || 'none'}`,
    commandLines.join(' \\\n  ')
  ].join('\n');
}

function buildRobotMissionReleaseCommandFromBundle(
  bundleLike,
  fileNameLike,
  fallbackConfig = null
) {
  const fileName =
    readNonEmptyString(fileNameLike) ||
    buildRobotMissionArtifactPackFileName(bundleLike);
  const releaseCommandLines = [
    'npm run release:robot-mission --',
    `--bundle ${quoteShellValue(`/absolute/path/to/${fileName}`)}`,
    '--force',
    '--validate'
  ];

  if (!fallbackConfig) {
    return releaseCommandLines.join(' \\\n  ');
  }

  const preview = buildPublishedRobotMissionPreviewFromBundle(bundleLike, fallbackConfig);
  const health = buildRobotMissionPreflightHealth(bundleLike, fallbackConfig);

  return [
    `# release input: /absolute/path/to/${fileName}`,
    `# preflight: ${health.label}`,
    `# detail: ${health.detail || 'none'}`,
    `# target: ${preview.payload.fragmentId || 'none'} / ${preview.payload.world.assetLabel || 'none'} / ${preview.payload.world.frameId || 'none'}`,
    `# zone: ${preview.payload.zoneMapUrl || 'none'}`,
    `# launch: ${preview.payload.launchUrl || 'none'}`,
    `# auto outputs: /absolute/path/to/${fileName.replace(/\.json$/i, '.preflight.txt')} + /absolute/path/to/${fileName.replace(/\.json$/i, '.publish-report.json')}`,
    releaseCommandLines.join(' \\\n  ')
  ].join('\n');
}

function buildPublishedRobotMissionFileName(missionLike, fallbackFragmentId = 'residency') {
  const mission = normalizeRobotMissionPayload(missionLike);
  const missionSlug = buildMissionSlug(
    mission.id || mission.label,
    `${fallbackFragmentId}-robot-mission`
  );

  return `${missionSlug}.mission.json`;
}

function buildPublishedRobotMissionPayloadFromBundle(bundleLike, activeConfig) {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const mission = normalizeRobotMissionPayload(bundle.mission);
  const route = normalizeRobotRoutePayload(bundle.route);
  const fragmentId =
    readNonEmptyString(mission.fragmentId) ||
    readNonEmptyString(route.fragmentId) ||
    readNonEmptyString(bundle.fragmentId) ||
    readNonEmptyString(activeConfig?.fragmentId) ||
    'residency';
  const fragmentLabel =
    readNonEmptyString(mission.fragmentLabel) ||
    readNonEmptyString(route.fragmentLabel) ||
    readNonEmptyString(activeConfig?.fragmentLabel) ||
    fragmentId;
  const routeId =
    extractRobotRouteIdFromUrl(mission.routeUrl) ||
    buildMissionSlug(route.label || `${fragmentId}-route`, `${fragmentId}-route`);
  const routeLabel =
    readNonEmptyString(route.label) || `${fragmentLabel} Route Preset`;
  const missionId = buildMissionSlug(
    mission.id || routeId,
    `${routeId}-mission`
  );
  const missionLabel =
    readNonEmptyString(mission.label) ||
    (routeLabel.includes('Mission')
      ? routeLabel
      : `${routeLabel} Mission`);
  const missionDescription =
    readNonEmptyString(mission.description) ||
    `${routeLabel} mission bundle`;
  const streamSceneId =
    readNonEmptyString(mission.streamSceneId) ||
    readNonEmptyString(activeConfig?.streamScenes?.[0]?.id);
  const startupStreamScene = streamSceneId
    ? activeConfig?.streamScenes?.find((scene) => scene.id === streamSceneId) ?? null
    : null;
  const cameraPresetId =
    readNonEmptyString(mission.cameraPresetId) ||
    readNonEmptyString(startupStreamScene?.presetId) ||
    readNonEmptyString(activeConfig?.homePresetId);
  const robotCameraId =
    readNonEmptyString(mission.robotCameraId) ||
    readNonEmptyString(activeConfig?.robotics?.defaultCameraId) ||
    readNonEmptyString(activeConfig?.robotics?.cameras?.[0]?.id);
  const startupMode =
    normalizeRobotMissionStartupMode(mission.startupMode) || 'robot';
  const zoneMapUrl =
    readNonEmptyString(mission.zoneMapUrl) ||
    readNonEmptyString(route.world?.zoneMapUrl) ||
    readNonEmptyString(activeConfig?.robotics?.semanticZoneMapUrl);
  const accent =
    readNonEmptyString(mission.accent) ||
    readNonEmptyString(activeConfig?.overlayBranding?.highlight) ||
    readNonEmptyString(activeConfig?.overlayBranding?.accent) ||
    '#85e3e1';
  const worldAssetLabel =
    readNonEmptyString(mission.world?.assetLabel) ||
    readNonEmptyString(route.world?.assetLabel);
  const worldFrameId =
    readNonEmptyString(mission.world?.frameId) ||
    readNonEmptyString(route.world?.frameId) ||
    readNonEmptyString(route.frameId) ||
    'dreamwalker_map';
  const routeUrl = `/robot-routes/${routeId}.json`;
  const missionUrl = `/robot-missions/${missionId}.mission.json`;

  return {
    version: 1,
    protocol: robotMissionProtocolId,
    id: missionId,
    label: missionLabel,
    description: missionDescription,
    fragmentId,
    fragmentLabel,
    accent,
    routeUrl,
    zoneMapUrl,
    launchUrl: `/?robotMission=${encodeURIComponent(missionUrl)}`,
    cameraPresetId,
    robotCameraId,
    streamSceneId,
    startupMode,
    world: {
      assetLabel: worldAssetLabel,
      frameId: worldFrameId
    }
  };
}

function resolveConfigForRobotMissionBundle(bundleLike, fallbackConfig) {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const fragmentId =
    readNonEmptyString(bundle.mission.fragmentId) ||
    readNonEmptyString(bundle.fragmentId) ||
    readNonEmptyString(bundle.route.fragmentId) ||
    readNonEmptyString(fallbackConfig?.fragmentId) ||
    dreamwalkerConfig.defaultFragmentId;

  if (Object.hasOwn(dreamwalkerConfig.fragments, fragmentId)) {
    return resolveDreamwalkerConfig(fragmentId);
  }

  return fallbackConfig;
}

function resolveMissionPreviewCameraPresetLabel(config, presetId) {
  if (!readNonEmptyString(presetId)) {
    return 'none';
  }

  const localLabel = config.cameraPresets.find((preset) => preset.id === presetId)?.label;
  if (readNonEmptyString(localLabel)) {
    return localLabel;
  }

  for (const fragment of Object.values(dreamwalkerConfig.fragments)) {
    const fragmentLabel = fragment.cameraPresets?.find((preset) => preset.id === presetId)?.label;
    if (readNonEmptyString(fragmentLabel)) {
      return fragmentLabel;
    }
  }

  return presetId;
}

function resolveMissionPreviewRobotCameraLabel(config, cameraId) {
  if (!readNonEmptyString(cameraId)) {
    return 'none';
  }

  const localLabel = config.robotics.cameras.find((camera) => camera.id === cameraId)?.label;
  if (readNonEmptyString(localLabel)) {
    return localLabel;
  }

  for (const fragment of Object.values(dreamwalkerConfig.fragments)) {
    const fragmentLabel = fragment.robotics?.cameras?.find((camera) => camera.id === cameraId)?.label;
    if (readNonEmptyString(fragmentLabel)) {
      return fragmentLabel;
    }
  }

  return cameraId;
}

function resolveMissionPreviewStreamSceneLabel(config, streamSceneId) {
  if (!readNonEmptyString(streamSceneId)) {
    return 'none';
  }

  const localScene = config.streamScenes.find((scene) => scene.id === streamSceneId) ?? null;
  const localLabel =
    readNonEmptyString(localScene?.title) || readNonEmptyString(localScene?.label);
  if (readNonEmptyString(localLabel)) {
    return localLabel;
  }

  for (const fragment of Object.values(dreamwalkerConfig.fragments)) {
    const fragmentScene = fragment.streamScenes?.find((scene) => scene.id === streamSceneId) ?? null;
    const fragmentLabel =
      readNonEmptyString(fragmentScene?.title) || readNonEmptyString(fragmentScene?.label);
    if (readNonEmptyString(fragmentLabel)) {
      return fragmentLabel;
    }
  }

  return streamSceneId;
}

function buildPublishedRobotMissionPreviewFromBundle(bundleLike, fallbackConfig) {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const config = resolveConfigForRobotMissionBundle(bundleLike, fallbackConfig);
  const payload = buildPublishedRobotMissionPayloadFromBundle(bundleLike, config);
  const fileName = buildPublishedRobotMissionFileName(payload, config.fragmentId);
  const routeId =
    extractRobotRouteIdFromUrl(payload.routeUrl) ||
    buildMissionSlug(payload.fragmentId, `${config.fragmentId}-route`);
  const routeFileName = routeId ? `${routeId}.json` : 'unknown-route.json';
  const cameraPresetLabel = resolveMissionPreviewCameraPresetLabel(
    config,
    payload.cameraPresetId
  );
  const robotCameraLabel = resolveMissionPreviewRobotCameraLabel(
    config,
    payload.robotCameraId
  );
  const streamSceneLabel = resolveMissionPreviewStreamSceneLabel(
    config,
    payload.streamSceneId
  );

  return {
    config,
    payload,
    fileName,
    routeId,
    routeFileName,
    routeLabel:
      readNonEmptyString(bundle.route.label) ||
      readNonEmptyString(bundle.route.id) ||
      'none',
    routeDescription: readNonEmptyString(bundle.route.description) || 'none',
    routeAccent: readNonEmptyString(bundle.route.accent) || 'none',
    cameraPresetLabel,
    robotCameraLabel,
    streamSceneLabel,
    missionId: payload.id || buildMissionSlug(payload.label, `${config.fragmentId}-robot-mission`),
    fragmentId: payload.fragmentId || config.fragmentId
  };
}

function buildRobotMissionPreflightHealth(bundleLike, fallbackConfig) {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const preview = buildPublishedRobotMissionPreviewFromBundle(bundle, fallbackConfig);
  const targetWorld = normalizeRobotRouteWorldContext(
    {
      fragmentId: preview.payload.fragmentId || preview.config.fragmentId,
      fragmentLabel: preview.payload.fragmentLabel || preview.config.fragmentLabel,
      assetLabel: preview.payload.world.assetLabel,
      frameId: preview.payload.world.frameId,
      zoneMapUrl: preview.payload.zoneMapUrl
    },
    {
      fragmentId: preview.config.fragmentId,
      fragmentLabel: preview.config.fragmentLabel,
      frameId: preview.payload.world.frameId || 'dreamwalker_map'
    }
  );

  return buildRobotMissionCompatibility(bundle.mission, bundle.route, targetWorld);
}

function buildRobotMissionPreflightSummary(bundleLike, fallbackConfig) {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const preview = buildPublishedRobotMissionPreviewFromBundle(bundle, fallbackConfig);
  const health = buildRobotMissionPreflightHealth(bundle, fallbackConfig);

  return [
    `status: ${health.label || 'unknown'}`,
    `detail: ${health.detail || 'none'}`,
    `missionId: ${preview.missionId || 'none'}`,
    `missionLabel: ${preview.payload.label || 'none'}`,
    `missionDescription: ${preview.payload.description || 'none'}`,
    `fragmentId: ${preview.payload.fragmentId || 'none'}`,
    `fragmentLabel: ${preview.payload.fragmentLabel || 'none'}`,
    `routeId: ${preview.routeId || 'none'}`,
    `routeFile: ${preview.routeFileName || 'none'}`,
    `routeLabel: ${preview.routeLabel || 'none'}`,
    `routeDescription: ${preview.routeDescription || 'none'}`,
    `routeAccent: ${preview.routeAccent || 'none'}`,
    `worldAsset: ${preview.payload.world.assetLabel || 'none'}`,
    `worldFrame: ${preview.payload.world.frameId || 'none'}`,
    `zoneMapUrl: ${preview.payload.zoneMapUrl || 'none'}`,
    `startupMode: ${preview.payload.startupMode || 'none'}`,
    `cameraPresetId: ${preview.payload.cameraPresetId || 'none'}`,
    `cameraPresetLabel: ${preview.cameraPresetLabel || 'none'}`,
    `robotCameraId: ${preview.payload.robotCameraId || 'none'}`,
    `robotCameraLabel: ${preview.robotCameraLabel || 'none'}`,
    `streamSceneId: ${preview.payload.streamSceneId || 'none'}`,
    `streamSceneLabel: ${preview.streamSceneLabel || 'none'}`,
    `launchUrl: ${preview.payload.launchUrl || 'none'}`
  ].join('\n');
}

function buildRobotMissionPublishReport(bundleLike, fallbackConfig) {
  const bundle = normalizeRobotMissionDraftBundle(bundleLike);
  const preview = buildPublishedRobotMissionPreviewFromBundle(bundle, fallbackConfig);
  const health = buildRobotMissionPreflightHealth(bundle, fallbackConfig);
  const missionUrl = preview.fileName ? `/robot-missions/${preview.fileName}` : '';

  return {
    version: 1,
    protocol: 'dreamwalker-robot-mission-publish-report/v1',
    dryRun: false,
    fragmentId: preview.payload.fragmentId || preview.config.fragmentId || '',
    publicRoot: '',
    mission: {
      id: preview.missionId || '',
      label: preview.payload.label || '',
      description: preview.payload.description || '',
      accent: preview.payload.accent || '',
      url: missionUrl,
      path: '',
      launchUrl: preview.payload.launchUrl || '',
      catalogPath: '',
      catalogUrl: '/robot-missions/index.json'
    },
    route: {
      id: preview.routeId || '',
      fileName: preview.routeFileName || '',
      label: preview.routeLabel || '',
      description: preview.routeDescription || '',
      accent: preview.routeAccent || '',
      url: preview.payload.routeUrl || '',
      path: '',
      catalogPath: '',
      source: ''
    },
    zones: {
      url: preview.payload.zoneMapUrl || '',
      path: '',
      source: ''
    },
    world: {
      assetLabel: preview.payload.world.assetLabel || '',
      frameId: preview.payload.world.frameId || ''
    },
    startup: {
      mode: preview.payload.startupMode || '',
      cameraPresetId: preview.payload.cameraPresetId || '',
      cameraPresetLabel: preview.cameraPresetLabel || '',
      robotCameraId: preview.payload.robotCameraId || '',
      robotCameraLabel: preview.robotCameraLabel || '',
      streamSceneId: preview.payload.streamSceneId || '',
      streamSceneLabel: preview.streamSceneLabel || ''
    },
    preflight: {
      label: health.label || '',
      detail: health.detail || '',
      summary: buildRobotMissionPreflightSummary(bundle, fallbackConfig),
      outputPath: ''
    },
    outputs: {
      reportOutputPath: ''
    },
    validation: {
      requested: false
    }
  };
}

function clampUnit(value) {
  return Math.max(0, Math.min(1, value));
}

function resolveSemanticZonePinDepthStyle(zDepth, isActive) {
  const depth = Number.isFinite(zDepth) ? zDepth : 0;
  const normalized = clampUnit((depth - 4) / 28);
  const opacity = isActive ? 1 - normalized * 0.22 : 0.92 - normalized * 0.52;
  const scale = isActive ? 1.04 - normalized * 0.1 : 1 - normalized * 0.12;

  return {
    opacity: Number(Math.max(isActive ? 0.78 : 0.32, opacity).toFixed(3)),
    scale: Number(Math.max(isActive ? 0.94 : 0.86, scale).toFixed(3)),
    zIndex: isActive ? 2000 : Math.max(20, 1500 - Math.round(depth * 12))
  };
}

function resolveSemanticZoneSurfaceDepthStyle(zDepth, isActive) {
  const depth = Number.isFinite(zDepth) ? zDepth : 0;
  const normalized = clampUnit((depth - 4) / 30);
  const depthOpacity = isActive ? 0.96 - normalized * 0.24 : 0.82 - normalized * 0.38;
  const strokeOpacity = isActive ? 0.98 - normalized * 0.16 : 0.8 - normalized * 0.34;

  return {
    fillOpacity: Number(Math.max(isActive ? 0.68 : 0.28, depthOpacity).toFixed(3)),
    strokeOpacity: Number(Math.max(isActive ? 0.84 : 0.32, strokeOpacity).toFixed(3))
  };
}

function RoboticsOverlay({ points }) {
  if (points.length === 0) {
    return null;
  }

  return (
    <div className="robotics-overlay" aria-hidden="false">
      {points.map((point) => (
        <div
          key={point.id}
          className={`robotics-pin robotics-${point.kind}`}
          style={{
            left: `${point.xPercent}%`,
            top: `${point.yPercent}%`,
            borderColor: point.accentColor
          }}>
          <span className="robotics-core" style={{ backgroundColor: point.accentColor }} />
          <span className="robotics-label">{point.label}</span>
        </div>
      ))}
    </div>
  );
}

function RobotRouteOverlay({ points }) {
  const trailPoints = points
    .filter((point) => point.kind === 'route-node')
    .sort((left, right) => left.order - right.order);
  const waypointPoint = points.find((point) => point.kind === 'route-waypoint-anchor') ?? null;
  const trailPolyline = trailPoints.map((point) => `${point.xPercent},${point.yPercent}`).join(' ');
  const routeAnchor = trailPoints[trailPoints.length - 1] ?? null;

  if (!routeAnchor && !waypointPoint) {
    return null;
  }

  return (
    <div className="robotics-route-overlay" aria-hidden="true">
      <svg
        className="robotics-route-svg"
        preserveAspectRatio="none"
        viewBox="0 0 100 100">
        {trailPoints.length >= 2 ? (
          <polyline className="robotics-route-trail" points={trailPolyline} />
        ) : null}
        {routeAnchor && waypointPoint ? (
          <line
            className="robotics-route-goal"
            x1={routeAnchor.xPercent}
            x2={waypointPoint.xPercent}
            y1={routeAnchor.yPercent}
            y2={waypointPoint.yPercent}
          />
        ) : null}
        {trailPoints.map((point) => (
          <circle
            key={point.id}
            className="robotics-route-node"
            cx={point.xPercent}
            cy={point.yPercent}
            r="0.44"
          />
        ))}
      </svg>
    </div>
  );
}

function BenchmarkRouteOverlay({ benchmark, points }) {
  if (!benchmark) {
    return null;
  }

  const groundTruthPoints = points
    .filter((point) => point.kind === 'benchmark-ground-truth-node')
    .sort((left, right) => left.order - right.order);
  const estimatePoints = points
    .filter((point) => point.kind === 'benchmark-estimate-node')
    .sort((left, right) => left.order - right.order);
  const groundTruthPolyline = groundTruthPoints
    .map((point) => `${point.xPercent},${point.yPercent}`)
    .join(' ');
  const estimatePolyline = estimatePoints
    .map((point) => `${point.xPercent},${point.yPercent}`)
    .join(' ');
  const estimatePointMap = new Map(
    estimatePoints.map((point) => [point.sampleIndex, point])
  );
  const connectors = groundTruthPoints
    .map((point) => {
      const estimatePoint = estimatePointMap.get(point.sampleIndex);

      if (!estimatePoint) {
        return null;
      }

      return {
        id: `benchmark-connector-${point.sampleIndex}`,
        isWorst: point.isWorst || estimatePoint.isWorst,
        x1: point.xPercent,
        y1: point.yPercent,
        x2: estimatePoint.xPercent,
        y2: estimatePoint.yPercent
      };
    })
    .filter(Boolean);

  return (
    <div className="benchmark-route-overlay" aria-hidden="true">
      {groundTruthPoints.length > 0 || estimatePoints.length > 0 ? (
        <svg
          className="benchmark-route-svg"
          preserveAspectRatio="none"
          viewBox="0 0 100 100">
          {connectors.map((connector) => (
            <line
              key={connector.id}
              className={connector.isWorst ? 'benchmark-route-connector worst' : 'benchmark-route-connector'}
              x1={connector.x1}
              x2={connector.x2}
              y1={connector.y1}
              y2={connector.y2}
            />
          ))}
          {groundTruthPoints.length >= 2 ? (
            <polyline
              className="benchmark-route-ground-truth"
              points={groundTruthPolyline}
            />
          ) : null}
          {estimatePoints.length >= 2 ? (
            <polyline
              className="benchmark-route-estimate"
              points={estimatePolyline}
            />
          ) : null}
          {groundTruthPoints.map((point) => (
            <circle
              key={point.id}
              className={point.isWorst ? 'benchmark-route-ground-truth-node worst' : 'benchmark-route-ground-truth-node'}
              cx={point.xPercent}
              cy={point.yPercent}
              r={point.isWorst ? '0.7' : '0.5'}
            />
          ))}
          {estimatePoints.map((point) => (
            <circle
              key={point.id}
              className={point.isWorst ? 'benchmark-route-estimate-node worst' : 'benchmark-route-estimate-node'}
              cx={point.xPercent}
              cy={point.yPercent}
              r={point.isWorst ? '0.68' : '0.46'}
            />
          ))}
        </svg>
      ) : null}
      <div className="benchmark-route-legend">
        <span className="benchmark-route-chip benchmark-route-chip-ground-truth">
          GT {benchmark.groundTruthLabel}
        </span>
        <span className="benchmark-route-chip benchmark-route-chip-estimate">
          EST {benchmark.estimateLabel}
        </span>
        <span className="benchmark-route-chip">
          ATE {Number.isFinite(benchmark.ateRmseMeters) ? `${benchmark.ateRmseMeters.toFixed(3)} m` : 'n/a'}
        </span>
        <span className="benchmark-route-chip">
          Match {benchmark.matchedCount}
        </span>
      </div>
    </div>
  );
}

function SemanticZoneOverlay({ points }) {
  if (points.length === 0) {
    return null;
  }

  const orderedPoints = [...points].sort((left, right) => {
    if (left.isActive !== right.isActive) {
      return left.isActive ? 1 : -1;
    }

    return right.zDepth - left.zDepth;
  });

  return (
    <div className="semantic-zone-overlay" aria-hidden="true">
      {orderedPoints.map((point) => {
        const depthStyle = resolveSemanticZonePinDepthStyle(point.zDepth, point.isActive);

        return (
          <div
            key={point.id}
            className={`semantic-zone-pin${point.isActive ? ' active' : ''}`}
            style={{
              left: `${point.xPercent}%`,
              top: `${point.yPercent}%`,
              borderColor: point.accentColor,
              opacity: depthStyle.opacity,
              transform: `translate(-50%, -50%) scale(${depthStyle.scale})`,
              zIndex: depthStyle.zIndex
            }}>
            <span
              className="semantic-zone-core"
              style={{ backgroundColor: point.accentColor }}
            />
            <span className="semantic-zone-label">
              {point.label}
              <small>{point.cost}</small>
            </span>
          </div>
        );
      })}
    </div>
  );
}

function SemanticZoneSurfaceOverlay({ zones }) {
  if (zones.length === 0) {
    return null;
  }

  const orderedZones = [...zones].sort((left, right) => {
    if (left.isActive !== right.isActive) {
      return left.isActive ? 1 : -1;
    }

    return right.averageDepth - left.averageDepth;
  });

  return (
    <div className="semantic-zone-surface-overlay" aria-hidden="true">
      <svg
        className="semantic-zone-surface-svg"
        preserveAspectRatio="none"
        viewBox="0 0 100 100">
        {orderedZones.map((zone) => {
          const depthStyle = resolveSemanticZoneSurfaceDepthStyle(
            zone.averageDepth,
            zone.isActive
          );

          return (
            <polygon
              key={zone.id}
              className={`semantic-zone-surface-shape${zone.isActive ? ' active' : ''}`}
              fill={zone.accentColor}
              fillOpacity={Number((zone.fillOpacity * depthStyle.fillOpacity).toFixed(3))}
              points={zone.points}
              stroke={zone.accentColor}
              strokeOpacity={depthStyle.strokeOpacity}
            />
          );
        })}
      </svg>
    </div>
  );
}

function mapSemanticNavPoint(zoneMap, x, z) {
  if (!zoneMap) {
    return null;
  }

  const width = Math.max(0.001, zoneMap.maxX - zoneMap.minX);
  const depth = Math.max(0.001, zoneMap.maxZ - zoneMap.minZ);
  const normalizedX = ((x - zoneMap.minX) / width) * 100;
  const normalizedY = 100 - ((z - zoneMap.minZ) / depth) * 100;

  return {
    x: Number(normalizedX.toFixed(2)),
    y: Number(normalizedY.toFixed(2))
  };
}

function SemanticNavPanel({
  zoneMap,
  robotPose,
  waypoint,
  robotTrail,
  activeZoneIds,
  benchmarkOverlay
}) {
  if (!zoneMap) {
    return null;
  }

  const activeIds = new Set(activeZoneIds);
  const robotPoint = mapSemanticNavPoint(zoneMap, robotPose.position[0], robotPose.position[2]);
  const waypointPoint = waypoint
    ? mapSemanticNavPoint(zoneMap, waypoint.position[0], waypoint.position[2])
    : null;
  const routePolyline = robotTrail
    .map((position) => mapSemanticNavPoint(zoneMap, position[0], position[2]))
    .filter(Boolean)
    .map((point) => `${point.x},${point.y}`)
    .join(' ');
  const forwardVector = getForwardVector(robotPose.yawDegrees);
  const headingPoint = mapSemanticNavPoint(
    zoneMap,
    robotPose.position[0] + forwardVector.x * 1.1,
    robotPose.position[2] + forwardVector.z * 1.1
  );
  const benchmarkGroundTruthPoints = benchmarkOverlay?.samples
    ?.map((sample) => ({
      index: sample.index,
      isWorst: benchmarkOverlay.worstSampleIndex === sample.index,
      point: mapSemanticNavPoint(
        zoneMap,
        sample.groundTruthPosition[0],
        sample.groundTruthPosition[2]
      )
    }))
    .filter((sample) => sample.point) ?? [];
  const benchmarkEstimatePoints = benchmarkOverlay?.samples
    ?.map((sample) => ({
      index: sample.index,
      isWorst: benchmarkOverlay.worstSampleIndex === sample.index,
      point: mapSemanticNavPoint(
        zoneMap,
        sample.estimatePosition[0],
        sample.estimatePosition[2]
      )
    }))
    .filter((sample) => sample.point) ?? [];
  const benchmarkGroundTruthPolyline = benchmarkGroundTruthPoints
    .map((sample) => `${sample.point.x},${sample.point.y}`)
    .join(' ');
  const benchmarkEstimatePolyline = benchmarkEstimatePoints
    .map((sample) => `${sample.point.x},${sample.point.y}`)
    .join(' ');
  const benchmarkEstimatePointMap = new Map(
    benchmarkEstimatePoints.map((sample) => [sample.index, sample.point])
  );
  const benchmarkConnectorPairs = benchmarkGroundTruthPoints
    .map((sample) => {
      const estimatePoint = benchmarkEstimatePointMap.get(sample.index);

      if (!estimatePoint) {
        return null;
      }

      return {
        index: sample.index,
        isWorst: sample.isWorst,
        groundTruthPoint: sample.point,
        estimatePoint
      };
    })
    .filter(Boolean);

  return (
    <div className="state-card robot-nav-panel">
      <span className="state-label">Nav / Cost Overlay</span>
      <strong>{zoneMap.frameId}</strong>
      <p className="panel-note">
        top-down で zone cost、robot pose、route、waypoint
        {benchmarkOverlay ? '、benchmark path' : ''}
        をまとめて確認します。
      </p>
      <svg
        className="robot-nav-svg"
        preserveAspectRatio="none"
        viewBox="0 0 100 100">
        <defs>
          <pattern
            id="robot-nav-grid"
            width="10"
            height="10"
            patternUnits="userSpaceOnUse">
            <path
              d="M 10 0 L 0 0 0 10"
              fill="none"
              stroke="rgba(164, 211, 216, 0.12)"
              strokeWidth="0.25"
            />
          </pattern>
        </defs>
        <rect className="robot-nav-bounds" x="0" y="0" width="100" height="100" />
        <rect className="robot-nav-grid" x="0" y="0" width="100" height="100" fill="url(#robot-nav-grid)" />
        {zoneMap.zones.map((zone) => {
          const isActive = activeIds.has(zone.id);
          const fillOpacity = 0.12 + zone.cost / 220;

          if (zone.shape === 'rect') {
            const minPoint = mapSemanticNavPoint(
              zoneMap,
              zone.centerX - zone.sizeX / 2,
              zone.centerZ + zone.sizeZ / 2
            );
            const maxPoint = mapSemanticNavPoint(
              zoneMap,
              zone.centerX + zone.sizeX / 2,
              zone.centerZ - zone.sizeZ / 2
            );

            return (
              <rect
                key={zone.id}
                className={`robot-nav-zone${isActive ? ' active' : ''}`}
                x={minPoint.x}
                y={maxPoint.y}
                width={Math.max(0.8, maxPoint.x - minPoint.x)}
                height={Math.max(0.8, minPoint.y - maxPoint.y)}
                fill={zone.accentColor}
                fillOpacity={fillOpacity}
                stroke={zone.accentColor}
              />
            );
          }

          const centerPoint = mapSemanticNavPoint(zoneMap, zone.centerX, zone.centerZ);
          const xRadius = (zone.radius / Math.max(0.001, zoneMap.maxX - zoneMap.minX)) * 100;
          const yRadius = (zone.radius / Math.max(0.001, zoneMap.maxZ - zoneMap.minZ)) * 100;

          return (
            <ellipse
              key={zone.id}
              className={`robot-nav-zone${isActive ? ' active' : ''}`}
              cx={centerPoint.x}
              cy={centerPoint.y}
              rx={Math.max(0.8, xRadius)}
              ry={Math.max(0.8, yRadius)}
              fill={zone.accentColor}
              fillOpacity={fillOpacity}
              stroke={zone.accentColor}
            />
          );
        })}
        {routePolyline ? (
          <polyline className="robot-nav-route" points={routePolyline} />
        ) : null}
        {benchmarkConnectorPairs.map((sample) => (
          <line
            key={`robot-nav-benchmark-connector-${sample.index}`}
            className={sample.isWorst ? 'robot-nav-benchmark-connector worst' : 'robot-nav-benchmark-connector'}
            x1={sample.groundTruthPoint.x}
            y1={sample.groundTruthPoint.y}
            x2={sample.estimatePoint.x}
            y2={sample.estimatePoint.y}
          />
        ))}
        {benchmarkGroundTruthPoints.length >= 2 ? (
          <polyline
            className="robot-nav-benchmark-ground-truth"
            points={benchmarkGroundTruthPolyline}
          />
        ) : null}
        {benchmarkEstimatePoints.length >= 2 ? (
          <polyline
            className="robot-nav-benchmark-estimate"
            points={benchmarkEstimatePolyline}
          />
        ) : null}
        {robotPoint && headingPoint ? (
          <line
            className="robot-nav-heading"
            x1={robotPoint.x}
            y1={robotPoint.y}
            x2={headingPoint.x}
            y2={headingPoint.y}
          />
        ) : null}
        {waypointPoint ? (
          <circle className="robot-nav-waypoint" cx={waypointPoint.x} cy={waypointPoint.y} r="2.2" />
        ) : null}
        {robotPoint ? (
          <circle className="robot-nav-robot" cx={robotPoint.x} cy={robotPoint.y} r="2.4" />
        ) : null}
        {benchmarkGroundTruthPoints.map((sample) => (
          <circle
            key={`robot-nav-benchmark-ground-truth-${sample.index}`}
            className={sample.isWorst ? 'robot-nav-benchmark-ground-truth-node worst' : 'robot-nav-benchmark-ground-truth-node'}
            cx={sample.point.x}
            cy={sample.point.y}
            r={sample.isWorst ? '1.5' : '1.2'}
          />
        ))}
        {benchmarkEstimatePoints.map((sample) => (
          <circle
            key={`robot-nav-benchmark-estimate-${sample.index}`}
            className={sample.isWorst ? 'robot-nav-benchmark-estimate-node worst' : 'robot-nav-benchmark-estimate-node'}
            cx={sample.point.x}
            cy={sample.point.y}
            r={sample.isWorst ? '1.35' : '1.05'}
          />
        ))}
      </svg>
      <div className="robot-nav-legend">
        <span>
          robot {robotPose.position[0].toFixed(1)}, {robotPose.position[2].toFixed(1)}
        </span>
        <span>
          bounds x {zoneMap.minX.toFixed(1)}..{zoneMap.maxX.toFixed(1)} / z {zoneMap.minZ.toFixed(1)}..{zoneMap.maxZ.toFixed(1)}
        </span>
        {benchmarkOverlay ? (
          <span>
            benchmark ATE {benchmarkOverlay.ateRmseMeters?.toFixed(3) ?? 'n/a'} m / match {benchmarkOverlay.matchedCount}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function EchoNoteModal({ hotspot, onClose }) {
  if (!hotspot) {
    return null;
  }

  return (
    <div className="echo-modal-backdrop" onClick={onClose} role="presentation">
      <section
        aria-label={hotspot.title}
        className="echo-modal glass-panel"
        onClick={(event) => event.stopPropagation()}>
        <div
          className="echo-modal-stripe"
          style={{ backgroundColor: hotspot.accentColor }}
        />
        <p className="echo-kind">{hotspot.kind}</p>
        <h2>{hotspot.title}</h2>
        <p className="echo-body">{hotspot.body}</p>
        <div className="echo-actions">
          {hotspot.presetId ? (
            <p className="echo-tip">このノートは camera preset と連動しています。</p>
          ) : null}
          <button className="ghost-button" onClick={onClose} type="button">
            Close
          </button>
        </div>
      </section>
    </div>
  );
}

function getReticleHint(item, candidate) {
  if (!item && !candidate) {
    return `${interactKey.toUpperCase()}: interactable を中央に捉える`;
  }

  if (!item && candidate) {
    return `${interactKey.toUpperCase()}: ${candidate.label} に近づく`;
  }

  if (item.kind === 'distortion-shard') {
    return `${interactKey.toUpperCase()}: ${item.label} を回収`;
  }

  if (item.kind === 'dream-gate') {
    return `${interactKey.toUpperCase()}: ${item.label}`;
  }

  return `${interactKey.toUpperCase()}: ${item.label} を開く`;
}

export default function App() {
  const isOverlayMode = useMemo(() => parseOverlayModeFromSearch(), []);
  const relayConfig = useMemo(() => parseOverlayRelayConfigFromSearch(), []);
  const robotBridgeConfig = useMemo(() => parseRobotBridgeConfigFromSearch(), []);
  const sim2realConfig = useMemo(() => parseSim2realConfigFromSearch(), []);
  const robotFrameStreamEnabled = useMemo(() => parseRobotFrameStreamEnabledFromSearch(), []);
  const robotDepthStreamEnabled = useMemo(() => parseRobotDepthStreamEnabledFromSearch(), []);
  const robotRoutePlaybackConfig = useMemo(() => parseRobotRoutePlaybackConfigFromSearch(), []);
  const dynamicMapDiagnosticsEnabled = useMemo(
    () => parseDynamicMapDiagnosticsEnabledFromSearch(),
    []
  );
  const initialAssetWorkspace = useMemo(() => loadAssetWorkspaceManifest(), []);
  const initialSceneWorkspace = useMemo(() => loadSceneWorkspace(), []);
  const initialSemanticZoneWorkspace = useMemo(() => loadSemanticZoneWorkspaceMap(), []);
  const assetManifestUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.assetManifest.defaultUrl,
        dreamwalkerConfig.assetManifest.queryParam
      ),
    []
  );
  const tileCatalogUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.tileCatalog.defaultUrl,
        dreamwalkerConfig.tileCatalog.queryParam
      ),
    []
  );
  const selectedTileId = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        '',
        dreamwalkerConfig.tileCatalog.tileQueryParam
      ),
    []
  );
  const dynamicMapTilePreloadMode = useMemo(
    () => parseDynamicMapTilePreloadModeFromSearch(),
    []
  );
  const dynamicMapTilePreloadLimit = useMemo(
    () => parseDynamicMapTilePreloadLimitFromSearch(),
    []
  );
  const dynamicMapTileResidentLimit = useMemo(
    () => parseDynamicMapTileResidentLimitFromSearch(dynamicMapTilePreloadLimit + 1),
    [dynamicMapTilePreloadLimit]
  );
  const studioBundleUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.studioBundle.defaultUrl,
        dreamwalkerConfig.studioBundle.queryParam
      ),
    []
  );
  const studioBundleCatalogUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.studioBundleCatalog.defaultUrl,
        dreamwalkerConfig.studioBundleCatalog.queryParam
      ),
    []
  );
  const robotRouteUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.robotRoute.defaultUrl,
        dreamwalkerConfig.robotRoute.queryParam
      ),
    []
  );
  const robotRouteCatalogUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.robotRouteCatalog.defaultUrl,
        dreamwalkerConfig.robotRouteCatalog.queryParam
      ),
    []
  );
  const robotMissionUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.robotMission.defaultUrl,
        dreamwalkerConfig.robotMission.queryParam
      ),
    []
  );
  const robotMissionCatalogUrl = useMemo(
    () =>
      parseAssetManifestUrlFromSearch(
        dreamwalkerConfig.robotMissionCatalog.defaultUrl,
        dreamwalkerConfig.robotMissionCatalog.queryParam
      ),
    []
  );
  const [currentFragmentId, setCurrentFragmentId] = useState(parseFragmentIdFromHash);
  const [cameraMode, setCameraMode] = useState('orbit');
  const [mode, setMode] = useState('explore');
  const [selectedOverlayPresetId, setSelectedOverlayPresetId] = useState(loadOverlayPresetId);
  const [selectedPresetId, setSelectedPresetId] = useState(
    () => resolveDreamwalkerConfig(parseFragmentIdFromHash()).homePresetId
  );
  const [selectedRatioId, setSelectedRatioId] = useState(
    dreamwalkerConfig.photoRatios[0].id
  );
  const [selectedFilterId, setSelectedFilterId] = useState(
    dreamwalkerConfig.dreamFilters[0].id
  );
  const [selectedStreamSceneId, setSelectedStreamSceneId] = useState(
    () => resolveDreamwalkerConfig(parseFragmentIdFromHash()).streamScenes[0]?.id ?? null
  );
  const [selectedRobotCameraId, setSelectedRobotCameraId] = useState(
    () => resolveDreamwalkerConfig(parseFragmentIdFromHash()).robotics.defaultCameraId ?? 'front'
  );
  const [showGuides, setShowGuides] = useState(true);
  const [statusMessage, setStatusMessage] = useState(
    'DreamWalker Live shell ready'
  );
  const [isPointerLocked, setIsPointerLocked] = useState(false);
  const [collectedShardIds, setCollectedShardIds] = useState(() =>
    loadCollectedShards(buildShardStorageKey(parseFragmentIdFromHash()))
  );
  const [projectedHotspots, setProjectedHotspots] = useState([]);
  const [projectedLoopItems, setProjectedLoopItems] = useState([]);
  const [projectedRobotPoints, setProjectedRobotPoints] = useState([]);
  const [projectedRobotRoutePoints, setProjectedRobotRoutePoints] = useState([]);
  const [projectedBenchmarkRoutePoints, setProjectedBenchmarkRoutePoints] = useState([]);
  const [projectedSemanticZonePoints, setProjectedSemanticZonePoints] = useState([]);
  const [projectedSemanticZoneSurfacePoints, setProjectedSemanticZoneSurfacePoints] = useState([]);
  const [selectedHotspotId, setSelectedHotspotId] = useState(null);
  const [activeModalItem, setActiveModalItem] = useState(null);
  const [robotPose, setRobotPose] = useState(
    () => buildRobotPoseFromConfig(resolveDreamwalkerConfig(parseFragmentIdFromHash()))
  );
  const [robotWaypoint, setRobotWaypoint] = useState(null);
  const [robotTrail, setRobotTrail] = useState(() =>
    buildRobotTrailFromPose(buildRobotPoseFromConfig(resolveDreamwalkerConfig(parseFragmentIdFromHash())))
  );
  const [sim2realBenchmarkOverlay, setSim2RealBenchmarkOverlay] = useState(null);
  const [walkColliderStatus, setWalkColliderStatus] = useState(() =>
    resolveDreamwalkerConfig(parseFragmentIdFromHash()).colliderMeshUrl
      ? { mode: 'idle', error: null }
      : { mode: 'proxy', error: null }
  );
  const [semanticZoneState, setSemanticZoneState] = useState(() => ({
    status: resolveDreamwalkerConfig(parseFragmentIdFromHash()).robotics.semanticZoneMapUrl
      ? 'loading'
      : 'disabled',
    url: resolveDreamwalkerConfig(parseFragmentIdFromHash()).robotics.semanticZoneMapUrl ?? '',
    zoneMap: null,
    error: null
  }));
  const [assetManifestState, setAssetManifestState] = useState(() => ({
    status: assetManifestUrl ? 'loading' : 'disabled',
    manifest: null,
    error: null,
    url: assetManifestUrl
  }));
  const [tileCatalogState, setTileCatalogState] = useState(() => ({
    status: tileCatalogUrl ? 'loading' : 'disabled',
    catalog: null,
    error: null,
    url: tileCatalogUrl
  }));
  const [studioBundleState, setStudioBundleState] = useState(() => ({
    status: studioBundleUrl ? 'loading' : 'disabled',
    bundle: null,
    error: null,
    url: studioBundleUrl
  }));
  const [studioBundleCatalogState, setStudioBundleCatalogState] = useState(() => ({
    status: studioBundleCatalogUrl ? 'loading' : 'disabled',
    catalog: null,
    error: null,
    url: studioBundleCatalogUrl
  }));
  const [robotRouteState, setRobotRouteState] = useState(() => ({
    status: robotRouteUrl ? 'loading' : 'disabled',
    route: null,
    error: null,
    url: robotRouteUrl
  }));
  const [robotRouteCatalogState, setRobotRouteCatalogState] = useState(() => ({
    status: robotRouteCatalogUrl ? 'loading' : 'disabled',
    catalog: null,
    error: null,
    url: robotRouteCatalogUrl
  }));
  const [robotMissionState, setRobotMissionState] = useState(() => ({
    status: robotMissionUrl ? 'loading' : 'disabled',
    mission: null,
    route: null,
    error: null,
    url: robotMissionUrl
  }));
  const [robotMissionCatalogState, setRobotMissionCatalogState] = useState(() => ({
    status: robotMissionCatalogUrl ? 'loading' : 'disabled',
    catalog: null,
    error: null,
    url: robotMissionCatalogUrl
  }));
  const [assetWorkspaceDraft, setAssetWorkspaceDraft] = useState(
    () => initialAssetWorkspace ?? defaultAssetManifestTemplate
  );
  const [assetWorkspaceBaselineJson, setAssetWorkspaceBaselineJson] = useState(
    () => JSON.stringify(initialAssetWorkspace ?? defaultAssetManifestTemplate)
  );
  const [assetWorkspaceImportText, setAssetWorkspaceImportText] = useState('');
  const [assetWorkspaceImportError, setAssetWorkspaceImportError] = useState('');
  const [hasSavedAssetWorkspace, setHasSavedAssetWorkspace] = useState(
    () => Boolean(initialAssetWorkspace)
  );
  const [sceneWorkspaceDraft, setSceneWorkspaceDraft] = useState(
    () => initialSceneWorkspace ?? defaultSceneWorkspaceTemplate
  );
  const [sceneWorkspaceBaselineJson, setSceneWorkspaceBaselineJson] = useState(
    () => JSON.stringify(initialSceneWorkspace ?? defaultSceneWorkspaceTemplate)
  );
  const [sceneWorkspaceImportText, setSceneWorkspaceImportText] = useState('');
  const [sceneWorkspaceImportError, setSceneWorkspaceImportError] = useState('');
  const [hasSavedSceneWorkspace, setHasSavedSceneWorkspace] = useState(
    () => Boolean(initialSceneWorkspace)
  );
  const [semanticZoneWorkspaceDrafts, setSemanticZoneWorkspaceDrafts] = useState(
    () => initialSemanticZoneWorkspace
  );
  const [semanticZoneWorkspaceBaselineJson, setSemanticZoneWorkspaceBaselineJson] = useState(
    () => JSON.stringify(initialSemanticZoneWorkspace)
  );
  const [semanticZoneImportText, setSemanticZoneImportText] = useState('');
  const [semanticZoneImportError, setSemanticZoneImportError] = useState('');
  const [hasSavedSemanticZoneWorkspace, setHasSavedSemanticZoneWorkspace] = useState(
    () => Object.keys(initialSemanticZoneWorkspace).length > 0
  );
  const [robotRouteImportText, setRobotRouteImportText] = useState('');
  const [robotRouteImportError, setRobotRouteImportError] = useState('');
  const [robotMissionDraftBundleImportText, setRobotMissionDraftBundleImportText] = useState('');
  const [robotMissionDraftBundleImportError, setRobotMissionDraftBundleImportError] = useState('');
  const [robotRouteShelf, setRobotRouteShelf] = useState(loadRobotRouteShelf);
  const [robotRouteShelfLabel, setRobotRouteShelfLabel] = useState('');
  const [robotMissionDraftBundleShelf, setRobotMissionDraftBundleShelf] = useState(
    loadRobotMissionDraftBundleShelf
  );
  const [robotMissionDraftBundleShelfLabel, setRobotMissionDraftBundleShelfLabel] = useState('');
  const [studioBundleImportText, setStudioBundleImportText] = useState('');
  const [studioBundleImportError, setStudioBundleImportError] = useState('');
  const [activeWorldHealth, setActiveWorldHealth] = useState({
    status: 'loading',
    label: 'Checking',
    detail: 'world asset を確認中です。'
  });
  const [mapRuntimeState, setMapRuntimeState] = useState({
    status: 'idle',
    url: '',
    fragmentId: ''
  });
  const [mapPositionState, setMapPositionState] = useState({
    status: 'idle',
    source: '',
    position: null,
    timestamp: 0
  });
  const [diagnosticMapPositionOverride, setDiagnosticMapPositionOverride] = useState(null);
  const [activeDynamicMapTileId, setActiveDynamicMapTileId] = useState('');
  const [tilePreloadState, setTilePreloadState] = useState(() => ({
    status: 'idle',
    mode: dynamicMapTilePreloadMode,
    activeTileId: '',
    entries: [],
    results: [],
    error: null
  }));
  const [robotRoutePlaybackPreviewPositions, setRobotRoutePlaybackPreviewPositions] = useState([]);
  const [studioBundleShelf, setStudioBundleShelf] = useState(loadStudioBundleShelf);
  const [studioBundleShelfLabel, setStudioBundleShelfLabel] = useState('');
  const [publicStudioBundleHealthMap, setPublicStudioBundleHealthMap] = useState({});
  const [publicRobotRouteHealthMap, setPublicRobotRouteHealthMap] = useState({});
  const [publicRobotMissionHealthMap, setPublicRobotMissionHealthMap] = useState({});
  const [overlayState, setOverlayState] = useState(loadOverlayState);
  const [robotBridgeState, setRobotBridgeState] = useState(() => ({
    status: robotBridgeConfig.enabled ? 'connecting' : 'disabled',
    lastInboundType: null,
    lastOutboundType: null,
    error: null
  }));
  const [gamepadState, setGamepadState] = useState({
    connected: false,
    label: 'No Gamepad',
    mapping: null
  });
  const [robotBridgeReconnectNonce, setRobotBridgeReconnectNonce] = useState(0);
  const assetWorkspaceTouchedRef = useRef(false);
  const pendingStudioStateRef = useRef(null);
  const pendingRobotRouteRef = useRef(null);
  const pendingRobotMissionStartupRef = useRef(null);
  const tilePreloadResultCacheRef = useRef(dynamicMapTilePreloadResultCache);
  const appliedStudioBundleUrlRef = useRef('');
  const appliedRobotRouteUrlRef = useRef('');
  const appliedRobotMissionUrlRef = useRef('');
  const robotBridgeSocketRef = useRef(null);
  const robotBridgePayloadRef = useRef('');
  const robotRoutePlaybackTimerRef = useRef(null);
  const gamepadCommandStateRef = useRef({
    moveAt: 0,
    turnAt: 0,
    buttonTimes: {},
    previousButtons: {},
    connectedId: null
  });
  const gamepadActionHandlersRef = useRef(null);
  const assetWorkspaceFileInputRef = useRef(null);
  const sceneWorkspaceFileInputRef = useRef(null);
  const semanticZoneFileInputRef = useRef(null);
  const robotRouteFileInputRef = useRef(null);
  const robotMissionDraftBundleFileInputRef = useRef(null);
  const robotMissionDraftBundleFileImportModeRef = useRef('apply');
  const studioBundleFileInputRef = useRef(null);
  const activeConfig = useMemo(
    () => resolveDreamwalkerConfig(currentFragmentId),
    [currentFragmentId]
  );
  const effectiveMapPositionState = diagnosticMapPositionOverride ?? mapPositionState;
  const dynamicMapLoadPlan = useMemo(
    () =>
      buildDynamicMapLoadPlan(activeConfig, assetWorkspaceDraft, {
        currentTileId: selectedTileId ? '' : activeDynamicMapTileId,
        maxResidentTiles: dynamicMapTileResidentLimit,
        maxTilePreloadCandidates: dynamicMapTilePreloadLimit,
        position: selectedTileId ? null : effectiveMapPositionState.position,
        routePreviewPositions: robotRoutePlaybackPreviewPositions,
        tileCatalog: tileCatalogState.catalog,
        tileId: selectedTileId
      }),
    [
      activeConfig,
      activeDynamicMapTileId,
      assetWorkspaceDraft,
      dynamicMapTileResidentLimit,
      dynamicMapTilePreloadLimit,
      effectiveMapPositionState.position,
      robotRoutePlaybackPreviewPositions,
      selectedTileId,
      tileCatalogState.catalog
    ]
  );
  const assetBundle =
    dynamicMapLoadPlan.active?.assetBundle ??
    resolveWorldAssetBundle(activeConfig, assetWorkspaceDraft);
  const nextDynamicMapCandidate = dynamicMapLoadPlan.preloadCandidates[0] ?? null;
  const activeDynamicMapTile = dynamicMapLoadPlan.activeTile;
  const hasDynamicMapTileCatalog = Boolean(dynamicMapLoadPlan.tileCatalog);
  const dynamicMapTilePreloadKey = useMemo(
    () =>
      dynamicMapLoadPlan.tilePreloadCandidates
        .map((entry) => `${entry.tileId}:${entry.splatUrl}`)
        .join('|'),
    [dynamicMapLoadPlan.tilePreloadCandidates]
  );
  const tileResidencyRows = useMemo(() => {
    const catalog = dynamicMapLoadPlan.tileCatalog;
    if (!catalog) {
      return [];
    }

    const activeTileId = activeDynamicMapTile?.id ?? '';
    const preloadEntriesByTileId = new Map(
      dynamicMapLoadPlan.tilePreloadCandidates.map((entry) => [entry.tileId, entry])
    );
    const evictionEntriesByTileId = new Map(
      dynamicMapLoadPlan.tileEvictionCandidates.map((entry) => [entry.tileId, entry])
    );
    const preloadResultsByTileId = new Map(
      tilePreloadState.entries.map((entry, index) => [
        entry.tileId,
        tilePreloadState.results[index] ?? null
      ])
    );
    const cache = tilePreloadResultCacheRef.current;

    return catalog.tiles.map((tile) => {
      const preloadEntry = preloadEntriesByTileId.get(tile.id);
      const evictionEntry = evictionEntriesByTileId.get(tile.id);
      const preloadResult = preloadResultsByTileId.get(tile.id);
      const cacheKey = buildDynamicMapTilePreloadCacheKey({
        catalogUrl: tileCatalogState.url,
        mode: dynamicMapTilePreloadMode,
        preloadCacheKey: `${catalog.sceneId}:${tile.id}:${tile.splatUrl}`,
        sceneId: catalog.sceneId,
        splatUrl: tile.splatUrl,
        tileId: tile.id
      });
      const isCached = Boolean(preloadResult?.cached || cache.get(cacheKey)?.result);
      let residency = 'ready';

      if (tile.status === 'missing-splat' || !tile.splatUrl) {
        residency = 'missing';
      } else if (tile.id === activeTileId) {
        residency = 'active';
      } else if (preloadEntry) {
        residency = isCached ? 'preload cached' : 'preload';
      } else if (isCached) {
        residency = evictionEntry ? 'evicted cached' : 'cached';
      } else if (evictionEntry) {
        residency = 'evicted';
      }

      return {
        id: tile.id,
        evictionRole: evictionEntry?.role ?? '',
        isResident: tile.id === activeTileId || Boolean(preloadEntry),
        preloadRole: preloadEntry?.role ?? '',
        residency,
        tileIndex: Object.entries(tile.tileIndex ?? {})
          .map(([axis, value]) => `${axis}${value}`)
          .join(' ')
      };
    });
  }, [
    activeDynamicMapTile?.id,
    dynamicMapLoadPlan.tileCatalog,
    dynamicMapLoadPlan.tileEvictionCandidates,
    dynamicMapLoadPlan.tilePreloadCandidates,
    dynamicMapTilePreloadMode,
    tileCatalogState.url,
    tilePreloadState.entries,
    tilePreloadState.results
  ]);
  const tileResidencySummary = useMemo(() => {
    if (tileResidencyRows.length === 0) {
      return '';
    }

    const counts = tileResidencyRows.reduce(
      (current, row) => ({
        ...current,
        [row.residency]: (current[row.residency] ?? 0) + 1
      }),
      {}
    );

    return [
      `active ${counts.active ?? 0}`,
      `preload ${(counts.preload ?? 0) + (counts['preload cached'] ?? 0)}`,
      `cached ${(counts.cached ?? 0) + (counts['preload cached'] ?? 0) + (counts['evicted cached'] ?? 0)}`,
      `evicted ${(counts.evicted ?? 0) + (counts['evicted cached'] ?? 0)}`,
      `ready ${counts.ready ?? 0}`,
      `missing ${counts.missing ?? 0}`
    ].join(' / ');
  }, [tileResidencyRows]);
  const tileResidencyPreview = useMemo(() => {
    if (tileResidencyRows.length === 0) {
      return '';
    }

    const visibleRows = tileResidencyRows.slice(0, 12);
    const suffix = tileResidencyRows.length > visibleRows.length ? ' / ...' : '';
    return (
      visibleRows
        .map((row) => `${row.id}:${row.residency}${row.tileIndex ? ` (${row.tileIndex})` : ''}`)
        .join(' / ') + suffix
    );
  }, [tileResidencyRows]);
  const dynamicMapDiagnosticsSnapshot = useMemo(() => {
    const tileCatalog = dynamicMapLoadPlan.tileCatalog
      ? {
          sceneId: dynamicMapLoadPlan.tileCatalog.sceneId,
          label: dynamicMapLoadPlan.tileCatalog.label,
          summary: { ...(dynamicMapLoadPlan.tileCatalog.summary ?? {}) }
        }
      : null;

    return {
      tileLoadStrategy: dynamicMapLoadPlan.strategy,
      sourceLabel: dynamicMapLoadPlan.sourceLabel,
      activeTile: serializeDynamicMapDiagnosticsTile(activeDynamicMapTile),
      activeTileId: activeDynamicMapTile?.id ?? '',
      tileCatalog,
      tileResidentLimit: dynamicMapLoadPlan.maxResidentTiles,
      tileResidentCandidates: dynamicMapLoadPlan.tileResidentCandidates.map(
        serializeDynamicMapDiagnosticsEntry
      ),
      tileEvictionCandidates: dynamicMapLoadPlan.tileEvictionCandidates.map(
        serializeDynamicMapDiagnosticsEntry
      ),
      tilePreloadCandidates: dynamicMapLoadPlan.tilePreloadCandidates.map(
        serializeDynamicMapDiagnosticsEntry
      ),
      tileResidencyRows,
      tileResidencySummary,
      tileResidencyPreview,
      routePreviewPositions: robotRoutePlaybackPreviewPositions
        .map(serializeDiagnosticMapPosition)
        .filter(Boolean)
    };
  }, [
    activeDynamicMapTile,
    dynamicMapLoadPlan.sourceLabel,
    dynamicMapLoadPlan.strategy,
    dynamicMapLoadPlan.tileCatalog,
    dynamicMapLoadPlan.tileEvictionCandidates,
    dynamicMapLoadPlan.tileResidentCandidates,
    dynamicMapLoadPlan.maxResidentTiles,
    dynamicMapLoadPlan.tilePreloadCandidates,
    robotRoutePlaybackPreviewPositions,
    tileResidencyPreview,
    tileResidencyRows,
    tileResidencySummary
  ]);
  useEffect(() => {
    if (!dynamicMapDiagnosticsEnabled || typeof window === 'undefined') {
      return;
    }

    window.__dreamwalkerDynamicMapDiagnostics = {
      ...(window.__dreamwalkerDynamicMapDiagnostics ?? {}),
      ...dynamicMapDiagnosticsSnapshot,
      updatedAt: Date.now()
    };
  }, [dynamicMapDiagnosticsEnabled, dynamicMapDiagnosticsSnapshot]);
  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }

    if (!dynamicMapDiagnosticsEnabled) {
      return undefined;
    }

    const setMapPosition = (positionLike, source = 'diagnostics') => {
      const x = Number(positionLike?.x ?? positionLike?.[0]);
      const y = Number(positionLike?.y ?? positionLike?.[1] ?? 0);
      const z = Number(positionLike?.z ?? positionLike?.[2]);

      if (![x, y, z].every(Number.isFinite)) {
        return false;
      }

      setDiagnosticMapPositionOverride({
        status: 'ready',
        source,
        position: {
          x: Number(x.toFixed(2)),
          y: Number(y.toFixed(2)),
          z: Number(z.toFixed(2))
        },
        timestamp: Date.now()
      });
      return true;
    };
    const clearMapPositionOverride = () => {
      setDiagnosticMapPositionOverride(null);
      return true;
    };

    window.__dreamwalkerDynamicMapDiagnostics = {
      ...(window.__dreamwalkerDynamicMapDiagnostics ?? {}),
      clearMapPositionOverride,
      setMapPosition
    };

    return () => {
      const diagnostics = window.__dreamwalkerDynamicMapDiagnostics;
      if (!diagnostics || diagnostics.setMapPosition !== setMapPosition) {
        return;
      }

      const { clearMapPositionOverride: _clearMapPositionOverride, setMapPosition: _setMapPosition, ...rest } = diagnostics;
      window.__dreamwalkerDynamicMapDiagnostics = rest;
    };
  }, []);
  const activeWorldConfig = useMemo(
    () => ({
      ...activeConfig,
      colliderMeshUrl: assetBundle.colliderMeshUrl
    }),
    [activeConfig, assetBundle.colliderMeshUrl]
  );
  const shardStorageKey = useMemo(
    () => buildShardStorageKey(activeConfig.fragmentId),
    [activeConfig.fragmentId]
  );
  const currentSceneWorkspaceFragment =
    sceneWorkspaceDraft.fragments?.[activeConfig.fragmentId] ??
    defaultSceneWorkspaceTemplate.fragments[activeConfig.fragmentId];
  const resolvedStreamScenes =
    currentSceneWorkspaceFragment.streamScenes ?? activeConfig.streamScenes;
  const activeRobotMissionZoneMapUrl =
    robotMissionState.status === 'loaded'
      ? robotMissionState.mission?.zoneMapUrl?.trim() ?? ''
      : '';
  const effectiveSemanticZoneMapUrl =
    activeRobotMissionZoneMapUrl || activeConfig.robotics.semanticZoneMapUrl?.trim() || '';
  const currentSemanticZoneWorkspace =
    semanticZoneWorkspaceDrafts[activeConfig.fragmentId] ?? null;
  const currentSemanticZoneSourcePayload = useMemo(
    () => serializeSemanticZoneMap(semanticZoneState.zoneMap),
    [semanticZoneState.zoneMap]
  );
  const currentSemanticZonePayload =
    currentSemanticZoneWorkspace ??
    currentSemanticZoneSourcePayload ??
    buildFallbackSemanticZonePayload(activeConfig);
  const effectiveSemanticZoneMap = useMemo(
    () => (currentSemanticZonePayload ? buildSemanticZoneMap(currentSemanticZonePayload) : null),
    [currentSemanticZonePayload]
  );
  const currentRobotWorldContext = useMemo(
    () =>
      normalizeRobotRouteWorldContext({
        fragmentId: activeConfig.fragmentId,
        fragmentLabel: activeConfig.fragmentLabel,
        assetLabel: assetBundle.assetLabel,
        manifestLabel: assetBundle.manifestLabel,
        splatUrl: assetBundle.splatUrl,
        colliderMeshUrl: assetBundle.colliderMeshUrl,
        frameId: effectiveSemanticZoneMap?.frameId ?? 'dreamwalker_map',
        zoneMapUrl: effectiveSemanticZoneMapUrl,
        usesDemoFallback: assetBundle.usesDemoFallback
      }),
    [
      activeConfig.fragmentId,
      activeConfig.fragmentLabel,
      assetBundle.assetLabel,
      assetBundle.colliderMeshUrl,
      assetBundle.manifestLabel,
      assetBundle.splatUrl,
      assetBundle.usesDemoFallback,
      effectiveSemanticZoneMapUrl,
      effectiveSemanticZoneMap?.frameId
    ]
  );

  const currentPreset =
    activeConfig.cameraPresets.find(
      (preset) => preset.id === selectedPresetId
    ) ?? activeConfig.cameraPresets[0];

  const selectedRatio =
    activeConfig.photoRatios.find((ratio) => ratio.id === selectedRatioId) ??
    activeConfig.photoRatios[0];

  const selectedFilter =
    activeConfig.dreamFilters.find(
      (filter) => filter.id === selectedFilterId
    ) ?? activeConfig.dreamFilters[0];
  const selectedOverlayPreset =
    activeConfig.overlayPresets.find((preset) => preset.id === selectedOverlayPresetId) ??
    activeConfig.overlayPresets[0] ??
    dreamwalkerConfig.overlayPresets[0];
  const selectedStreamScene =
    resolvedStreamScenes.find((scene) => scene.id === selectedStreamSceneId) ??
    resolvedStreamScenes[0] ??
    null;
  const effectiveSplatUrl = assetBundle.splatUrl;
  const isUsingDemoSplat = assetBundle.usesDemoFallback;
  const dynamicMapRuntimeHealth = useMemo(() => {
    if (mapRuntimeState.status === 'ready') {
      return {
        status: 'ready',
        label: 'Runtime Ready',
        detail: 'active fragment の splat を表示中です。'
      };
    }

    if (mapRuntimeState.status === 'loading') {
      return {
        status: 'neutral',
        label: 'Runtime Loading',
        detail: 'active fragment の splat を読み込み中です。'
      };
    }

    if (mapRuntimeState.status === 'missing') {
      return {
        status: 'error',
        label: 'Runtime Missing',
        detail: 'active fragment に splat URL がありません。'
      };
    }

    return {
      status: 'neutral',
      label: 'Runtime Idle',
      detail: 'map runtime はまだ起動前です。'
    };
  }, [mapRuntimeState.status]);
  const assetManifestStatusLabel =
    assetManifestState.status === 'loaded'
      ? 'Loaded'
      : assetManifestState.status === 'loading'
        ? 'Loading'
        : assetManifestState.status === 'missing'
          ? 'Optional / Missing'
          : assetManifestState.status === 'error'
            ? 'Error'
            : 'Disabled';
  const tileCatalogStatusLabel =
    tileCatalogState.status === 'loaded'
      ? 'Loaded'
      : tileCatalogState.status === 'loading'
        ? 'Loading'
        : tileCatalogState.status === 'missing'
          ? 'Optional / Missing'
          : tileCatalogState.status === 'error'
            ? 'Error'
            : 'Disabled';
  const tileCatalogHealth = useMemo(() => {
    if (tileCatalogState.status === 'loaded') {
      if ((dynamicMapLoadPlan.tileCatalog?.summary.readyTileCount ?? 0) > 0) {
        return {
          status: 'ready',
          label: 'Tile Ready',
          detail: 'large-scale 3DGS tile catalog を active map に接続しています。'
        };
      }

      return {
        status: 'warning',
        label: 'No Ready Tile',
        detail: 'tile catalog に読み込める splat tile がありません。'
      };
    }

    if (tileCatalogState.status === 'loading') {
      return {
        status: 'neutral',
        label: 'Tile Loading',
        detail: 'large-scale 3DGS tile catalog を読み込み中です。'
      };
    }

    if (tileCatalogState.status === 'error') {
      return {
        status: 'error',
        label: 'Tile Error',
        detail: tileCatalogState.error ?? 'tile catalog の読み込みに失敗しました。'
      };
    }

    return {
      status: 'neutral',
      label: 'Tile Optional',
      detail: 'tile catalog は未指定です。'
    };
  }, [
    dynamicMapLoadPlan.tileCatalog,
    tileCatalogState.error,
    tileCatalogState.status
  ]);
  const tilePreloadHealth = useMemo(() => {
    if (tilePreloadState.status === 'ready') {
      return {
        status: 'ready',
        label: 'Preload Ready',
        detail: `${tilePreloadState.results.length} tile preload checks passed.`
      };
    }

    if (tilePreloadState.status === 'loading') {
      return {
        status: 'neutral',
        label: 'Preloading',
        detail: 'next tile metadata を確認中です。'
      };
    }

    if (tilePreloadState.status === 'warning') {
      return {
        status: 'warning',
        label: 'Preload Warning',
        detail: '一部のnext tile preloadに失敗しました。'
      };
    }

    if (tilePreloadState.status === 'error') {
      return {
        status: 'error',
        label: 'Preload Error',
        detail: tilePreloadState.error ?? 'tile preload に失敗しました。'
      };
    }

    if (tilePreloadState.status === 'skipped') {
      return {
        status: 'neutral',
        label: 'Preload Off',
        detail: 'tile preload は無効です。'
      };
    }

    return {
      status: 'neutral',
      label: 'Preload Idle',
      detail: 'preload候補tileはまだありません。'
    };
  }, [
    tilePreloadState.error,
    tilePreloadState.results.length,
    tilePreloadState.status
  ]);
  const studioBundleStatusLabel =
    studioBundleState.status === 'loaded'
      ? 'Loaded'
      : studioBundleState.status === 'loading'
        ? 'Loading'
        : studioBundleState.status === 'missing'
          ? 'Optional / Missing'
          : studioBundleState.status === 'error'
            ? 'Error'
            : 'Disabled';
  const studioBundleCatalogStatusLabel =
    studioBundleCatalogState.status === 'loaded'
      ? 'Loaded'
      : studioBundleCatalogState.status === 'loading'
        ? 'Loading'
        : studioBundleCatalogState.status === 'missing'
          ? 'Optional / Missing'
          : studioBundleCatalogState.status === 'error'
            ? 'Error'
            : 'Disabled';
  const splatAssetSourceLabel =
    assetBundle.splatSource === 'tile-catalog'
      ? 'Tile Catalog'
      : assetBundle.splatSource === 'manifest'
      ? 'Manifest Asset'
      : assetBundle.splatSource === 'config'
        ? 'Config Asset'
        : assetBundle.splatSource === 'demo'
          ? 'Demo Fallback'
          : 'Missing';
  const colliderAssetSourceLabel =
    assetBundle.colliderSource === 'manifest'
      ? 'Manifest GLB'
      : assetBundle.colliderSource === 'config'
        ? 'Config GLB'
        : 'Proxy Floor';
  const semanticZoneStatusLabel =
    semanticZoneState.status === 'ready'
      ? 'Loaded'
      : semanticZoneState.status === 'loading'
        ? 'Loading'
        : semanticZoneState.status === 'error'
          ? 'Error'
          : semanticZoneState.status === 'disabled'
            ? 'Disabled'
            : 'Idle';

  useEffect(() => {
    setMapRuntimeState({
      status: effectiveSplatUrl ? 'loading' : 'missing',
      url: effectiveSplatUrl,
      fragmentId: activeConfig.fragmentId
    });
  }, [activeConfig.fragmentId, effectiveSplatUrl]);

  useEffect(() => {
    const zoneMapUrl = effectiveSemanticZoneMapUrl;

    if (!zoneMapUrl) {
      setSemanticZoneState({
        status: 'disabled',
        url: '',
        zoneMap: null,
        error: null
      });
      return;
    }

    const abortController = new AbortController();

    setSemanticZoneState({
      status: 'loading',
      url: zoneMapUrl,
      zoneMap: null,
      error: null
    });

    async function loadSemanticZoneMap() {
      try {
        const response = await fetch(zoneMapUrl, {
          signal: abortController.signal
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        const zoneMap = buildSemanticZoneMap(payload);
        setSemanticZoneState({
          status: 'ready',
          url: zoneMapUrl,
          zoneMap,
          error: null
        });
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }

        setSemanticZoneState({
          status: 'error',
          url: zoneMapUrl,
          zoneMap: null,
          error: error instanceof Error ? error.message : 'semantic zone map load failed'
        });
      }
    }

    loadSemanticZoneMap();

    return () => {
      abortController.abort();
    };
  }, [effectiveSemanticZoneMapUrl]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    if (!selectedOverlayPreset?.id) {
      return;
    }

    window.localStorage.setItem(overlayPresetStorageKey, selectedOverlayPreset.id);
  }, [selectedOverlayPreset]);

  useEffect(() => {
    if (
      activeConfig.overlayPresets.length > 0 &&
      !activeConfig.overlayPresets.some((preset) => preset.id === selectedOverlayPresetId)
    ) {
      setSelectedOverlayPresetId(activeConfig.overlayPresets[0].id);
    }
  }, [activeConfig.overlayPresets, selectedOverlayPresetId]);
  useEffect(() => {
    if (
      resolvedStreamScenes.length > 0 &&
      !resolvedStreamScenes.some((scene) => scene.id === selectedStreamSceneId)
    ) {
      setSelectedStreamSceneId(resolvedStreamScenes[0].id);
    }
  }, [resolvedStreamScenes, selectedStreamSceneId]);
  const currentAssetWorkspaceFragment =
    assetWorkspaceDraft.fragments?.[activeConfig.fragmentId] ??
    defaultAssetManifestTemplate.fragments[activeConfig.fragmentId];
  const assetWorkspaceJson = useMemo(
    () => JSON.stringify(assetWorkspaceDraft, null, 2),
    [assetWorkspaceDraft]
  );
  const sceneWorkspaceJson = useMemo(
    () => JSON.stringify(sceneWorkspaceDraft, null, 2),
    [sceneWorkspaceDraft]
  );
  const semanticZoneWorkspaceJson = useMemo(
    () => JSON.stringify(currentSemanticZonePayload ?? { zones: [] }, null, 2),
    [currentSemanticZonePayload]
  );
  const studioBundleJson = useMemo(
    () =>
      JSON.stringify(
        normalizeStudioBundle({
          label: 'Current DreamWalker Studio Bundle',
          assetWorkspace: assetWorkspaceDraft,
          sceneWorkspace: sceneWorkspaceDraft,
          semanticZoneWorkspace: semanticZoneWorkspaceDrafts,
          robotRoute: {
            version: 1,
            protocol: robotRouteProtocolId,
            label: `${activeConfig.fragmentLabel} Route Snapshot`,
            fragmentId: activeConfig.fragmentId,
            fragmentLabel: activeConfig.fragmentLabel,
            frameId: currentSemanticZonePayload?.frameId ?? 'dreamwalker_map',
            pose: robotPose,
            waypoint: robotWaypoint,
            route: robotTrail
          },
          state: {
            fragmentId: activeConfig.fragmentId,
            streamSceneId: selectedStreamScene?.id ?? null,
            overlayPresetId: selectedOverlayPreset.id,
            filterId: selectedFilter.id,
            ratioId: selectedRatio.id,
            cameraPresetId: currentPreset.id
          }
        }),
        null,
        2
      ),
    [
      activeConfig.fragmentId,
      activeConfig.fragmentLabel,
      assetWorkspaceDraft,
      currentPreset.id,
      currentSemanticZonePayload?.frameId,
      robotPose,
      robotTrail,
      robotWaypoint,
      semanticZoneWorkspaceDrafts,
      sceneWorkspaceDraft,
      selectedFilter.id,
      selectedOverlayPreset.id,
      selectedRatio.id,
      selectedStreamScene?.id
    ]
  );
  const isAssetWorkspaceDirty =
    JSON.stringify(assetWorkspaceDraft) !== assetWorkspaceBaselineJson;
  const isSceneWorkspaceDirty =
    JSON.stringify(sceneWorkspaceDraft) !== sceneWorkspaceBaselineJson;
  const isSemanticZoneWorkspaceDirty =
    JSON.stringify(semanticZoneWorkspaceDrafts) !== semanticZoneWorkspaceBaselineJson;
  const hasCurrentSemanticZoneWorkspace = Boolean(currentSemanticZoneWorkspace);
  const assetWorkspaceModeLabel = hasSavedAssetWorkspace
    ? 'Saved Workspace'
    : assetManifestState.status === 'loaded'
      ? 'Manifest Linked'
      : 'Template Only';
  const sceneWorkspaceModeLabel = hasSavedSceneWorkspace
    ? 'Saved Scene Workspace'
    : 'Template Scene Defaults';
  const semanticZoneWorkspaceModeLabel = hasCurrentSemanticZoneWorkspace
    ? 'Saved Zone Workspace'
    : activeRobotMissionZoneMapUrl
      ? 'Mission Zone Map'
      : semanticZoneState.status === 'ready'
        ? 'Config Zone Map'
        : 'Fallback Zone Draft';
  const publicStudioBundleCatalog = useMemo(
    () =>
      studioBundleCatalogState.catalog
        ? normalizeStudioBundleCatalog(studioBundleCatalogState.catalog)
        : null,
    [studioBundleCatalogState.catalog]
  );
  const publicRobotRouteCatalog = useMemo(
    () =>
      robotRouteCatalogState.catalog
        ? normalizeRobotRouteCatalog(robotRouteCatalogState.catalog)
        : null,
    [robotRouteCatalogState.catalog]
  );
  const publicRobotMissionCatalog = useMemo(
    () =>
      robotMissionCatalogState.catalog
        ? normalizeRobotMissionCatalog(robotMissionCatalogState.catalog)
        : null,
    [robotMissionCatalogState.catalog]
  );
  const robotRouteCatalogStatusLabel =
    robotRouteCatalogState.status === 'loaded'
      ? 'Catalog Loaded'
      : robotRouteCatalogState.status === 'loading'
        ? 'Catalog Loading'
        : robotRouteCatalogState.status === 'missing'
          ? 'Catalog Missing'
          : robotRouteCatalogState.status === 'error'
            ? 'Catalog Error'
            : 'Catalog Disabled';
  const robotRouteStatusLabel =
    robotRouteState.status === 'loaded'
      ? 'Route Loaded'
      : robotRouteState.status === 'loading'
        ? 'Route Loading'
        : robotRouteState.status === 'missing'
          ? 'Route Missing'
          : robotRouteState.status === 'error'
            ? 'Route Error'
            : 'Route Disabled';
  const robotMissionCatalogStatusLabel =
    robotMissionCatalogState.status === 'loaded'
      ? 'Catalog Loaded'
      : robotMissionCatalogState.status === 'loading'
        ? 'Catalog Loading'
        : robotMissionCatalogState.status === 'missing'
          ? 'Catalog Missing'
          : robotMissionCatalogState.status === 'error'
            ? 'Catalog Error'
            : 'Catalog Disabled';
  const robotMissionStatusLabel =
    robotMissionState.status === 'loaded'
      ? 'Mission Loaded'
      : robotMissionState.status === 'loading'
        ? 'Mission Loading'
        : robotMissionState.status === 'missing'
          ? 'Mission Missing'
          : robotMissionState.status === 'error'
            ? 'Mission Error'
            : 'Mission Disabled';
  const activeRobotRouteHealth = useMemo(
    () =>
      robotRouteState.status === 'loaded' && robotRouteState.route
        ? buildRobotRouteCompatibility(robotRouteState.route, currentRobotWorldContext)
        : null,
    [currentRobotWorldContext, robotRouteState.route, robotRouteState.status]
  );
  const activeRobotMissionHealth = useMemo(
    () =>
      robotMissionState.status === 'loaded' && robotMissionState.mission && robotMissionState.route
        ? buildRobotMissionCompatibility(
            robotMissionState.mission,
            robotMissionState.route,
            currentRobotWorldContext
          )
        : null,
    [
      currentRobotWorldContext,
      robotMissionState.mission,
      robotMissionState.route,
      robotMissionState.status
    ]
  );

  useEffect(() => {
    let disposed = false;
    const hasLocalSplat = Boolean(normalizeLocalAssetPath(assetBundle.splatUrl));
    const hasLocalCollider = Boolean(normalizeLocalAssetPath(assetBundle.colliderMeshUrl));

    if (!hasLocalSplat && !hasLocalCollider) {
      setActiveWorldHealth(buildWorldAssetHealth(assetBundle));
      return;
    }

    setActiveWorldHealth({
      status: 'loading',
      label: 'Checking',
      detail: 'local splat / collider file を確認中です。'
    });

    async function loadCurrentWorldHealth() {
      const resolvedHealth = buildWorldAssetHealth(assetBundle, {
        splatExists: await probeLocalAssetAvailability(assetBundle.splatUrl),
        colliderExists: await probeLocalAssetAvailability(assetBundle.colliderMeshUrl)
      });

      if (!disposed) {
        setActiveWorldHealth(resolvedHealth);
      }
    }

    loadCurrentWorldHealth();

    return () => {
      disposed = true;
    };
  }, [
    assetBundle.colliderMeshUrl,
    assetBundle.hasColliderMesh,
    assetBundle.hasConfiguredSplat,
    assetBundle.splatUrl,
    assetBundle.usesDemoFallback
  ]);

  useEffect(() => {
    if (!publicStudioBundleCatalog?.bundles?.length) {
      setPublicStudioBundleHealthMap({});
      return;
    }

    let disposed = false;

    setPublicStudioBundleHealthMap(
      Object.fromEntries(
        publicStudioBundleCatalog.bundles.map((entry) => [
          entry.id,
          {
            status: 'loading',
            label: 'Checking',
            detail: 'bundle file と world asset を確認中です。'
          }
        ])
      )
    );

    async function loadBundleHealth() {
      const nextEntries = await Promise.all(
        publicStudioBundleCatalog.bundles.map(async (entry) => {
          if (!entry.url) {
            return [
              entry.id,
              {
                status: 'error',
                label: 'Missing URL',
                detail: 'bundle URL が空です。'
              }
            ];
          }

          try {
            const response = await fetch(entry.url, {
              cache: 'no-store',
              headers: {
                Accept: 'application/json'
              }
            });

            if (!response.ok) {
              throw new Error(`HTTP ${response.status}`);
            }

            const bundle = normalizeStudioBundle(await response.json());
            const initialHealth = resolveBundleWorldHealth(bundle, entry);
            const resolvedHealth = resolveBundleWorldHealth(bundle, entry, {
              splatExists: await probeLocalAssetAvailability(
                initialHealth.assetBundle.splatUrl
              ),
              colliderExists: await probeLocalAssetAvailability(
                initialHealth.assetBundle.colliderMeshUrl
              )
            });

            return [
              entry.id,
              {
                status: resolvedHealth.status,
                label: resolvedHealth.label,
                detail: resolvedHealth.detail,
                fragmentId: resolvedHealth.fragmentId
              }
            ];
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);

            return [
              entry.id,
              {
                status: 'error',
                label: 'Bundle Missing',
                detail: `bundle file を読めません: ${message}`
              }
            ];
          }
        })
      );

      if (!disposed) {
        setPublicStudioBundleHealthMap(Object.fromEntries(nextEntries));
      }
    }

    loadBundleHealth();

    return () => {
      disposed = true;
    };
  }, [publicStudioBundleCatalog]);

  useEffect(() => {
    if (!publicRobotRouteCatalog?.routes?.length) {
      setPublicRobotRouteHealthMap({});
      return;
    }

    let disposed = false;

    setPublicRobotRouteHealthMap(
      Object.fromEntries(
        publicRobotRouteCatalog.routes.map((entry) => [
          entry.id,
          {
            status: 'loading',
            label: 'Checking',
            detail: 'route preset と world metadata を確認中です。'
          }
        ])
      )
    );

    async function loadRouteHealth() {
      const nextEntries = await Promise.all(
        publicRobotRouteCatalog.routes.map(async (entry) => {
          if (!entry.url) {
            return [
              entry.id,
              {
                status: 'error',
                label: 'Missing URL',
                detail: 'route URL が空です。'
              }
            ];
          }

          try {
            const response = await fetch(entry.url, {
              cache: 'no-store',
              headers: {
                Accept: 'application/json'
              }
            });

            if (!response.ok) {
              throw new Error(`HTTP ${response.status}`);
            }

            const route = normalizeRobotRoutePayload(await response.json());
            return [entry.id, buildRobotRouteCompatibility(route, currentRobotWorldContext)];
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);

            return [
              entry.id,
              {
                status: 'error',
                label: 'Route Missing',
                detail: `route file を読めません: ${message}`
              }
            ];
          }
        })
      );

      if (!disposed) {
        setPublicRobotRouteHealthMap(Object.fromEntries(nextEntries));
      }
    }

    loadRouteHealth();

    return () => {
      disposed = true;
    };
  }, [currentRobotWorldContext, publicRobotRouteCatalog]);

  useEffect(() => {
    if (!publicRobotMissionCatalog?.missions?.length) {
      setPublicRobotMissionHealthMap({});
      return;
    }

    let disposed = false;

    setPublicRobotMissionHealthMap(
      Object.fromEntries(
        publicRobotMissionCatalog.missions.map((entry) => [
          entry.id,
          {
            status: 'loading',
            label: 'Checking',
            detail: 'mission manifest と route/world metadata を確認中です。'
          }
        ])
      )
    );

    async function loadMissionHealth() {
      const nextEntries = await Promise.all(
        publicRobotMissionCatalog.missions.map(async (entry) => {
          if (!entry.url) {
            return [
              entry.id,
              {
                status: 'error',
                label: 'Missing URL',
                detail: 'mission URL が空です。'
              }
            ];
          }

          try {
            const { mission, route } = await loadRobotMissionResource(entry.url);
            return [
              entry.id,
              buildRobotMissionCompatibility(mission, route, currentRobotWorldContext)
            ];
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);

            return [
              entry.id,
              {
                status: 'error',
                label: 'Mission Missing',
                detail: `mission file を読めません: ${message}`
              }
            ];
          }
        })
      );

      if (!disposed) {
        setPublicRobotMissionHealthMap(Object.fromEntries(nextEntries));
      }
    }

    loadMissionHealth();

    return () => {
      disposed = true;
    };
  }, [currentRobotWorldContext, publicRobotMissionCatalog]);

  const totalShardCount = activeConfig.shards.length;
  const collectedShardCount = collectedShardIds.length;
  const remainingShardCount = Math.max(0, totalShardCount - collectedShardCount);
  const isGateUnlocked = remainingShardCount === 0;

  const availableShards = useMemo(
    () =>
      activeConfig.shards.filter(
        (shard) => !collectedShardIds.includes(shard.id)
      ),
    [activeConfig.shards, collectedShardIds]
  );

  const gateCard = useMemo(
    () => ({
      id: activeConfig.gate.id,
      kind: activeConfig.gate.kind,
      label: isGateUnlocked ? 'Gate Open' : `Gate ${remainingShardCount}`,
      title: isGateUnlocked
        ? activeConfig.gate.openTitle
        : activeConfig.gate.lockedTitle,
      body: isGateUnlocked
        ? activeConfig.gate.openBody
        : `${activeConfig.gate.lockedBody}\n\nあと ${remainingShardCount} 個必要。`,
      position: activeConfig.gate.position,
      presetId: activeConfig.gate.presetId,
      targetFragmentId: activeConfig.gate.targetFragmentId,
      targetFragmentLabel: activeConfig.gate.targetFragmentLabel,
      accentColor: isGateUnlocked
        ? activeConfig.gate.openAccentColor
        : activeConfig.gate.lockedAccentColor,
      isGateUnlocked
    }),
    [activeConfig.gate, isGateUnlocked, remainingShardCount]
  );

  const visibleHotspots = useMemo(
    () =>
      projectedHotspots
        .map((projectedHotspot) => {
          const hotspot = activeConfig.hotspots.find(
            (candidate) => candidate.id === projectedHotspot.id
          );

          if (!hotspot) {
            return null;
          }

          return {
            ...hotspot,
            xPercent: projectedHotspot.xPercent,
            yPercent: projectedHotspot.yPercent,
            zDepth: projectedHotspot.zDepth
          };
        })
        .filter(Boolean),
    [activeConfig.hotspots, projectedHotspots]
  );

  const loopItems = useMemo(
    () => [...availableShards, gateCard],
    [availableShards, gateCard]
  );

  const visibleLoopItems = useMemo(
    () =>
      projectedLoopItems
        .map((projectedItem) => {
          const item = loopItems.find((candidate) => candidate.id === projectedItem.id);

          if (!item) {
            return null;
          }

          return {
            ...item,
            xPercent: projectedItem.xPercent,
            yPercent: projectedItem.yPercent,
            zDepth: projectedItem.zDepth
          };
        })
        .filter(Boolean),
    [loopItems, projectedLoopItems]
  );
  const robotProjectionPoints = useMemo(
    () => buildRobotProjectionPoints(robotPose, robotWaypoint, activeConfig.robotics),
    [activeConfig.robotics, robotPose, robotWaypoint]
  );
  const robotRouteProjectionPoints = useMemo(
    () => buildRobotRouteProjectionPoints(robotTrail, robotWaypoint, activeConfig.robotics),
    [activeConfig.robotics, robotTrail, robotWaypoint]
  );
  const benchmarkRouteProjectionPoints = useMemo(
    () => buildBenchmarkRouteProjectionPoints(sim2realBenchmarkOverlay, activeConfig.robotics),
    [activeConfig.robotics, sim2realBenchmarkOverlay]
  );
  const semanticZoneHits = useMemo(
    () =>
      findSemanticZoneHits(
        effectiveSemanticZoneMap,
        robotPose.position[0],
        robotPose.position[2]
      ),
    [effectiveSemanticZoneMap, robotPose.position]
  );
  const semanticZoneSummary = useMemo(
    () => summarizeSemanticZoneHits(semanticZoneHits),
    [semanticZoneHits]
  );
  const semanticZoneProjectionPoints = useMemo(
    () =>
      buildSemanticZoneProjectionPoints(
        effectiveSemanticZoneMap,
        semanticZoneHits.map((zone) => zone.id),
        activeConfig.robotics.zoneAnchorHeight
      ),
    [activeConfig.robotics.zoneAnchorHeight, effectiveSemanticZoneMap, semanticZoneHits]
  );
  const semanticZoneSurfacePoints = useMemo(
    () => buildSemanticZoneSurfacePoints(effectiveSemanticZoneMap, 0.05),
    [effectiveSemanticZoneMap]
  );
  const visibleRobotPoints = useMemo(
    () =>
      projectedRobotPoints
        .map((projectedPoint) => {
          const point = robotProjectionPoints.find((candidate) => candidate.id === projectedPoint.id);

          if (!point) {
            return null;
          }

          return {
            ...point,
            xPercent: projectedPoint.xPercent,
            yPercent: projectedPoint.yPercent,
            zDepth: projectedPoint.zDepth
          };
        })
        .filter(Boolean),
    [projectedRobotPoints, robotProjectionPoints]
  );
  const visibleRobotRoutePoints = useMemo(
    () =>
      projectedRobotRoutePoints
        .map((projectedPoint) => {
          const point = robotRouteProjectionPoints.find((candidate) => candidate.id === projectedPoint.id);

          if (!point) {
            return null;
          }

          return {
            ...point,
            xPercent: projectedPoint.xPercent,
            yPercent: projectedPoint.yPercent,
            zDepth: projectedPoint.zDepth
          };
        })
        .filter(Boolean),
    [projectedRobotRoutePoints, robotRouteProjectionPoints]
  );
  const visibleBenchmarkRoutePoints = useMemo(
    () =>
      projectedBenchmarkRoutePoints
        .map((projectedPoint) => {
          const point = benchmarkRouteProjectionPoints.find(
            (candidate) => candidate.id === projectedPoint.id
          );

          if (!point) {
            return null;
          }

          return {
            ...point,
            xPercent: projectedPoint.xPercent,
            yPercent: projectedPoint.yPercent,
            zDepth: projectedPoint.zDepth
          };
        })
        .filter(Boolean),
    [benchmarkRouteProjectionPoints, projectedBenchmarkRoutePoints]
  );
  const visibleSemanticZonePoints = useMemo(
    () =>
      projectedSemanticZonePoints
        .map((projectedPoint) => {
          const point = semanticZoneProjectionPoints.find(
            (candidate) => candidate.id === projectedPoint.id
          );

          if (!point) {
            return null;
          }

          return {
            ...point,
            xPercent: projectedPoint.xPercent,
            yPercent: projectedPoint.yPercent,
            zDepth: projectedPoint.zDepth
          };
        })
        .filter(Boolean),
    [projectedSemanticZonePoints, semanticZoneProjectionPoints]
  );
  const visibleSemanticZoneSurfaces = useMemo(() => {
    if (!effectiveSemanticZoneMap) {
      return [];
    }

    const projectedPointMap = new Map(
      projectedSemanticZoneSurfacePoints.map((projectedPoint) => [projectedPoint.id, projectedPoint])
    );

    return effectiveSemanticZoneMap.zones
      .map((zone) => {
        const sourcePoints = semanticZoneSurfacePoints
          .filter((point) => point.zoneId === zone.id)
          .sort((left, right) => left.order - right.order);

        if (sourcePoints.length < 3) {
          return null;
        }

        const orderedProjectedPoints = sourcePoints
          .map((point) => projectedPointMap.get(point.id))
          .filter(Boolean);

        // 部分的にしか投影できない contour は崩れやすいので描かない。
        if (orderedProjectedPoints.length !== sourcePoints.length) {
          return null;
        }

        const averageDepth =
          orderedProjectedPoints.reduce((sum, point) => sum + point.zDepth, 0) /
          orderedProjectedPoints.length;

        return {
          id: zone.id,
          accentColor: zone.accentColor,
          fillOpacity: zone.cost >= 80 ? 0.22 : zone.cost >= 45 ? 0.18 : 0.12,
          isActive: semanticZoneHits.some((hit) => hit.id === zone.id),
          averageDepth: Number(averageDepth.toFixed(3)),
          points: orderedProjectedPoints
            .map((point) => `${point.xPercent},${point.yPercent}`)
            .join(' ')
        };
      })
      .filter(Boolean);
  }, [
    effectiveSemanticZoneMap,
    projectedSemanticZoneSurfacePoints,
    semanticZoneHits,
    semanticZoneSurfacePoints
  ]);

  const reticleCandidate = useMemo(() => {
    const candidates = [...visibleHotspots, ...visibleLoopItems]
      .map((item) => ({
        ...item,
        centerDistance: Math.hypot(item.xPercent - 50, item.yPercent - 50)
      }))
      .filter(
        (item) =>
          item.centerDistance <= activeConfig.interactSettings.maxScreenDistance
      )
      .sort((left, right) => {
        if (left.centerDistance !== right.centerDistance) {
          return left.centerDistance - right.centerDistance;
        }

        return left.zDepth - right.zDepth;
      });

    return candidates[0] ?? null;
  }, [activeConfig.interactSettings.maxScreenDistance, visibleHotspots, visibleLoopItems]);
  const reticleTarget =
    reticleCandidate &&
    reticleCandidate.zDepth <= activeConfig.interactSettings.maxDepth
      ? reticleCandidate
      : null;

  const isPhotoMode = mode === 'photo';
  const isLiveMode = mode === 'live';
  const isRobotMode = mode === 'robot';
  const isWalkMode = cameraMode === 'walk';
  const guideStyle = getGuideStyle(selectedRatio);
  const selectedRobotCamera =
    activeConfig.robotics.cameras.find((camera) => camera.id === selectedRobotCameraId) ??
    activeConfig.robotics.cameras[0] ??
    null;
  const roboticsCamera = useMemo(
    () => ({
      enabled: isRobotMode && !isWalkMode,
      robotPose,
      selectedCamera: selectedRobotCamera
    }),
    [isRobotMode, isWalkMode, robotPose, selectedRobotCamera]
  );
  const resolvedLiveOverlayBranding = useMemo(
    () =>
      resolveOverlayBrandingForScene(
        activeConfig.overlayBranding,
        selectedStreamScene
      ),
    [activeConfig.overlayBranding, selectedStreamScene]
  );
  const resolvedLiveOverlayMemo = useMemo(
    () => ({
      title: selectedStreamScene?.overlayMemo?.title ?? null,
      items: normalizeOverlayMemoItems(selectedStreamScene?.overlayMemo?.items),
      footer: selectedStreamScene?.overlayMemo?.footer ?? null
    }),
    [selectedStreamScene]
  );
  const liveScenePayload = useMemo(
    () => ({
      appTitle: dreamwalkerConfig.appTitle,
      fragmentId: activeConfig.fragmentId,
      fragmentLabel: activeConfig.fragmentLabel,
      assetLabel: assetBundle.assetLabel,
      assetManifestLabel: assetManifestState.manifest?.label ?? null,
      splatSource: assetBundle.splatSource,
      colliderSource: assetBundle.colliderSource,
      overlayBrandingId: resolvedLiveOverlayBranding.id,
      overlayBrandingLabel: resolvedLiveOverlayBranding.label,
      overlayBrandingBadge: resolvedLiveOverlayBranding.badge,
      overlayBrandingStrapline: resolvedLiveOverlayBranding.strapline,
      overlayBrandingAccent: resolvedLiveOverlayBranding.accent,
      overlayBrandingHighlight: resolvedLiveOverlayBranding.highlight,
      overlayBrandingGlow: resolvedLiveOverlayBranding.glow,
      overlayPresetId: selectedOverlayPreset.id,
      overlayPresetLabel: selectedOverlayPreset.label,
      streamSceneId: selectedStreamScene?.id ?? null,
      streamSceneLabel: selectedStreamScene?.label ?? null,
      streamSceneTitle: selectedStreamScene?.title ?? null,
      streamSceneTopic: selectedStreamScene?.topic ?? null,
      overlayMemoTitle: resolvedLiveOverlayMemo.title,
      overlayMemoItems: resolvedLiveOverlayMemo.items,
      overlayMemoFooter: resolvedLiveOverlayMemo.footer,
      cameraPresetId: currentPreset.id,
      cameraPresetLabel: currentPreset.label,
      dreamFilterId: selectedFilter.id,
      dreamFilterLabel: selectedFilter.label,
      walkColliderMode: walkColliderStatus.mode,
      gateStatus: isGateUnlocked ? 'open' : 'locked',
      shardProgress: {
        collected: collectedShardCount,
        total: totalShardCount
      },
      sceneHash: `#${activeConfig.fragmentId}`
    }),
    [
      activeConfig.fragmentId,
      activeConfig.fragmentLabel,
      assetBundle.assetLabel,
      assetBundle.colliderSource,
      assetBundle.splatSource,
      assetManifestState.manifest,
      collectedShardCount,
      currentPreset.id,
      currentPreset.label,
      isGateUnlocked,
      resolvedLiveOverlayBranding.accent,
      resolvedLiveOverlayBranding.badge,
      resolvedLiveOverlayBranding.glow,
      resolvedLiveOverlayBranding.highlight,
      resolvedLiveOverlayBranding.id,
      resolvedLiveOverlayBranding.label,
      resolvedLiveOverlayBranding.strapline,
      resolvedLiveOverlayMemo.footer,
      resolvedLiveOverlayMemo.items,
      resolvedLiveOverlayMemo.title,
      selectedOverlayPreset.id,
      selectedOverlayPreset.label,
      selectedFilter.id,
      selectedFilter.label,
      selectedStreamScene,
      totalShardCount,
      walkColliderStatus.mode
    ]
  );
  const liveScenePayloadJson = useMemo(
    () => JSON.stringify(liveScenePayload, null, 2),
    [liveScenePayload]
  );
  const overlayUrl = useMemo(() => {
    if (typeof window === 'undefined') {
      return relayConfig.enabled ? '/overlay.html?relay=1' : '/overlay.html';
    }

    const url = new URL('/overlay.html', window.location.href);
    if (relayConfig.enabled) {
      url.searchParams.set('relay', '1');
      if (relayConfig.url !== overlayRelayDefaultUrl) {
        url.searchParams.set('relayUrl', relayConfig.url);
      } else {
        url.searchParams.delete('relayUrl');
      }
    } else {
      url.searchParams.delete('relay');
      url.searchParams.delete('relayUrl');
    }
    return url.toString();
  }, [relayConfig.enabled, relayConfig.url]);
  const overlayTransportLabel = relayConfig.enabled ? 'Relay SSE' : 'LocalStorage';
  const walkColliderLabel =
    walkColliderStatus.mode === 'mesh'
      ? 'GLB Mesh'
      : walkColliderStatus.mode === 'error'
        ? 'Proxy Fallback'
        : walkColliderStatus.mode === 'idle'
          ? 'On Demand'
        : walkColliderStatus.mode === 'loading'
          ? 'Loading Collider'
          : 'Proxy Floor';
  const robotPoseSummary = useMemo(
    () => ({
      x: robotPose.position[0].toFixed(2),
      y: robotPose.position[1].toFixed(2),
      z: robotPose.position[2].toFixed(2),
      yaw: Math.round(robotPose.yawDegrees)
    }),
    [robotPose]
  );
  const robotWaypointDistance = useMemo(() => {
    if (!robotWaypoint) {
      return null;
    }

    const dx = robotWaypoint.position[0] - robotPose.position[0];
    const dz = robotWaypoint.position[2] - robotPose.position[2];
    return Math.hypot(dx, dz).toFixed(2);
  }, [robotPose, robotWaypoint]);
  const robotTrailDistance = useMemo(
    () => calculateRobotTrailDistance(robotTrail).toFixed(2),
    [robotTrail]
  );
  const robotNodeLabel = useMemo(
    () => formatRobotNodeLabel(robotTrail.length),
    [robotTrail.length]
  );
  const robotRoutePayload = useMemo(
    () => ({
      version: 1,
      protocol: robotRouteProtocolId,
      label:
        robotMissionState.route?.label ||
        `${activeConfig.fragmentLabel} Route Snapshot`,
      description:
        robotMissionState.route?.description ||
        `${activeConfig.fragmentLabel} robot route snapshot`,
      accent:
        robotMissionState.route?.accent ||
        activeConfig.overlayBranding?.highlight ||
        activeConfig.overlayBranding?.accent ||
        '#85e3e1',
      fragmentId: activeConfig.fragmentId,
      fragmentLabel: activeConfig.fragmentLabel,
      frameId: effectiveSemanticZoneMap?.frameId ?? 'dreamwalker_map',
      world: currentRobotWorldContext,
      pose: robotPose,
      waypoint: robotWaypoint,
      route: robotTrail,
      routeNodeCount: robotTrail.length,
      routeDistance: Number(robotTrailDistance),
      currentZoneLabel: semanticZoneHits.length ? semanticZoneSummary.label : 'Outside Map'
    }),
    [
      activeConfig.fragmentId,
      activeConfig.fragmentLabel,
      activeConfig.overlayBranding?.accent,
      activeConfig.overlayBranding?.highlight,
      currentRobotWorldContext,
      effectiveSemanticZoneMap,
      robotPose,
      robotMissionState.route?.accent,
      robotMissionState.route?.description,
      robotMissionState.route?.label,
      robotTrail,
      robotTrailDistance,
      robotWaypoint,
      semanticZoneHits.length,
      semanticZoneSummary.label
    ]
  );
  const robotRoutePayloadJson = useMemo(
    () => JSON.stringify(robotRoutePayload, null, 2),
    [robotRoutePayload]
  );
  const robotMissionExportId = useMemo(
    () =>
      buildMissionSlug(
        robotMissionState.mission?.id ||
          robotMissionState.mission?.label ||
          `${activeConfig.fragmentId}-${selectedPresetId || 'robot-mission'}`
      ),
    [
      activeConfig.fragmentId,
      robotMissionState.mission?.id,
      robotMissionState.mission?.label,
      selectedPresetId
    ]
  );
  const robotMissionPayload = useMemo(
    () => ({
      version: 1,
      protocol: robotMissionProtocolId,
      id: robotMissionState.mission?.id || robotMissionExportId,
      label:
        robotMissionState.mission?.label ||
        `${activeConfig.fragmentLabel} Robot Mission`,
      description:
        robotMissionState.mission?.description ||
        `${activeConfig.fragmentLabel} robot mission snapshot`,
      fragmentId:
        robotMissionState.mission?.fragmentId || activeConfig.fragmentId,
      fragmentLabel:
        robotMissionState.mission?.fragmentLabel || activeConfig.fragmentLabel,
      accent:
        robotMissionState.mission?.accent ||
        activeConfig.overlayBranding?.highlight ||
        activeConfig.overlayBranding?.accent ||
        '#85e3e1',
      routeUrl:
        robotMissionState.mission?.routeUrl ||
        `/robot-routes/${robotMissionExportId}.json`,
      zoneMapUrl:
        robotMissionState.mission?.zoneMapUrl ||
        effectiveSemanticZoneMapUrl ||
        `/manifests/robotics-${activeConfig.fragmentId}.zones.json`,
      launchUrl:
        robotMissionState.mission?.launchUrl ||
        `/?robotMission=${encodeURIComponent(`/robot-missions/${robotMissionExportId}.mission.json`)}`,
      cameraPresetId: selectedPresetId || '',
      robotCameraId: selectedRobotCameraId || '',
      streamSceneId: selectedStreamScene?.id || '',
      startupMode: isRobotMode ? 'robot' : mode,
      world: {
        assetLabel:
          robotMissionState.mission?.world?.assetLabel ||
          currentRobotWorldContext.assetLabel,
        frameId:
          robotMissionState.mission?.world?.frameId ||
          currentRobotWorldContext.frameId
      }
    }),
    [
      activeConfig.fragmentId,
      activeConfig.fragmentLabel,
      activeConfig.overlayBranding?.accent,
      activeConfig.overlayBranding?.highlight,
      currentRobotWorldContext.assetLabel,
      currentRobotWorldContext.frameId,
      effectiveSemanticZoneMapUrl,
      isRobotMode,
      mode,
      robotMissionExportId,
      robotMissionState.mission?.accent,
      robotMissionState.mission?.description,
      robotMissionState.mission?.id,
      robotMissionState.mission?.label,
      robotMissionState.mission?.launchUrl,
      robotMissionState.mission?.fragmentId,
      robotMissionState.mission?.fragmentLabel,
      robotMissionState.mission?.routeUrl,
      robotMissionState.mission?.zoneMapUrl,
      robotMissionState.mission?.world?.assetLabel,
      robotMissionState.mission?.world?.frameId,
      selectedPresetId,
      selectedRobotCameraId,
      selectedStreamScene?.id
    ]
  );
  const robotMissionPayloadJson = useMemo(
    () => JSON.stringify(robotMissionPayload, null, 2),
    [robotMissionPayload]
  );
  const robotMissionDraftBundleShelfSummaryLabel = useMemo(
    () =>
      robotMissionDraftBundleShelfLabel.trim() ||
      robotMissionPayload.label ||
      `${activeConfig.fragmentLabel} Mission Draft Bundle`,
    [
      activeConfig.fragmentLabel,
      robotMissionDraftBundleShelfLabel,
      robotMissionPayload.label
    ]
  );
  const robotMissionDraftBundle = useMemo(
    () => ({
      version: 1,
      label: `${robotMissionPayload.label} Draft Bundle`,
      fragmentId: robotMissionPayload.fragmentId,
      fragmentLabel: robotMissionPayload.fragmentLabel,
      mission: robotMissionPayload,
      route: robotRoutePayload,
      zones: currentSemanticZonePayload ?? null
    }),
    [
      currentSemanticZonePayload,
      robotMissionPayload,
      robotRoutePayload
    ]
  );
  const robotMissionDraftBundleFileName = useMemo(
    () => buildRobotMissionDraftBundleFileName(robotMissionDraftBundle, activeConfig.fragmentId),
    [activeConfig.fragmentId, robotMissionDraftBundle]
  );
  const robotMissionDraftBundleJson = useMemo(
    () => JSON.stringify(robotMissionDraftBundle, null, 2),
    [robotMissionDraftBundle]
  );
  const robotMissionArtifactPackFileName = useMemo(
    () => buildRobotMissionArtifactPackFileName(robotMissionDraftBundle, activeConfig.fragmentId),
    [activeConfig.fragmentId, robotMissionDraftBundle]
  );
  const robotMissionDraftBundleImportPreview = useMemo(() => {
    const rawImportText = robotMissionDraftBundleImportText.trim();

    if (!rawImportText) {
      return null;
    }

    try {
      return tryParseRobotMissionDraftBundleImport(rawImportText);
    } catch {
      return null;
    }
  }, [robotMissionDraftBundleImportText]);
  const publishedRobotMissionPayload = useMemo(
    () => buildPublishedRobotMissionPayloadFromBundle(robotMissionDraftBundle, activeConfig),
    [activeConfig, robotMissionDraftBundle]
  );
  const publishedRobotMissionPreview = useMemo(
    () => buildPublishedRobotMissionPreviewFromBundle(robotMissionDraftBundle, activeConfig),
    [activeConfig, robotMissionDraftBundle]
  );
  const publishedRobotMissionFileName = useMemo(
    () => publishedRobotMissionPreview.fileName,
    [publishedRobotMissionPreview.fileName]
  );
  const publishedRobotMissionPayloadJson = useMemo(
    () => JSON.stringify(publishedRobotMissionPayload, null, 2),
    [publishedRobotMissionPayload]
  );
  const robotMissionPublishReportJson = useMemo(
    () =>
      JSON.stringify(
        buildRobotMissionPublishReport(robotMissionDraftBundle, activeConfig),
        null,
        2
      ),
    [activeConfig, robotMissionDraftBundle]
  );
  const robotMissionPublishReportFileName = useMemo(
    () =>
      robotMissionDraftBundleFileName.endsWith('.json')
        ? robotMissionDraftBundleFileName.replace(/\.json$/i, '.publish-report.json')
        : `${robotMissionDraftBundleFileName}.publish-report.json`,
    [robotMissionDraftBundleFileName]
  );
  const robotMissionArtifactPackJson = useMemo(
    () => buildCurrentRobotMissionDraftBundleArtifacts().artifactPackContent,
    [activeConfig.fragmentId, robotMissionDraftBundle, robotMissionPayload.label, robotMissionDraftBundleShelfSummaryLabel]
  );
  const robotMissionValidateCommand = useMemo(
    () =>
      buildRobotMissionValidateCommandFromBundle(
        robotMissionDraftBundle,
        robotMissionArtifactPackFileName,
        activeConfig
      ),
    [activeConfig, robotMissionArtifactPackFileName, robotMissionDraftBundle]
  );
  const robotMissionReleaseCommand = useMemo(
    () =>
      buildRobotMissionReleaseCommandFromBundle(
        robotMissionDraftBundle,
        robotMissionArtifactPackFileName,
        activeConfig
      ),
    [activeConfig, robotMissionArtifactPackFileName, robotMissionDraftBundle]
  );
  const robotMissionPublishCommand = useMemo(
    () =>
      buildRobotMissionPublishCommandFromBundle(
        robotMissionDraftBundle,
        robotMissionArtifactPackFileName,
        activeConfig
      ),
    [activeConfig, robotMissionArtifactPackFileName, robotMissionDraftBundle]
  );
  const robotMissionDraftBundleHealth = useMemo(
    () => buildRobotMissionPreflightHealth(robotMissionDraftBundle, activeConfig),
    [activeConfig, robotMissionDraftBundle]
  );
  const robotMissionDraftBundleShelfHealthMap = useMemo(
    () =>
      Object.fromEntries(
        robotMissionDraftBundleShelf.map((entry) => [
          entry.id,
          buildRobotMissionPreflightHealth(entry.bundle, activeConfig)
        ])
      ),
    [activeConfig, robotMissionDraftBundleShelf]
  );
  const robotRouteShelfSummaryLabel = useMemo(
    () =>
      robotRouteShelfLabel.trim() ||
      `${activeConfig.fragmentLabel} / ${robotNodeLabel}`,
    [activeConfig.fragmentLabel, robotNodeLabel, robotRouteShelfLabel]
  );
  const robotRouteShelfHealthMap = useMemo(
    () =>
      Object.fromEntries(
        robotRouteShelf.map((entry) => [
          entry.id,
          buildRobotRouteCompatibility(entry.route, currentRobotWorldContext)
        ])
      ),
    [currentRobotWorldContext, robotRouteShelf]
  );
  const semanticZoneCount = effectiveSemanticZoneMap?.zones.length ?? 0;
  const semanticZoneCurrentLabel = semanticZoneHits.length
    ? semanticZoneSummary.label
    : 'Outside Map';
  const semanticZoneCostLabel =
    semanticZoneSummary.maxCost === null ? 'n/a' : `${semanticZoneSummary.maxCost}`;
  const robotBridgeStatusLabel =
    robotBridgeState.status === 'connected'
      ? 'Connected'
      : robotBridgeState.status === 'connecting'
        ? 'Connecting'
        : robotBridgeState.status === 'error'
          ? 'Error'
          : robotBridgeState.status === 'closed'
          ? 'Closed'
          : 'Disabled';
  const gamepadStatusLabel = gamepadState.connected ? 'Connected' : 'Idle';
  const robotFrameCameraId =
    isRobotMode && !isWalkMode && selectedRobotCamera?.id
      ? selectedRobotCamera.id
      : `${cameraMode}-camera`;
  const shouldStreamRobotFrames =
    robotFrameStreamEnabled && robotBridgeState.status === 'connected';
  const shouldStreamRobotDepthFrames =
    robotDepthStreamEnabled && robotBridgeState.status === 'connected';
  const robotBridgePayloadJson = useMemo(
    () =>
      stringifyRobotBridgeMessage(
        'robot-state',
        {
          fragmentId: activeConfig.fragmentId,
          fragmentLabel: activeConfig.fragmentLabel,
          mode,
          cameraMode,
          robotCameraId: selectedRobotCamera?.id ?? null,
          pose: robotPose,
          waypoint: robotWaypoint,
          route: robotTrail,
          routeNodeCount: robotTrail.length,
          routeDistance: Number(robotTrailDistance),
          walkColliderMode: walkColliderStatus.mode,
          semanticZoneMapStatus: semanticZoneState.status,
          semanticZoneIds: semanticZoneHits.map((zone) => zone.id),
          semanticZoneLabels: semanticZoneHits.map((zone) => zone.label)
        },
        {
          source: robotBridgeBrowserSource
        }
      ),
    [
      activeConfig.fragmentId,
      activeConfig.fragmentLabel,
      cameraMode,
      mode,
      robotPose,
      robotTrail,
      robotTrailDistance,
      robotWaypoint,
      semanticZoneHits,
      semanticZoneState.status,
      selectedRobotCamera,
      walkColliderStatus.mode
    ]
  );
  const handleRobotFrame = useCallback(
    (blob, metadata) => {
      if (!robotFrameStreamEnabled) {
        return;
      }

      const socket = robotBridgeSocketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }

      void buildCameraFrameMessage(blob, {
        ...metadata,
        cameraId: robotFrameCameraId
      })
        .then((arrayBuffer) => {
          if (robotBridgeSocketRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
            return;
          }

          try {
            socket.send(arrayBuffer);
          } catch (error) {
            setRobotBridgeState((current) => ({
              ...current,
              error: error instanceof Error ? error.message : 'camera frame send failed'
            }));
            return;
          }

          setRobotBridgeState((current) =>
            current.lastOutboundType === 'camera-frame'
              ? current
              : {
                  ...current,
                  lastOutboundType: 'camera-frame'
                }
          );
        })
        .catch((error) => {
          setRobotBridgeState((current) => ({
            ...current,
            error:
              error instanceof Error ? error.message : 'camera frame message build failed'
          }));
        });
    },
    [robotFrameCameraId, robotFrameStreamEnabled]
  );
  const handleRobotDepthFrame = useCallback(
    (depthBuffer, metadata) => {
      if (!robotDepthStreamEnabled) {
        return;
      }

      const socket = robotBridgeSocketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }

      void buildDepthFrameMessage(depthBuffer, {
        ...metadata,
        cameraId: robotFrameCameraId
      })
        .then((arrayBuffer) => {
          if (robotBridgeSocketRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
            return;
          }

          try {
            socket.send(arrayBuffer);
          } catch (error) {
            setRobotBridgeState((current) => ({
              ...current,
              error: error instanceof Error ? error.message : 'depth frame send failed'
            }));
            return;
          }

          setRobotBridgeState((current) =>
            current.lastOutboundType === 'depth-frame'
              ? current
              : {
                  ...current,
                  lastOutboundType: 'depth-frame'
                }
          );
        })
        .catch((error) => {
          setRobotBridgeState((current) => ({
            ...current,
            error:
              error instanceof Error ? error.message : 'depth frame message build failed'
          }));
        });
    },
    [robotDepthStreamEnabled, robotFrameCameraId]
  );

  useEffect(() => {
    const nextRobotPose = buildRobotPoseFromConfig(activeConfig);
    setRobotPose(nextRobotPose);
    setRobotWaypoint(null);
    setRobotTrail(buildRobotTrailFromPose(nextRobotPose));
    setProjectedRobotPoints([]);
    setProjectedRobotRoutePoints([]);
    setProjectedBenchmarkRoutePoints([]);
    setProjectedSemanticZonePoints([]);
    setProjectedSemanticZoneSurfacePoints([]);
    setSim2RealBenchmarkOverlay(null);
    setSelectedRobotCameraId(
      activeConfig.robotics.defaultCameraId ??
      activeConfig.robotics.cameras[0]?.id ??
      'front'
    );
  }, [activeConfig.fragmentId]);

  useEffect(() => {
    if (!sim2realBenchmarkOverlay) {
      setProjectedBenchmarkRoutePoints([]);
    }
  }, [sim2realBenchmarkOverlay]);

  function applyStudioState(studioState, config = activeConfig, streamScenes = resolvedStreamScenes) {
    if (!studioState) {
      return;
    }

    if (
      studioState.overlayPresetId &&
      config.overlayPresets.some((preset) => preset.id === studioState.overlayPresetId)
    ) {
      setSelectedOverlayPresetId(studioState.overlayPresetId);
    }

    if (
      studioState.filterId &&
      config.dreamFilters.some((filter) => filter.id === studioState.filterId)
    ) {
      setSelectedFilterId(studioState.filterId);
    }

    if (
      studioState.ratioId &&
      config.photoRatios.some((ratio) => ratio.id === studioState.ratioId)
    ) {
      setSelectedRatioId(studioState.ratioId);
    }

    const nextStreamScene = studioState.streamSceneId
      ? streamScenes.find((scene) => scene.id === studioState.streamSceneId)
      : null;
    if (nextStreamScene) {
      setSelectedStreamSceneId(nextStreamScene.id);
      if (!studioState.cameraPresetId && nextStreamScene.presetId) {
        setSelectedPresetId(nextStreamScene.presetId);
      }
    }

    if (
      studioState.cameraPresetId &&
      config.cameraPresets.some((preset) => preset.id === studioState.cameraPresetId)
    ) {
      setSelectedPresetId(studioState.cameraPresetId);
    }
  }

  function applyRobotMissionStartupState(
    mission,
    config = activeConfig,
    streamScenes = resolvedStreamScenes
  ) {
    const nextStreamScene =
      mission.streamSceneId
        ? streamScenes.find((scene) => scene.id === mission.streamSceneId) ?? null
        : null;

    if (nextStreamScene) {
      setSelectedStreamSceneId(nextStreamScene.id);
    }

    const nextPresetId =
      mission.cameraPresetId &&
      config.cameraPresets.some((preset) => preset.id === mission.cameraPresetId)
        ? mission.cameraPresetId
        : !mission.cameraPresetId && nextStreamScene?.presetId
          ? nextStreamScene.presetId
          : '';

    if (nextPresetId) {
      setSelectedPresetId(nextPresetId);
    }

    if (
      mission.robotCameraId &&
      config.robotics.cameras.some((camera) => camera.id === mission.robotCameraId)
    ) {
      setSelectedRobotCameraId(mission.robotCameraId);
    }

    if (document.pointerLockElement) {
      document.exitPointerLock?.();
    }

    if (mission.startupMode === 'live') {
      if (nextStreamScene) {
        activateStreamScene(nextStreamScene);
      } else {
        setCameraMode('orbit');
        setMode('live');
      }
      return;
    }

    if (mission.startupMode === 'photo') {
      setCameraMode('orbit');
      setMode('photo');
      return;
    }

    if (mission.startupMode === 'robot') {
      setSelectedHotspotId(null);
      setActiveModalItem(null);
      setCameraMode('orbit');
      setMode('robot');
      return;
    }

    if (mission.startupMode === 'explore') {
      setCameraMode('orbit');
      setMode('explore');
    }
  }

  function enterWalkMode() {
    if (mode === 'robot') {
      setMode('explore');
    }

    if (selectedHotspotId) {
      setSelectedHotspotId(null);
    }

    if (activeModalItem) {
      setActiveModalItem(null);
    }

    setCameraMode('walk');
    setStatusMessage('Walk Mode に入りました。canvas をクリックして視点を固定します');
  }

  function exitWalkMode() {
    setCameraMode('orbit');
    if (document.pointerLockElement) {
      document.exitPointerLock?.();
    }
    setStatusMessage('Orbit camera に戻りました');
  }

  function togglePhotoMode() {
    if (isWalkMode) {
      exitWalkMode();
    }

    setMode((current) => (current === 'photo' ? 'explore' : 'photo'));
    setStatusMessage('Photo Mode を切り替えました');
  }

  function toggleLiveMode() {
    if (isWalkMode) {
      exitWalkMode();
    }

    if (mode === 'live') {
      setMode('explore');
      setStatusMessage('Live Mode を切り替えました');
      return;
    }

    activateStreamScene(selectedStreamScene ?? resolvedStreamScenes[0] ?? null);
  }

  function setExploreMode() {
    if (isWalkMode) {
      exitWalkMode();
    }

    setMode('explore');
    setStatusMessage('Explore Mode に戻りました');
  }

  function toggleRobotMode() {
    if (isWalkMode) {
      exitWalkMode();
    }

    if (selectedHotspotId || activeModalItem) {
      setSelectedHotspotId(null);
      setActiveModalItem(null);
    }

    setMode((current) => (current === 'robot' ? 'explore' : 'robot'));
    setStatusMessage(
      mode === 'robot'
        ? 'Robot Mode を終了しました'
        : 'Robot Mode に入りました。main stage は robot camera view へ切り替わります'
    );
  }

  function moveRobot(action, options = {}) {
    const { announce = true, statusPrefix = 'Robot teleop' } = options;

    setRobotPose((current) => {
      const nextPose = stepRobotPose(current, action, activeConfig.robotics);
      if (action === 'forward' || action === 'backward') {
        setRobotTrail((currentTrail) => appendRobotTrail(currentTrail, nextPose, activeConfig.robotics));
      }
      return nextPose;
    });

    const actionLabel =
      action === 'forward'
        ? 'forward'
        : action === 'backward'
          ? 'backward'
          : action === 'turn-left'
            ? 'turn left'
            : 'turn right';
    if (announce) {
      setStatusMessage(`${statusPrefix}: ${actionLabel}`);
    }
  }

  function dropRobotWaypoint(options = {}) {
    const { announce = true, statusMessageText = 'Robot waypoint を前方へ配置しました' } = options;
    setRobotWaypoint(buildWaypointAhead(robotPose, activeConfig.robotics));
    if (announce) {
      setStatusMessage(statusMessageText);
    }
  }

  function clearRobotWaypoint(options = {}) {
    const { announce = true, statusMessageText = 'Robot waypoint をクリアしました' } = options;
    setRobotWaypoint(null);
    if (announce) {
      setStatusMessage(statusMessageText);
    }
  }

  function clearRobotRoute(options = {}) {
    const { announce = true, statusMessageText = 'Robot route を現在位置から引き直します' } = options;
    setRobotTrail(buildRobotTrailFromPose(robotPose));
    if (announce) {
      setStatusMessage(statusMessageText);
    }
  }

  function applyRobotRoutePayload(nextPayload, sourceLabel) {
    setRobotRouteImportText(JSON.stringify(nextPayload, null, 2));
    setRobotRouteImportError('');

    if (
      nextPayload.fragmentId &&
      nextPayload.fragmentId !== activeConfig.fragmentId
    ) {
      pendingRobotRouteRef.current = nextPayload;
      navigateToFragment(nextPayload.fragmentId);
      return;
    }

    setRobotPose(nextPayload.pose);
    setRobotTrail(nextPayload.route);
    setRobotWaypoint(nextPayload.waypoint);
    setStatusMessage(`Robot route を適用しました: ${sourceLabel}`);
  }

  function applyRobotMissionPayload(nextMission, nextRoute, sourceLabel, missionUrl = '') {
    const normalizedMission = normalizeRobotMissionPayload(nextMission);
    const normalizedRoute = normalizeRobotRoutePayload(nextRoute);
    const targetFragmentId = normalizedRoute.fragmentId || normalizedMission.fragmentId;

    setRobotMissionState({
      status: 'loaded',
      mission: normalizedMission,
      route: normalizedRoute,
      error: null,
      url: missionUrl
    });

    if (missionUrl) {
      appliedRobotMissionUrlRef.current = missionUrl;
    }

    if (targetFragmentId && targetFragmentId !== activeConfig.fragmentId) {
      pendingRobotMissionStartupRef.current = normalizedMission;
    } else {
      pendingRobotMissionStartupRef.current = null;
      applyRobotMissionStartupState(normalizedMission);
    }

    applyRobotRoutePayload(normalizedRoute, sourceLabel);
  }

  function applyRobotMissionDraftBundle(nextBundle, sourceLabel) {
    const normalizedBundle = normalizeRobotMissionDraftBundle(nextBundle);
    const bundleJson = JSON.stringify(normalizedBundle, null, 2);

    setRobotMissionDraftBundleImportText(bundleJson);
    setRobotMissionDraftBundleImportError('');

    if (normalizedBundle.zones) {
      setSemanticZoneWorkspaceDrafts((current) => ({
        ...current,
        [normalizedBundle.fragmentId]: normalizedBundle.zones
      }));
      setHasSavedSemanticZoneWorkspace(false);
    }

    applyRobotMissionPayload(
      normalizedBundle.mission,
      normalizedBundle.route,
      sourceLabel
    );
    setStatusMessage(`Mission draft bundle を適用しました: ${sourceLabel}`);
  }

  function updateRobotMissionDraftField(field, value) {
    const nextValue =
      field === 'id' ? buildMissionSlug(value, '') : value;

    setRobotMissionState((current) => ({
      ...current,
      mission: {
        ...normalizeRobotMissionPayload({
          ...(current.mission ?? {}),
          id: current.mission?.id ?? robotMissionPayload.id,
          label: current.mission?.label ?? robotMissionPayload.label,
          description:
            current.mission?.description ?? robotMissionPayload.description,
          fragmentId: current.mission?.fragmentId ?? activeConfig.fragmentId,
          fragmentLabel:
            current.mission?.fragmentLabel ?? activeConfig.fragmentLabel,
          accent: current.mission?.accent ?? robotMissionPayload.accent,
          routeUrl: current.mission?.routeUrl ?? robotMissionPayload.routeUrl,
          zoneMapUrl:
            current.mission?.zoneMapUrl ?? robotMissionPayload.zoneMapUrl,
          launchUrl: current.mission?.launchUrl ?? robotMissionPayload.launchUrl,
          cameraPresetId:
            current.mission?.cameraPresetId ?? robotMissionPayload.cameraPresetId,
          robotCameraId:
            current.mission?.robotCameraId ?? robotMissionPayload.robotCameraId,
          streamSceneId:
            current.mission?.streamSceneId ?? robotMissionPayload.streamSceneId,
          startupMode:
            current.mission?.startupMode ?? robotMissionPayload.startupMode,
          world: {
            assetLabel:
              current.mission?.world?.assetLabel ??
              robotMissionPayload.world.assetLabel,
            frameId:
              current.mission?.world?.frameId ??
              robotMissionPayload.world.frameId
          },
          [field]: nextValue
        }),
        [field]: nextValue
      }
    }));
  }

  function updateRobotMissionDraftRouteId(value) {
    setRobotMissionState((current) => {
      const fallbackRouteId =
        extractRobotRouteIdFromUrl(current.mission?.routeUrl) ||
        extractRobotRouteIdFromUrl(robotMissionPayload.routeUrl) ||
        buildMissionSlug(robotRoutePayload.label, `${activeConfig.fragmentId}-route`);
      const nextRouteUrl = buildRobotRouteUrlFromId(value, fallbackRouteId);

      return {
        ...current,
        mission: {
          ...normalizeRobotMissionPayload({
            ...(current.mission ?? {}),
            id: current.mission?.id ?? robotMissionPayload.id,
            label: current.mission?.label ?? robotMissionPayload.label,
            description:
              current.mission?.description ?? robotMissionPayload.description,
            fragmentId: current.mission?.fragmentId ?? robotMissionPayload.fragmentId,
            fragmentLabel:
              current.mission?.fragmentLabel ?? robotMissionPayload.fragmentLabel,
            accent: current.mission?.accent ?? robotMissionPayload.accent,
            routeUrl: nextRouteUrl || robotMissionPayload.routeUrl,
            zoneMapUrl:
              current.mission?.zoneMapUrl ?? robotMissionPayload.zoneMapUrl,
            launchUrl: current.mission?.launchUrl ?? robotMissionPayload.launchUrl,
            cameraPresetId:
              current.mission?.cameraPresetId ?? robotMissionPayload.cameraPresetId,
            robotCameraId:
              current.mission?.robotCameraId ?? robotMissionPayload.robotCameraId,
            streamSceneId:
              current.mission?.streamSceneId ?? robotMissionPayload.streamSceneId,
            startupMode:
              current.mission?.startupMode ?? robotMissionPayload.startupMode,
            world: {
              assetLabel:
                current.mission?.world?.assetLabel ??
                robotMissionPayload.world.assetLabel,
              frameId:
                current.mission?.world?.frameId ??
                robotMissionPayload.world.frameId
            }
          }),
          routeUrl: nextRouteUrl || robotMissionPayload.routeUrl
        }
      };
    });
  }

  function updateRobotMissionDraftRouteField(field, value) {
    setRobotMissionState((current) => ({
      ...current,
      route: normalizeRobotRoutePayload({
        ...(current.route ?? {}),
        ...robotRoutePayload,
        [field]: value
      })
    }));
  }

  function updateRobotMissionDraftWorldField(field, value) {
    setRobotMissionState((current) => ({
      ...current,
      mission: normalizeRobotMissionPayload({
        ...(current.mission ?? {}),
        id: current.mission?.id ?? robotMissionPayload.id,
        label: current.mission?.label ?? robotMissionPayload.label,
        description:
          current.mission?.description ?? robotMissionPayload.description,
        fragmentId: current.mission?.fragmentId ?? activeConfig.fragmentId,
        fragmentLabel:
          current.mission?.fragmentLabel ?? activeConfig.fragmentLabel,
        accent: current.mission?.accent ?? robotMissionPayload.accent,
        routeUrl: current.mission?.routeUrl ?? robotMissionPayload.routeUrl,
        zoneMapUrl:
          current.mission?.zoneMapUrl ?? robotMissionPayload.zoneMapUrl,
        launchUrl: current.mission?.launchUrl ?? robotMissionPayload.launchUrl,
        cameraPresetId:
          current.mission?.cameraPresetId ?? robotMissionPayload.cameraPresetId,
        robotCameraId:
          current.mission?.robotCameraId ?? robotMissionPayload.robotCameraId,
        streamSceneId:
          current.mission?.streamSceneId ?? robotMissionPayload.streamSceneId,
        startupMode:
          current.mission?.startupMode ?? robotMissionPayload.startupMode,
        world: {
          assetLabel:
            current.mission?.world?.assetLabel ??
            robotMissionPayload.world.assetLabel,
          frameId:
            current.mission?.world?.frameId ??
            robotMissionPayload.world.frameId,
          [field]: value
        }
      })
    }));
  }

  async function copyRobotRouteJson() {
    try {
      await navigator.clipboard.writeText(robotRoutePayloadJson);
      setStatusMessage('Robot route JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Robot route JSON をコピーできませんでした');
    }
  }

  async function copyRobotMissionJson() {
    try {
      await navigator.clipboard.writeText(robotMissionPayloadJson);
      setStatusMessage('Robot mission JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Robot mission JSON をコピーできませんでした');
    }
  }

  function downloadRobotRouteJson() {
    downloadTextFile(
      `dreamwalker-live-${activeConfig.fragmentId}-robot-route.json`,
      robotRoutePayloadJson
    );
    setStatusMessage('Robot route JSON を保存しました');
  }

  function downloadRobotMissionJson() {
    downloadTextFile(
      `dreamwalker-live-${activeConfig.fragmentId}-robot-mission.json`,
      robotMissionPayloadJson
    );
    setStatusMessage('Robot mission JSON を保存しました');
  }

  async function copyPublishedRobotMissionJson() {
    try {
      await navigator.clipboard.writeText(publishedRobotMissionPayloadJson);
      setStatusMessage('Published mission preview JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Published mission preview JSON をコピーできませんでした');
    }
  }

  function downloadPublishedRobotMissionJson() {
    downloadTextFile(
      publishedRobotMissionFileName,
      publishedRobotMissionPayloadJson
    );
    setStatusMessage('Published mission preview JSON を保存しました');
  }

  async function copyRobotMissionDraftBundleJson() {
    try {
      await navigator.clipboard.writeText(robotMissionDraftBundleJson);
      setStatusMessage('Mission draft bundle JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Mission draft bundle JSON をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleJson() {
    downloadTextFile(
      robotMissionDraftBundleFileName,
      robotMissionDraftBundleJson
    );
    setStatusMessage('Mission draft bundle JSON を保存しました');
  }

  async function copyRobotMissionPublishCommand() {
    try {
      await navigator.clipboard.writeText(robotMissionPublishCommand);
      setStatusMessage('Mission publish command を clipboard にコピーしました');
    } catch {
      setStatusMessage('Mission publish command をコピーできませんでした');
    }
  }

  async function copyRobotMissionValidateCommand() {
    try {
      await navigator.clipboard.writeText(robotMissionValidateCommand);
      setStatusMessage('Mission validate command を clipboard にコピーしました');
    } catch {
      setStatusMessage('Mission validate command をコピーできませんでした');
    }
  }

  async function copyRobotMissionReleaseCommand() {
    try {
      await navigator.clipboard.writeText(robotMissionReleaseCommand);
      setStatusMessage('Mission release command を clipboard にコピーしました');
    } catch {
      setStatusMessage('Mission release command をコピーできませんでした');
    }
  }

  function downloadRobotMissionPublishCommand() {
    const { publishCommandContent, publishCommandFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(publishCommandFileName, publishCommandContent);
    setStatusMessage(`Mission publish command を保存しました: ${publishCommandFileName}`);
  }

  function downloadRobotMissionValidateCommand() {
    const { validateCommandContent, validateCommandFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(validateCommandFileName, validateCommandContent);
    setStatusMessage(`Mission validate command を保存しました: ${validateCommandFileName}`);
  }

  function downloadRobotMissionReleaseCommand() {
    const { releaseCommandContent, releaseCommandFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(releaseCommandFileName, releaseCommandContent);
    setStatusMessage(`Mission release command を保存しました: ${releaseCommandFileName}`);
  }

  async function copyRobotMissionLaunchUrl() {
    const { launchContent } = buildCurrentRobotMissionDraftBundleArtifacts();

    try {
      await navigator.clipboard.writeText(launchContent);
      setStatusMessage(`Mission launch URL をコピーしました: ${launchContent}`);
    } catch {
      setStatusMessage('Mission launch URL をコピーできませんでした');
    }
  }

  function downloadRobotMissionLaunchUrl() {
    const { launchContent, launchFileName } = buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(launchFileName, launchContent);
    setStatusMessage(`Mission launch URL を保存しました: ${launchFileName}`);
  }

  async function copyRobotMissionPreflightSummary() {
    const { preflightSummaryContent } = buildCurrentRobotMissionDraftBundleArtifacts();

    try {
      await navigator.clipboard.writeText(preflightSummaryContent);
      setStatusMessage('Mission preflight summary をコピーしました');
    } catch {
      setStatusMessage('Mission preflight summary をコピーできませんでした');
    }
  }

  function downloadRobotMissionPreflightSummary() {
    const { preflightSummaryContent, preflightSummaryFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(preflightSummaryFileName, preflightSummaryContent);
    setStatusMessage(`Mission preflight summary を保存しました: ${preflightSummaryFileName}`);
  }

  async function copyRobotMissionPublishReport() {
    const { publishReportContent, publishReportFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    try {
      await navigator.clipboard.writeText(publishReportContent);
      setStatusMessage(`Mission publish report をコピーしました: ${publishReportFileName}`);
    } catch {
      setStatusMessage('Mission publish report をコピーできませんでした');
    }
  }

  function downloadRobotMissionPublishReport() {
    const { publishReportContent, publishReportFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(publishReportFileName, publishReportContent);
    setStatusMessage(`Mission publish report を保存しました: ${publishReportFileName}`);
  }

  async function copyRobotMissionArtifactPack() {
    const { artifactPackContent, artifactPackFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    try {
      await navigator.clipboard.writeText(artifactPackContent);
      setStatusMessage(`Mission artifact pack をコピーしました: ${artifactPackFileName}`);
    } catch {
      setStatusMessage('Mission artifact pack をコピーできませんでした');
    }
  }

  function downloadRobotMissionArtifactPack() {
    const { artifactPackContent, artifactPackFileName } =
      buildCurrentRobotMissionDraftBundleArtifacts();

    downloadTextFile(artifactPackFileName, artifactPackContent);
    setStatusMessage(`Mission artifact pack を保存しました: ${artifactPackFileName}`);
  }

  function applyRobotMissionDraftBundleImportText() {
    try {
      const { bundle: nextBundle } = tryParseRobotMissionDraftBundleImport(
        robotMissionDraftBundleImportText
      );
      applyRobotMissionDraftBundle(nextBundle, 'pasted JSON');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRobotMissionDraftBundleImportError(message);
      setStatusMessage('Mission draft bundle JSON を適用できませんでした');
    }
  }

  function applyRobotMissionDraftBundleImportTextToShelf() {
    try {
      const { bundle: nextBundle, importLabel } = tryParseRobotMissionDraftBundleImport(
        robotMissionDraftBundleImportText
      );
      applyRobotMissionDraftBundle(nextBundle, 'pasted JSON');
      saveRobotMissionDraftBundleSnapshotFromBundle(
        nextBundle,
        importLabel,
        'pasted JSON'
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRobotMissionDraftBundleImportError(message);
      setStatusMessage('Mission draft bundle JSON を shelf へ保存できませんでした');
    }
  }

  function openRobotMissionDraftBundleFilePicker(importMode = 'apply') {
    robotMissionDraftBundleFileImportModeRef.current =
      importMode === 'shelf' ? 'shelf' : 'apply';
    robotMissionDraftBundleFileInputRef.current?.click();
  }

  async function handleRobotMissionDraftBundleFileImport(event) {
    const [file] = event.target.files ?? [];

    if (!file) {
      return;
    }

    try {
      const fileText = await file.text();
      const importMode = robotMissionDraftBundleFileImportModeRef.current;
      const { bundle: nextBundle, importLabel } = tryParseRobotMissionDraftBundleImport(fileText);
      applyRobotMissionDraftBundle(nextBundle, file.name);
      if (importMode === 'shelf') {
        saveRobotMissionDraftBundleSnapshotFromBundle(
          nextBundle,
          importLabel || file.name,
          file.name
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRobotMissionDraftBundleImportError(message);
      setStatusMessage(`Mission draft bundle file を読み込めませんでした: ${file.name}`);
    } finally {
      robotMissionDraftBundleFileImportModeRef.current = 'apply';
      event.target.value = '';
    }
  }

  function applyRobotRouteImportText() {
    try {
      const nextPayload = tryParseRobotRouteJson(robotRouteImportText);
      applyRobotRoutePayload(nextPayload, 'pasted JSON');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRobotRouteImportError(message);
      setStatusMessage('Robot route JSON を適用できませんでした');
    }
  }

  async function handleRobotRouteFileImport(event) {
    const [file] = Array.from(event.target.files ?? []);
    if (!file) {
      return;
    }

    try {
      const fileText = await file.text();
      const nextPayload = tryParseRobotRouteJson(fileText);
      applyRobotRoutePayload(nextPayload, file.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRobotRouteImportError(message);
      setStatusMessage(`Robot route file を読み込めませんでした: ${file.name}`);
    } finally {
      event.target.value = '';
    }
  }

  function persistRobotRouteShelf(nextShelf) {
    const normalizedShelf = nextShelf
      .map((entry, index) => normalizeRobotRouteShelfEntry(entry, index))
      .slice(0, 8);

    setRobotRouteShelf(normalizedShelf);

    if (typeof window !== 'undefined') {
      if (normalizedShelf.length === 0) {
        window.localStorage.removeItem(robotRouteShelfStorageKey);
      } else {
        window.localStorage.setItem(
          robotRouteShelfStorageKey,
          JSON.stringify(normalizedShelf)
        );
      }
    }
  }

  function saveRobotRouteSnapshot() {
    const snapshotLabel = robotRouteShelfSummaryLabel;
    const snapshotEntry = {
      id: `robot-route-${Date.now().toString(36)}`,
      label: snapshotLabel,
      route: normalizeRobotRoutePayload(robotRoutePayload)
    };

    persistRobotRouteShelf([
      snapshotEntry,
      ...robotRouteShelf.filter((entry) => entry.label !== snapshotLabel)
    ]);
    setRobotRouteShelfLabel('');
    setStatusMessage(`Robot route snapshot を保存しました: ${snapshotLabel}`);
  }

  function applyRobotRouteSnapshot(entry) {
    if (!entry) {
      return;
    }

    applyRobotRoutePayload(entry.route, `snapshot:${entry.label}`);
  }

  function downloadRobotRouteSnapshot(entry) {
    if (!entry) {
      return;
    }

    const safeLabel = entry.label.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    downloadTextFile(
      `dreamwalker-live-${safeLabel || 'robot-route'}.json`,
      JSON.stringify(entry.route, null, 2)
    );
    setStatusMessage(`Robot route snapshot を保存しました: ${entry.label}`);
  }

  function deleteRobotRouteSnapshot(entryId) {
    persistRobotRouteShelf(
      robotRouteShelf.filter((entry) => entry.id !== entryId)
    );
    setStatusMessage('Robot route snapshot を削除しました');
  }

  function clearRobotRouteShelf() {
    persistRobotRouteShelf([]);
    setStatusMessage('Robot route shelf を空にしました');
  }

  function persistRobotMissionDraftBundleShelf(nextShelf) {
    const normalizedShelf = nextShelf
      .map((entry, index) =>
        normalizeRobotMissionDraftBundleShelfEntry(entry, index)
      )
      .slice(0, 8);

    setRobotMissionDraftBundleShelf(normalizedShelf);

    if (typeof window !== 'undefined') {
      if (normalizedShelf.length === 0) {
        window.localStorage.removeItem(robotMissionDraftBundleShelfStorageKey);
      } else {
        window.localStorage.setItem(
          robotMissionDraftBundleShelfStorageKey,
          JSON.stringify(normalizedShelf)
        );
      }
    }
  }

  function saveRobotMissionDraftBundleSnapshotFromBundle(
    nextBundle,
    preferredLabel = '',
    sourceLabel = 'snapshot'
  ) {
    const normalizedBundle = normalizeRobotMissionDraftBundle(nextBundle);
    const snapshotLabel =
      preferredLabel.trim() ||
      readNonEmptyString(normalizedBundle.label) ||
      readNonEmptyString(normalizedBundle.mission.label) ||
      `${normalizedBundle.fragmentLabel || normalizedBundle.fragmentId} Draft Bundle`;
    const snapshotEntry = {
      id: `robot-mission-draft-bundle-${Date.now().toString(36)}`,
      label: snapshotLabel,
      bundle: normalizedBundle
    };

    persistRobotMissionDraftBundleShelf([
      snapshotEntry,
      ...robotMissionDraftBundleShelf.filter(
        (entry) => entry.label !== snapshotLabel
      )
    ]);
    setRobotMissionDraftBundleShelfLabel('');
    setStatusMessage(`Mission draft bundle snapshot を保存しました: ${snapshotLabel} (${sourceLabel})`);
    return snapshotEntry;
  }

  function updateRobotMissionDraftBundleSnapshotEntry(entryId, updater) {
    persistRobotMissionDraftBundleShelf(
      robotMissionDraftBundleShelf.map((entry, index) => {
        if (entry.id !== entryId) {
          return entry;
        }

        const updatedEntry =
          typeof updater === 'function' ? updater(entry, index) : entry;

        return normalizeRobotMissionDraftBundleShelfEntry(updatedEntry, index);
      })
    );
  }

  function saveRobotMissionDraftBundleSnapshot() {
    saveRobotMissionDraftBundleSnapshotFromBundle(
      robotMissionDraftBundle,
      robotMissionDraftBundleShelfSummaryLabel,
      'current draft'
    );
  }

  function updateRobotMissionDraftBundleSnapshotLabel(entryId, value) {
    updateRobotMissionDraftBundleSnapshotEntry(entryId, (entry) => ({
      ...entry,
      label: value
    }));
  }

  function updateRobotMissionDraftBundleSnapshotMissionField(entryId, field, value) {
    const nextValue = field === 'id' ? buildMissionSlug(value, '') : value;

    updateRobotMissionDraftBundleSnapshotEntry(entryId, (entry) => {
      const nextMission = normalizeRobotMissionPayload({
        ...entry.bundle.mission,
        [field]: nextValue
      });
      const nextBundle = normalizeRobotMissionDraftBundle({
        ...entry.bundle,
        label:
          field === 'label' && readNonEmptyString(nextValue)
            ? `${nextValue} Draft Bundle`
            : entry.bundle.label,
        mission: nextMission
      });

      return {
        ...entry,
        bundle: nextBundle
      };
    });
  }

  function updateRobotMissionDraftBundleSnapshotMissionRouteId(entryId, value) {
    updateRobotMissionDraftBundleSnapshotEntry(entryId, (entry) => {
      const fallbackRouteId =
        extractRobotRouteIdFromUrl(entry.bundle.mission.routeUrl) ||
        buildMissionSlug(entry.bundle.route.label, `${entry.bundle.fragmentId}-route`);
      const nextMission = normalizeRobotMissionPayload({
        ...entry.bundle.mission,
        routeUrl:
          buildRobotRouteUrlFromId(value, fallbackRouteId) ||
          entry.bundle.mission.routeUrl
      });
      const nextBundle = normalizeRobotMissionDraftBundle({
        ...entry.bundle,
        mission: nextMission
      });

      return {
        ...entry,
        bundle: nextBundle
      };
    });
  }

  function updateRobotMissionDraftBundleSnapshotRouteField(entryId, field, value) {
    updateRobotMissionDraftBundleSnapshotEntry(entryId, (entry) => {
      const nextRoute = normalizeRobotRoutePayload({
        ...entry.bundle.route,
        [field]: value
      });
      const nextBundle = normalizeRobotMissionDraftBundle({
        ...entry.bundle,
        route: nextRoute
      });

      return {
        ...entry,
        bundle: nextBundle
      };
    });
  }

  function updateRobotMissionDraftBundleSnapshotMissionWorldField(
    entryId,
    field,
    value
  ) {
    updateRobotMissionDraftBundleSnapshotEntry(entryId, (entry) => {
      const nextMission = normalizeRobotMissionPayload({
        ...entry.bundle.mission,
        world: {
          ...entry.bundle.mission.world,
          [field]: value
        }
      });
      const nextBundle = normalizeRobotMissionDraftBundle({
        ...entry.bundle,
        mission: nextMission
      });

      return {
        ...entry,
        bundle: nextBundle
      };
    });
  }

  function applyRobotMissionDraftBundleSnapshot(entry) {
    if (!entry) {
      return;
    }

    applyRobotMissionDraftBundle(entry.bundle, `snapshot:${entry.label}`);
  }

  function buildRobotMissionDraftBundleArtifacts(bundle, options = {}) {
    const {
      label = '',
      fragmentIdHint = activeConfig.fragmentId
    } = options;
    const draftBundleFileName = buildRobotMissionDraftBundleFileName(
      bundle,
      fragmentIdHint
    );
    const draftBundleContent = JSON.stringify(bundle, null, 2);
    const missionPayload = normalizeRobotMissionPayload(bundle.mission);
    const missionId = buildMissionSlug(
      missionPayload.id || missionPayload.label || bundle.fragmentId,
      `${bundle.fragmentId}-robot-mission`
    );
    const missionFileName = `${missionId}.robot-mission.json`;
    const missionContent = JSON.stringify(missionPayload, null, 2);
    const {
      payload: publishedPreviewPayload,
      fileName: publishedPreviewFileName
    } = buildPublishedRobotMissionPreviewFromBundle(bundle, activeConfig);
    const preflightSummaryContent = buildRobotMissionPreflightSummary(
      bundle,
      activeConfig
    );
    const publishReportContent = JSON.stringify(
      buildRobotMissionPublishReport(bundle, activeConfig),
      null,
      2
    );
    const publishedPreviewContent = JSON.stringify(publishedPreviewPayload, null, 2);
    const launchFileName = publishedPreviewFileName.endsWith('.mission.json')
      ? publishedPreviewFileName.replace(/\.mission\.json$/i, '.launch-url.txt')
      : `${publishedPreviewFileName}.launch-url.txt`;
    const launchContent = publishedPreviewPayload.launchUrl;
    const preflightSummaryFileName = draftBundleFileName.endsWith('.json')
      ? draftBundleFileName.replace(/\.json$/i, '.preflight.txt')
      : `${draftBundleFileName}.preflight.txt`;
    const publishReportFileName = draftBundleFileName.endsWith('.json')
      ? draftBundleFileName.replace(/\.json$/i, '.publish-report.json')
      : `${draftBundleFileName}.publish-report.json`;
    const artifactPackFileName = buildRobotMissionArtifactPackFileName(
      bundle,
      fragmentIdHint
    );
    const validateCommandContent = buildRobotMissionValidateCommandFromBundle(
      bundle,
      artifactPackFileName,
      activeConfig
    );
    const validateCommandFileName = artifactPackFileName.endsWith('.json')
      ? artifactPackFileName.replace(/\.json$/i, '.validate-command.txt')
      : `${artifactPackFileName}.validate-command.txt`;
    const releaseCommandContent = buildRobotMissionReleaseCommandFromBundle(
      bundle,
      artifactPackFileName,
      activeConfig
    );
    const releaseCommandFileName = artifactPackFileName.endsWith('.json')
      ? artifactPackFileName.replace(/\.json$/i, '.release-command.txt')
      : `${artifactPackFileName}.release-command.txt`;
    const publishCommandContent = buildRobotMissionPublishCommandFromBundle(
      bundle,
      artifactPackFileName,
      activeConfig
    );
    const publishCommandFileName = artifactPackFileName.endsWith('.json')
      ? artifactPackFileName.replace(/\.json$/i, '.publish-command.txt')
      : `${artifactPackFileName}.publish-command.txt`;
    const artifactPackContent = JSON.stringify(
      {
        version: 1,
        protocol: robotMissionArtifactPackProtocolId,
        label: label || missionPayload.label || 'Mission Artifact Pack',
        missionId,
        fragmentId: bundle.fragmentId || fragmentIdHint,
        files: [
          {
            kind: 'draft-bundle',
            fileName: draftBundleFileName,
            mediaType: 'application/json',
            content: draftBundleContent
          },
          {
            kind: 'mission',
            fileName: missionFileName,
            mediaType: 'application/json',
            content: missionContent
          },
          {
            kind: 'published-preview',
            fileName: publishedPreviewFileName,
            mediaType: 'application/json',
            content: publishedPreviewContent
          },
          {
            kind: 'launch-url',
            fileName: launchFileName,
            mediaType: 'text/plain',
            content: launchContent
          },
          {
            kind: 'preflight-summary',
            fileName: preflightSummaryFileName,
            mediaType: 'text/plain',
            content: preflightSummaryContent
          },
          {
            kind: 'publish-report',
            fileName: publishReportFileName,
            mediaType: 'application/json',
            content: publishReportContent
          },
          {
            kind: 'validate-command',
            fileName: validateCommandFileName,
            mediaType: 'text/plain',
            content: validateCommandContent
          },
          {
            kind: 'release-command',
            fileName: releaseCommandFileName,
            mediaType: 'text/plain',
            content: releaseCommandContent
          },
          {
            kind: 'publish-command',
            fileName: publishCommandFileName,
            mediaType: 'text/plain',
            content: publishCommandContent
          }
        ]
      },
      null,
      2
    );

    return {
      draftBundleFileName,
      draftBundleContent,
      missionFileName,
      missionContent,
      publishedPreviewFileName,
      publishedPreviewContent,
      launchFileName,
      launchContent,
      preflightSummaryFileName,
      preflightSummaryContent,
      publishReportFileName,
      publishReportContent,
      validateCommandFileName,
      validateCommandContent,
      releaseCommandFileName,
      releaseCommandContent,
      publishCommandFileName,
      publishCommandContent,
      artifactPackFileName,
      artifactPackContent
    };
  }

  function buildRobotMissionDraftBundleSnapshotArtifacts(entry) {
    return buildRobotMissionDraftBundleArtifacts(entry.bundle, {
      label: entry.label,
      fragmentIdHint: activeConfig.fragmentId
    });
  }

  function buildCurrentRobotMissionDraftBundleArtifacts() {
    return buildRobotMissionDraftBundleArtifacts(robotMissionDraftBundle, {
      label: robotMissionDraftBundleShelfSummaryLabel,
      fragmentIdHint: activeConfig.fragmentId
    });
  }

  function downloadRobotMissionDraftBundleSnapshot(entry) {
    if (!entry) {
      return;
    }

    const fileName = buildRobotMissionDraftBundleFileName(
      entry.bundle,
      activeConfig.fragmentId
    );
    downloadTextFile(
      fileName,
      JSON.stringify(entry.bundle, null, 2)
    );
    setStatusMessage(`Mission draft bundle snapshot を保存しました: ${entry.label}`);
  }

  async function copyRobotMissionDraftBundleSnapshotBundleJson(entry) {
    if (!entry) {
      return;
    }

    const { draftBundleFileName, draftBundleContent } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(draftBundleContent);
      setStatusMessage(`Mission draft bundle をコピーしました: ${draftBundleFileName}`);
    } catch {
      setStatusMessage('Mission draft bundle をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotMissionJson(entry) {
    if (!entry) {
      return;
    }

    const payload = normalizeRobotMissionPayload(entry.bundle.mission);
    const missionId = buildMissionSlug(
      payload.id || payload.label || entry.bundle.fragmentId,
      `${entry.bundle.fragmentId}-robot-mission`
    );
    const fileName = `${missionId}.robot-mission.json`;

    downloadTextFile(fileName, JSON.stringify(payload, null, 2));
    setStatusMessage(`Robot mission JSON を保存しました: ${fileName}`);
  }

  async function copyRobotMissionDraftBundleSnapshotMissionJson(entry) {
    if (!entry) {
      return;
    }

    const { missionFileName, missionContent } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(missionContent);
      setStatusMessage(`Robot mission JSON をコピーしました: ${missionFileName}`);
    } catch {
      setStatusMessage('Robot mission JSON をコピーできませんでした');
    }
  }

  async function copyRobotMissionDraftBundleSnapshotPublishCommand(entry) {
    if (!entry) {
      return;
    }

    const fileName = buildRobotMissionArtifactPackFileName(
      entry.bundle,
      activeConfig.fragmentId
    );
    const command = buildRobotMissionPublishCommandFromBundle(
      entry.bundle,
      fileName,
      activeConfig
    );

    try {
      await navigator.clipboard.writeText(command);
      setStatusMessage(`Mission publish command をコピーしました: ${fileName}`);
    } catch {
      setStatusMessage('Mission publish command をコピーできませんでした');
    }
  }

  async function copyRobotMissionDraftBundleSnapshotValidateCommand(entry) {
    if (!entry) {
      return;
    }

    const fileName = buildRobotMissionArtifactPackFileName(
      entry.bundle,
      activeConfig.fragmentId
    );
    const command = buildRobotMissionValidateCommandFromBundle(
      entry.bundle,
      fileName,
      activeConfig
    );

    try {
      await navigator.clipboard.writeText(command);
      setStatusMessage(`Mission validate command をコピーしました: ${fileName}`);
    } catch {
      setStatusMessage('Mission validate command をコピーできませんでした');
    }
  }

  async function copyRobotMissionDraftBundleSnapshotReleaseCommand(entry) {
    if (!entry) {
      return;
    }

    const fileName = buildRobotMissionArtifactPackFileName(
      entry.bundle,
      activeConfig.fragmentId
    );
    const command = buildRobotMissionReleaseCommandFromBundle(
      entry.bundle,
      fileName,
      activeConfig
    );

    try {
      await navigator.clipboard.writeText(command);
      setStatusMessage(`Mission release command をコピーしました: ${fileName}`);
    } catch {
      setStatusMessage('Mission release command をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotValidateCommand(entry) {
    if (!entry) {
      return;
    }

    const { validateCommandContent, validateCommandFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(validateCommandFileName, validateCommandContent);
    setStatusMessage(`Mission validate command を保存しました: ${validateCommandFileName}`);
  }

  function downloadRobotMissionDraftBundleSnapshotReleaseCommand(entry) {
    if (!entry) {
      return;
    }

    const { releaseCommandContent, releaseCommandFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(releaseCommandFileName, releaseCommandContent);
    setStatusMessage(`Mission release command を保存しました: ${releaseCommandFileName}`);
  }

  function downloadRobotMissionDraftBundleSnapshotPublishCommand(entry) {
    if (!entry) {
      return;
    }

    const { publishCommandContent, publishCommandFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(publishCommandFileName, publishCommandContent);
    setStatusMessage(`Mission publish command を保存しました: ${publishCommandFileName}`);
  }

  async function copyRobotMissionDraftBundleSnapshotPublishedPreview(entry) {
    if (!entry) {
      return;
    }

    const { publishedPreviewContent, publishedPreviewFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(publishedPreviewContent);
      setStatusMessage(`Published mission preview をコピーしました: ${publishedPreviewFileName}`);
    } catch {
      setStatusMessage('Published mission preview をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotPublishedPreview(entry) {
    if (!entry) {
      return;
    }

    const { publishedPreviewContent, publishedPreviewFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(publishedPreviewFileName, publishedPreviewContent);
    setStatusMessage(`Published mission preview を保存しました: ${publishedPreviewFileName}`);
  }

  async function copyRobotMissionDraftBundleSnapshotLaunchUrl(entry) {
    if (!entry) {
      return;
    }

    const { launchContent } = buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(launchContent);
      setStatusMessage(`Mission launch URL をコピーしました: ${launchContent}`);
    } catch {
      setStatusMessage('Mission launch URL をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotLaunchUrl(entry) {
    if (!entry) {
      return;
    }

    const { launchContent, launchFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(launchFileName, launchContent);
    setStatusMessage(`Mission launch URL を保存しました: ${launchFileName}`);
  }

  async function copyRobotMissionDraftBundleSnapshotPreflightSummary(entry) {
    if (!entry) {
      return;
    }

    const { preflightSummaryContent } = buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(preflightSummaryContent);
      setStatusMessage('Mission preflight summary をコピーしました');
    } catch {
      setStatusMessage('Mission preflight summary をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotPreflightSummary(entry) {
    if (!entry) {
      return;
    }

    const { preflightSummaryContent, preflightSummaryFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(preflightSummaryFileName, preflightSummaryContent);
    setStatusMessage(`Mission preflight summary を保存しました: ${preflightSummaryFileName}`);
  }

  async function copyRobotMissionDraftBundleSnapshotPublishReport(entry) {
    if (!entry) {
      return;
    }

    const { publishReportContent, publishReportFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(publishReportContent);
      setStatusMessage(`Mission publish report をコピーしました: ${publishReportFileName}`);
    } catch {
      setStatusMessage('Mission publish report をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotPublishReport(entry) {
    if (!entry) {
      return;
    }

    const { publishReportContent, publishReportFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(publishReportFileName, publishReportContent);
    setStatusMessage(`Mission publish report を保存しました: ${publishReportFileName}`);
  }

  async function copyRobotMissionDraftBundleSnapshotArtifactPack(entry) {
    if (!entry) {
      return;
    }

    const { artifactPackContent, artifactPackFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    try {
      await navigator.clipboard.writeText(artifactPackContent);
      setStatusMessage(`Mission artifact pack をコピーしました: ${artifactPackFileName}`);
    } catch {
      setStatusMessage('Mission artifact pack をコピーできませんでした');
    }
  }

  function downloadRobotMissionDraftBundleSnapshotArtifactPack(entry) {
    if (!entry) {
      return;
    }

    const { artifactPackContent, artifactPackFileName } =
      buildRobotMissionDraftBundleSnapshotArtifacts(entry);

    downloadTextFile(artifactPackFileName, artifactPackContent);
    setStatusMessage(`Mission artifact pack を保存しました: ${artifactPackFileName}`);
  }

  function deleteRobotMissionDraftBundleSnapshot(entryId) {
    persistRobotMissionDraftBundleShelf(
      robotMissionDraftBundleShelf.filter((entry) => entry.id !== entryId)
    );
    setStatusMessage('Mission draft bundle snapshot を削除しました');
  }

  function clearRobotMissionDraftBundleShelf() {
    persistRobotMissionDraftBundleShelf([]);
    setStatusMessage('Mission draft bundle shelf を空にしました');
  }

  function resetRobotPose(options = {}) {
    const { announce = true, statusMessageText = 'Robot pose を fragment spawn に戻しました' } = options;
    const nextRobotPose = buildRobotPoseFromConfig(activeConfig);
    setRobotPose(nextRobotPose);
    setRobotWaypoint(null);
    setRobotTrail(buildRobotTrailFromPose(nextRobotPose));
    if (announce) {
      setStatusMessage(statusMessageText);
    }
  }

  function cycleRobotCamera(direction, options = {}) {
    const { announce = true, statusPrefix = 'Robot camera' } = options;
    const cameras = activeConfig.robotics.cameras;

    if (!cameras.length) {
      return;
    }

    const currentIndex = Math.max(
      0,
      cameras.findIndex((camera) => camera.id === selectedRobotCameraId)
    );
    const offset = direction === 'previous' ? -1 : 1;
    const nextIndex = (currentIndex + offset + cameras.length) % cameras.length;
    const nextCamera = cameras[nextIndex];

    setSelectedRobotCameraId(nextCamera.id);
    if (announce) {
      setStatusMessage(`${statusPrefix}: ${nextCamera.label}`);
    }
  }

  function publishRobotBridgeSnapshot(force = false) {
    const socket = robotBridgeSocketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    if (!force && robotBridgePayloadRef.current === robotBridgePayloadJson) {
      return false;
    }

    socket.send(robotBridgePayloadJson);
    robotBridgePayloadRef.current = robotBridgePayloadJson;
    setRobotBridgeState((current) => ({
      ...current,
      lastOutboundType: 'robot-state'
    }));
    return true;
  }

  function reconnectRobotBridge() {
    setRobotBridgeReconnectNonce((current) => current + 1);
    setStatusMessage('Robot bridge を再接続しています');
  }

  function ensureRobotBridgeMode() {
    if (cameraMode === 'walk') {
      setCameraMode('orbit');
      if (document.pointerLockElement) {
        document.exitPointerLock?.();
      }
    }

    setMode('robot');
  }

  function handleRobotBridgeMessage(rawMessage) {
    try {
      const { message, messageType } = parseRobotBridgeMessage(rawMessage);

      setRobotBridgeState((current) => ({
        ...current,
        lastInboundType: messageType,
        error: null
      }));

      if (messageType === 'bridge-ready') {
        return;
      }

      if (messageType === 'bridge-error') {
        throw new Error(
          typeof message.error === 'string' && message.error.trim()
            ? message.error
            : 'robot bridge error'
        );
      }

      if (messageType === 'request-state') {
        publishRobotBridgeSnapshot(true);
        return;
      }

      if (messageType === 'set-pose') {
        const nextPose = normalizeBridgePoseMessage(message.pose ?? message);

        if (!nextPose) {
          throw new Error('set-pose payload が不正です');
        }

        ensureRobotBridgeMode();
        setRobotPose(nextPose);
        setRobotTrail((currentTrail) =>
          message.resetRoute
            ? buildRobotTrailFromPose(nextPose)
            : appendRobotTrail(currentTrail, nextPose, activeConfig.robotics)
        );
        if (message.clearWaypoint) {
          setRobotWaypoint(null);
        }
        setStatusMessage('Robot bridge: pose update');
        return;
      }

      if (messageType === 'teleop') {
        if (typeof message.action !== 'string') {
          throw new Error('teleop action が不正です');
        }

        ensureRobotBridgeMode();
        moveRobot(message.action);
        return;
      }

      if (messageType === 'set-waypoint') {
        const nextPosition = normalizeBridgePosition(message.position);

        if (!nextPosition) {
          throw new Error('set-waypoint payload が不正です');
        }

        ensureRobotBridgeMode();
        setRobotWaypoint({ position: nextPosition });
        setStatusMessage('Robot bridge: waypoint update');
        return;
      }

      if (messageType === 'clear-waypoint') {
        setRobotWaypoint(null);
        setStatusMessage('Robot bridge: waypoint cleared');
        return;
      }

      if (messageType === 'clear-route') {
        clearRobotRoute();
        return;
      }

      if (messageType === 'reset-pose') {
        resetRobotPose();
        return;
      }

      if (messageType === 'set-camera') {
        if (typeof message.cameraId !== 'string') {
          throw new Error('set-camera cameraId が不正です');
        }

        const nextCamera = activeConfig.robotics.cameras.find(
          (camera) => camera.id === message.cameraId
        );

        if (!nextCamera) {
          throw new Error(`unknown robot camera: ${message.cameraId}`);
        }

        ensureRobotBridgeMode();
        setSelectedRobotCameraId(nextCamera.id);
        setStatusMessage(`Robot bridge: camera ${nextCamera.label}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setRobotBridgeState((current) => ({
        ...current,
        status: 'error',
        error: message
      }));
      setStatusMessage(`Robot bridge message error: ${message}`);
    }
  }

  gamepadActionHandlersRef.current = {
    moveRobot,
    dropRobotWaypoint,
    clearRobotWaypoint,
    clearRobotRoute,
    resetRobotPose,
    cycleRobotCamera
  };

  useEffect(() => {
    const nextTileId = dynamicMapLoadPlan.activeTile?.id ?? '';
    setActiveDynamicMapTileId((currentTileId) =>
      currentTileId === nextTileId ? currentTileId : nextTileId
    );
  }, [dynamicMapLoadPlan.activeTile?.id]);

  useEffect(() => {
    const candidates = dynamicMapLoadPlan.tilePreloadCandidates;
    const activeTileId = dynamicMapLoadPlan.activeTile?.id ?? '';

    if (!hasDynamicMapTileCatalog || candidates.length === 0) {
      setTilePreloadState({
        status: 'idle',
        mode: dynamicMapTilePreloadMode,
        activeTileId,
        entries: [],
        results: [],
        error: null
      });
      return undefined;
    }

    if (dynamicMapTilePreloadMode === 'off') {
      setTilePreloadState({
        status: 'skipped',
        mode: dynamicMapTilePreloadMode,
        activeTileId,
        entries: candidates,
        results: [],
        error: null
      });
      return undefined;
    }

    const abortController = new AbortController();
    setTilePreloadState({
      status: 'loading',
      mode: dynamicMapTilePreloadMode,
      activeTileId,
      entries: candidates,
      results: [],
      error: null
    });

    const buildCacheKey = (entry) =>
      buildDynamicMapTilePreloadCacheKey({
        catalogUrl: tileCatalogState.url,
        colliderMeshUrl: entry.colliderMeshUrl,
        mode: dynamicMapTilePreloadMode,
        preloadCacheKey: entry.preloadCacheKey,
        sceneId: entry.tileCatalogSceneId,
        splatUrl: entry.splatUrl,
        tileId: entry.tileId
      });
    const preloadResultCache = tilePreloadResultCacheRef.current;

    Promise.all(
      candidates.map(async (entry) => {
        const cacheKey = buildCacheKey(entry);
        const cached = preloadResultCache.get(cacheKey);
        if (cached?.result) {
          return {
            ...cached.result,
            cached: true
          };
        }

        if (cached?.promise) {
          const result = await cached.promise;
          return {
            ...result,
            cached: true
          };
        }

        const promise = preloadDynamicMapEntry(entry, {
          includeDemoFallback: true,
          mode: dynamicMapTilePreloadMode
        });
        preloadResultCache.set(cacheKey, { promise });
        const result = await promise;

        if (['ready', 'skipped'].includes(result.status)) {
          preloadResultCache.set(cacheKey, { result });
        } else {
          preloadResultCache.delete(cacheKey);
        }

        return result;
      })
    )
      .then((results) => {
        if (abortController.signal.aborted) {
          return;
        }

        const hasWarning = results.some((result) =>
          ['warning', 'error'].includes(result.status)
        );
        const hasAborted = results.some((result) => result.status === 'aborted');

        setTilePreloadState({
          status: hasAborted ? 'aborted' : hasWarning ? 'warning' : 'ready',
          mode: dynamicMapTilePreloadMode,
          activeTileId,
          entries: candidates,
          results,
          error: null
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setTilePreloadState({
          status: 'error',
          mode: dynamicMapTilePreloadMode,
          activeTileId,
          entries: candidates,
          results: [],
          error: error instanceof Error ? error.message : String(error)
        });
      });

    return () => abortController.abort();
  }, [
    dynamicMapLoadPlan.activeTile?.id,
    dynamicMapTilePreloadKey,
    hasDynamicMapTileCatalog,
    dynamicMapTilePreloadMode,
    tileCatalogState.url
  ]);

  useEffect(() => {
    if (assetManifestState.status !== 'loaded') {
      return;
    }

    if (studioBundleState.status === 'loaded') {
      return;
    }

    if (hasSavedAssetWorkspace || assetWorkspaceTouchedRef.current) {
      return;
    }

    const normalizedManifest = normalizeAssetManifest(assetManifestState.manifest);
    setAssetWorkspaceDraft(normalizedManifest);
    setAssetWorkspaceBaselineJson(JSON.stringify(normalizedManifest));
  }, [
    assetManifestState.manifest,
    assetManifestState.status,
    hasSavedAssetWorkspace,
    studioBundleState.status
  ]);

  useEffect(() => {
    if (!assetManifestUrl) {
      setAssetManifestState({
        status: 'disabled',
        manifest: null,
        error: null,
        url: ''
      });
      return;
    }

    const abortController = new AbortController();

    setAssetManifestState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: assetManifestUrl
    }));

    fetch(assetManifestUrl, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json'
      },
      signal: abortController.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 404) {
            setAssetManifestState({
              status: 'missing',
              manifest: null,
              error: null,
              url: assetManifestUrl
            });
            return;
          }

          throw new Error(`HTTP ${response.status}`);
        }

        const manifest = await response.json();
        setAssetManifestState({
          status: 'loaded',
          manifest,
          error: null,
          url: assetManifestUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setAssetManifestState({
          status: 'error',
          manifest: null,
          error: error instanceof Error ? error.message : String(error),
          url: assetManifestUrl
        });
      });

    return () => abortController.abort();
  }, [assetManifestUrl]);

  useEffect(() => {
    if (!tileCatalogUrl) {
      setTileCatalogState({
        status: 'disabled',
        catalog: null,
        error: null,
        url: ''
      });
      return;
    }

    const abortController = new AbortController();

    setTileCatalogState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: tileCatalogUrl
    }));

    fetch(tileCatalogUrl, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json'
      },
      signal: abortController.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 404) {
            setTileCatalogState({
              status: 'missing',
              catalog: null,
              error: null,
              url: tileCatalogUrl
            });
            return;
          }

          throw new Error(`HTTP ${response.status}`);
        }

        const catalog = await response.json();
        setTileCatalogState({
          status: 'loaded',
          catalog,
          error: null,
          url: tileCatalogUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setTileCatalogState({
          status: 'error',
          catalog: null,
          error: error instanceof Error ? error.message : String(error),
          url: tileCatalogUrl
        });
      });

    return () => abortController.abort();
  }, [tileCatalogUrl]);

  useEffect(() => {
    if (!studioBundleUrl) {
      setStudioBundleState({
        status: 'disabled',
        bundle: null,
        error: null,
        url: ''
      });
      appliedStudioBundleUrlRef.current = '';
      return;
    }

    const abortController = new AbortController();

    setStudioBundleState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: studioBundleUrl
    }));

    fetch(studioBundleUrl, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json'
      },
      signal: abortController.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 404) {
            setStudioBundleState({
              status: 'missing',
              bundle: null,
              error: null,
              url: studioBundleUrl
            });
            return;
          }

          throw new Error(`HTTP ${response.status}`);
        }

        const bundle = await response.json();
        setStudioBundleState({
          status: 'loaded',
          bundle,
          error: null,
          url: studioBundleUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setStudioBundleState({
          status: 'error',
          bundle: null,
          error: error instanceof Error ? error.message : String(error),
          url: studioBundleUrl
        });
      });

    return () => abortController.abort();
  }, [studioBundleUrl]);

  useEffect(() => {
    if (!studioBundleCatalogUrl) {
      setStudioBundleCatalogState({
        status: 'disabled',
        catalog: null,
        error: null,
        url: ''
      });
      return;
    }

    const abortController = new AbortController();

    setStudioBundleCatalogState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: studioBundleCatalogUrl
    }));

    fetch(studioBundleCatalogUrl, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json'
      },
      signal: abortController.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 404) {
            setStudioBundleCatalogState({
              status: 'missing',
              catalog: null,
              error: null,
              url: studioBundleCatalogUrl
            });
            return;
          }

          throw new Error(`HTTP ${response.status}`);
        }

        const catalog = await response.json();
        setStudioBundleCatalogState({
          status: 'loaded',
          catalog,
          error: null,
          url: studioBundleCatalogUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setStudioBundleCatalogState({
          status: 'error',
          catalog: null,
          error: error instanceof Error ? error.message : String(error),
          url: studioBundleCatalogUrl
        });
      });

    return () => abortController.abort();
  }, [studioBundleCatalogUrl]);

  useEffect(() => {
    if (!robotRouteUrl) {
      setRobotRouteState({
        status: 'disabled',
        route: null,
        error: null,
        url: ''
      });
      appliedRobotRouteUrlRef.current = '';
      return;
    }

    const abortController = new AbortController();

    setRobotRouteState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: robotRouteUrl
    }));

    fetch(robotRouteUrl, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json'
      },
      signal: abortController.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 404) {
            setRobotRouteState({
              status: 'missing',
              route: null,
              error: null,
              url: robotRouteUrl
            });
            return;
          }

          throw new Error(`HTTP ${response.status}`);
        }

        const route = await response.json();
        setRobotRouteState({
          status: 'loaded',
          route,
          error: null,
          url: robotRouteUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setRobotRouteState({
          status: 'error',
          route: null,
          error: error instanceof Error ? error.message : String(error),
          url: robotRouteUrl
        });
      });

    return () => abortController.abort();
  }, [robotRouteUrl]);

  useEffect(() => {
    if (!robotRouteCatalogUrl) {
      setRobotRouteCatalogState({
        status: 'disabled',
        catalog: null,
        error: null,
        url: ''
      });
      return;
    }

    const abortController = new AbortController();

    setRobotRouteCatalogState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: robotRouteCatalogUrl
    }));

    fetch(robotRouteCatalogUrl, {
      cache: 'no-store',
      headers: {
        Accept: 'application/json'
      },
      signal: abortController.signal
    })
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 404) {
            setRobotRouteCatalogState({
              status: 'missing',
              catalog: null,
              error: null,
              url: robotRouteCatalogUrl
            });
            return;
          }

          throw new Error(`HTTP ${response.status}`);
        }

        const catalog = await response.json();
        setRobotRouteCatalogState({
          status: 'loaded',
          catalog,
          error: null,
          url: robotRouteCatalogUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        setRobotRouteCatalogState({
          status: 'error',
          catalog: null,
          error: error instanceof Error ? error.message : String(error),
          url: robotRouteCatalogUrl
        });
      });

    return () => abortController.abort();
  }, [robotRouteCatalogUrl]);

  useEffect(() => {
    if (!robotMissionUrl) {
      setRobotMissionState({
        status: 'disabled',
        mission: null,
        route: null,
        error: null,
        url: ''
      });
      appliedRobotMissionUrlRef.current = '';
      return;
    }

    const abortController = new AbortController();

    setRobotMissionState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: robotMissionUrl
    }));

    loadRobotMissionResource(robotMissionUrl, {
      signal: abortController.signal
    })
      .then(({ mission, route }) => {
        setRobotMissionState({
          status: 'loaded',
          mission,
          route,
          error: null,
          url: robotMissionUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        const message = error instanceof Error ? error.message : String(error);
        const isMissing = message.startsWith('HTTP 404');

        setRobotMissionState({
          status: isMissing ? 'missing' : 'error',
          mission: null,
          route: null,
          error: isMissing ? null : message,
          url: robotMissionUrl
        });
      });

    return () => abortController.abort();
  }, [robotMissionUrl]);

  useEffect(() => {
    if (!robotMissionCatalogUrl) {
      setRobotMissionCatalogState({
        status: 'disabled',
        catalog: null,
        error: null,
        url: ''
      });
      return;
    }

    const abortController = new AbortController();

    setRobotMissionCatalogState((current) => ({
      ...current,
      status: 'loading',
      error: null,
      url: robotMissionCatalogUrl
    }));

    fetchJsonResource(robotMissionCatalogUrl, {
      signal: abortController.signal
    })
      .then((catalog) => {
        setRobotMissionCatalogState({
          status: 'loaded',
          catalog,
          error: null,
          url: robotMissionCatalogUrl
        });
      })
      .catch((error) => {
        if (abortController.signal.aborted) {
          return;
        }

        const message = error instanceof Error ? error.message : String(error);
        const isMissing = message.startsWith('HTTP 404');

        setRobotMissionCatalogState({
          status: isMissing ? 'missing' : 'error',
          catalog: null,
          error: isMissing ? null : message,
          url: robotMissionCatalogUrl
        });
      });

    return () => abortController.abort();
  }, [robotMissionCatalogUrl]);

  useEffect(() => {
    if (studioBundleState.status !== 'loaded') {
      return;
    }

    if (
      appliedStudioBundleUrlRef.current &&
      appliedStudioBundleUrlRef.current === studioBundleState.url
    ) {
      return;
    }

    const nextBundle = normalizeStudioBundle(studioBundleState.bundle);
    appliedStudioBundleUrlRef.current = studioBundleState.url;
    applyStudioBundle(nextBundle, `url:${studioBundleState.url}`);
  }, [studioBundleState.bundle, studioBundleState.status, studioBundleState.url]);

  useEffect(() => {
    if (robotRouteState.status !== 'loaded') {
      return;
    }

    if (
      appliedRobotRouteUrlRef.current &&
      appliedRobotRouteUrlRef.current === robotRouteState.url
    ) {
      return;
    }

    const nextRoute = normalizeRobotRoutePayload(robotRouteState.route);
    appliedRobotRouteUrlRef.current = robotRouteState.url;
    applyRobotRoutePayload(nextRoute, `url:${robotRouteState.url}`);
  }, [robotRouteState.route, robotRouteState.status, robotRouteState.url]);

  useEffect(() => {
    if (robotRoutePlaybackTimerRef.current) {
      clearTimeout(robotRoutePlaybackTimerRef.current);
      robotRoutePlaybackTimerRef.current = null;
    }

    if (
      !robotRoutePlaybackConfig.enabled ||
      robotRouteState.status !== 'loaded' ||
      !robotRouteState.route
    ) {
      setRobotRoutePlaybackPreviewPositions([]);
      return undefined;
    }

    let route;
    try {
      route = normalizeRobotRoutePayload(robotRouteState.route);
    } catch (error) {
      setRobotRoutePlaybackPreviewPositions([]);
      setStatusMessage(
        `Robot route playback error: ${error instanceof Error ? error.message : String(error)}`
      );
      return undefined;
    }

    if (route.fragmentId && route.fragmentId !== activeConfig.fragmentId) {
      setRobotRoutePlaybackPreviewPositions([]);
      return undefined;
    }

    const routePoints = route.route.length > 0
      ? route.route.map((position) => [...position])
      : [[...route.pose.position]];
    if (routePoints.length === 0) {
      setRobotRoutePlaybackPreviewPositions([]);
      return undefined;
    }

    let disposed = false;
    let routePointIndex = 0;
    const routeLabel = route.label || route.frameId || 'route';

    setCameraMode((current) => (current === 'walk' ? 'orbit' : current));
    setMode('robot');
    setRobotWaypoint(route.waypoint);

    function buildRoutePreviewPositions(currentIndex) {
      const previewLimit = Math.max(0, dynamicMapTilePreloadLimit);
      const positions = [];

      for (let offset = 1; offset <= previewLimit; offset += 1) {
        const nextIndex = currentIndex + offset;
        if (nextIndex >= routePoints.length && !robotRoutePlaybackConfig.loop) {
          break;
        }

        const routePoint = routePoints[nextIndex % routePoints.length];
        positions.push({
          x: routePoint[0],
          y: routePoint[1],
          z: routePoint[2]
        });
      }

      return positions;
    }

    function applyRoutePlaybackPoint() {
      if (disposed) {
        return;
      }

      const position = routePoints[routePointIndex];
      const yawDegrees =
        routePointIndex < routePoints.length - 1
          ? yawDegreesFromRouteSegment(
              position,
              routePoints[routePointIndex + 1],
              route.pose.yawDegrees
            )
          : yawDegreesFromRouteSegment(
              routePoints[routePointIndex - 1],
              position,
              route.pose.yawDegrees
            );

      setRobotPose({
        position: [...position],
        yawDegrees
      });
      setRobotTrail(routePoints.slice(0, routePointIndex + 1).map((point) => [...point]));
      setRobotRoutePlaybackPreviewPositions(buildRoutePreviewPositions(routePointIndex));
      setStatusMessage(
        `Robot route playback: ${routeLabel} ${routePointIndex + 1}/${routePoints.length}`
      );

      routePointIndex += 1;
      if (routePointIndex >= routePoints.length) {
        if (!robotRoutePlaybackConfig.loop) {
          robotRoutePlaybackTimerRef.current = null;
          return;
        }

        routePointIndex = 0;
      }

      robotRoutePlaybackTimerRef.current = setTimeout(
        applyRoutePlaybackPoint,
        robotRoutePlaybackConfig.intervalMs
      );
    }

    robotRoutePlaybackTimerRef.current = setTimeout(applyRoutePlaybackPoint, 0);

    return () => {
      disposed = true;
      if (robotRoutePlaybackTimerRef.current) {
        clearTimeout(robotRoutePlaybackTimerRef.current);
        robotRoutePlaybackTimerRef.current = null;
      }
    };
  }, [
    activeConfig.fragmentId,
    dynamicMapTilePreloadLimit,
    robotRoutePlaybackConfig.enabled,
    robotRoutePlaybackConfig.intervalMs,
    robotRoutePlaybackConfig.loop,
    robotRouteState.route,
    robotRouteState.status
  ]);

  useEffect(() => {
    if (
      robotMissionState.status !== 'loaded' ||
      !robotMissionState.route ||
      !robotMissionState.mission
    ) {
      return;
    }

    if (
      appliedRobotMissionUrlRef.current &&
      appliedRobotMissionUrlRef.current === robotMissionState.url
    ) {
      return;
    }

    appliedRobotMissionUrlRef.current = robotMissionState.url;
    applyRobotMissionPayload(
      robotMissionState.mission,
      robotMissionState.route,
      `mission:${robotMissionState.url}`,
      robotMissionState.url
    );
  }, [
    robotMissionState.mission,
    robotMissionState.route,
    robotMissionState.status,
    robotMissionState.url
  ]);

  useEffect(() => {
    function handleHashChange() {
      setCurrentFragmentId(parseFragmentIdFromHash());
    }

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    setSelectedPresetId(activeConfig.homePresetId);
    setSelectedStreamSceneId(resolvedStreamScenes[0]?.id ?? null);
    setCollectedShardIds(loadCollectedShards(shardStorageKey));
    setSelectedHotspotId(null);
    setActiveModalItem(null);
    setProjectedHotspots([]);
    setProjectedLoopItems([]);
    setWalkColliderStatus(
      activeWorldConfig.colliderMeshUrl
        ? { mode: 'idle', error: null }
        : { mode: 'proxy', error: null }
    );

    if (document.pointerLockElement) {
      document.exitPointerLock?.();
    }

    setCameraMode('orbit');
    setMode('explore');
    setStatusMessage(`Fragment: ${activeConfig.fragmentLabel}`);
  }, [
    activeConfig.fragmentId,
    activeConfig.fragmentLabel,
    activeConfig.homePresetId,
    activeWorldConfig.colliderMeshUrl,
    shardStorageKey
  ]);

  useEffect(() => {
    const pendingStudioState = pendingStudioStateRef.current;
    if (!pendingStudioState) {
      return;
    }

    if (
      pendingStudioState.fragmentId &&
      pendingStudioState.fragmentId !== activeConfig.fragmentId
    ) {
      return;
    }

    applyStudioState(pendingStudioState, activeConfig, resolvedStreamScenes);
    pendingStudioStateRef.current = null;
  }, [activeConfig, resolvedStreamScenes]);

  useEffect(() => {
    const pendingRobotRoute = pendingRobotRouteRef.current;
    if (!pendingRobotRoute) {
      return;
    }

    if (
      pendingRobotRoute.fragmentId &&
      pendingRobotRoute.fragmentId !== activeConfig.fragmentId
    ) {
      return;
    }

    setRobotPose(pendingRobotRoute.pose);
    setRobotTrail(pendingRobotRoute.route);
    setRobotWaypoint(pendingRobotRoute.waypoint);
    setRobotRouteImportText(JSON.stringify(pendingRobotRoute, null, 2));
    setRobotRouteImportError('');
    pendingRobotRouteRef.current = null;
    setStatusMessage(
      `Robot route を適用しました: ${pendingRobotRoute.label || activeConfig.fragmentLabel}`
    );
  }, [activeConfig.fragmentId]);

  useEffect(() => {
    const pendingRobotMissionStartup = pendingRobotMissionStartupRef.current;
    if (!pendingRobotMissionStartup) {
      return;
    }

    if (
      pendingRobotMissionStartup.fragmentId &&
      pendingRobotMissionStartup.fragmentId !== activeConfig.fragmentId
    ) {
      return;
    }

    applyRobotMissionStartupState(
      pendingRobotMissionStartup,
      activeConfig,
      resolvedStreamScenes
    );
    pendingRobotMissionStartupRef.current = null;
  }, [activeConfig, resolvedStreamScenes]);

  useEffect(() => {
    function handlePointerLockChange() {
      const canvas = document.querySelector('.dreamwalker-stage canvas');
      setIsPointerLocked(document.pointerLockElement === canvas);
    }

    document.addEventListener('pointerlockchange', handlePointerLockChange);
    return () => document.removeEventListener('pointerlockchange', handlePointerLockChange);
  }, []);

  useEffect(() => {
    const canvas = document.querySelector('.dreamwalker-stage canvas');
    if (!(canvas instanceof HTMLCanvasElement)) {
      return undefined;
    }

    function requestPointerLock() {
      if (!isWalkMode || document.pointerLockElement === canvas) {
        return;
      }

      canvas.requestPointerLock?.();
    }

    canvas.addEventListener('click', requestPointerLock);
    return () => canvas.removeEventListener('click', requestPointerLock);
  }, [isWalkMode]);

  useEffect(() => {
    if ((selectedHotspotId || activeModalItem) && isPointerLocked) {
      document.exitPointerLock?.();
    }
  }, [activeModalItem, isPointerLocked, selectedHotspotId]);

  useEffect(() => {
    if (!isWalkMode && isPointerLocked) {
      document.exitPointerLock?.();
    }
  }, [isPointerLocked, isWalkMode]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    window.localStorage.setItem(
      shardStorageKey,
      JSON.stringify(collectedShardIds)
    );
  }, [collectedShardIds]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    if (isOverlayMode) {
      setOverlayState(loadOverlayState());

      function handleStorage(event) {
        if (event.key !== overlayStateKey) {
          return;
        }

        try {
          setOverlayState(event.newValue ? JSON.parse(event.newValue) : null);
        } catch {
          setOverlayState(null);
        }
      }

      window.addEventListener('storage', handleStorage);
      let eventSource = null;
      let handleOverlayEvent = null;

      if (relayConfig.enabled) {
        try {
          eventSource = new EventSource(buildRelayEndpoint(relayConfig.url, '/events'));
          handleOverlayEvent = (event) => {
            try {
              setOverlayState(event.data ? JSON.parse(event.data) : null);
            } catch {
              setOverlayState(null);
            }
          };
          eventSource.addEventListener('overlay', handleOverlayEvent);
        } catch {
          eventSource = null;
        }
      }

      return () => {
        window.removeEventListener('storage', handleStorage);
        if (eventSource) {
          if (handleOverlayEvent) {
            eventSource.removeEventListener('overlay', handleOverlayEvent);
          }
          eventSource.close();
        }
      };
    }

    window.localStorage.setItem(overlayStateKey, liveScenePayloadJson);
    setOverlayState(liveScenePayload);

    if (!relayConfig.enabled) {
      return;
    }

    const abortController = new AbortController();
    fetch(buildRelayEndpoint(relayConfig.url, '/publish'), {
      method: 'POST',
      headers: {
        'Content-Type': 'text/plain;charset=UTF-8'
      },
      body: liveScenePayloadJson,
      mode: 'cors',
      signal: abortController.signal
    }).catch(() => {});

    return () => abortController.abort();
  }, [isOverlayMode, liveScenePayload, liveScenePayloadJson, relayConfig.enabled, relayConfig.url]);

  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }

    document.body.classList.toggle('overlay-mode', isOverlayMode);
    return () => document.body.classList.remove('overlay-mode');
  }, [isOverlayMode]);

  useEffect(() => {
    if (isOverlayMode || !robotBridgeConfig.enabled) {
      robotBridgeSocketRef.current?.close(1000, 'robot bridge disabled');
      robotBridgeSocketRef.current = null;
      robotBridgePayloadRef.current = '';
      setRobotBridgeState({
        status: 'disabled',
        lastInboundType: null,
        lastOutboundType: null,
        error: null
      });
      return;
    }

    let disposed = false;
    const socket = new WebSocket(robotBridgeConfig.url);
    robotBridgeSocketRef.current = socket;
    robotBridgePayloadRef.current = '';
    setRobotBridgeState((current) => ({
      ...current,
      status: 'connecting',
      error: null
    }));

    socket.addEventListener('open', () => {
      if (disposed) {
        return;
      }

      setRobotBridgeState((current) => ({
        ...current,
        status: 'connected',
        error: null
      }));
      setStatusMessage('Robot bridge connected');
    });

    socket.addEventListener('message', (event) => {
      if (disposed) {
        return;
      }

      handleRobotBridgeMessage(event.data);
    });

    socket.addEventListener('error', () => {
      if (disposed) {
        return;
      }

      setRobotBridgeState((current) => ({
        ...current,
        status: 'error',
        error: `robot bridge に接続できません: ${robotBridgeConfig.url}`
      }));
    });

    socket.addEventListener('close', () => {
      if (robotBridgeSocketRef.current === socket) {
        robotBridgeSocketRef.current = null;
      }

      if (disposed) {
        return;
      }

      robotBridgePayloadRef.current = '';
      setRobotBridgeState((current) => ({
        ...current,
        status: current.status === 'error' ? 'error' : 'closed'
      }));
    });

    return () => {
      disposed = true;
      if (robotBridgeSocketRef.current === socket) {
        robotBridgeSocketRef.current = null;
      }
      socket.close(1000, 'cleanup');
    };
  }, [isOverlayMode, robotBridgeConfig.enabled, robotBridgeConfig.url, robotBridgeReconnectNonce]);

  useEffect(() => {
    if (isOverlayMode || !robotBridgeConfig.enabled) {
      return;
    }

    publishRobotBridgeSnapshot();
  }, [isOverlayMode, robotBridgeConfig.enabled, robotBridgePayloadJson]);

  useEffect(() => {
    if (typeof window === 'undefined' || typeof navigator === 'undefined') {
      return;
    }

    if (isOverlayMode || !isRobotMode || isWalkMode) {
      gamepadCommandStateRef.current = {
        moveAt: 0,
        turnAt: 0,
        buttonTimes: {},
        previousButtons: {},
        connectedId: null
      };
      setGamepadState({
        connected: false,
        label: 'No Gamepad',
        mapping: null
      });
      return;
    }

    let frameId = 0;

    function setDisconnectedState() {
      const current = gamepadCommandStateRef.current;
      if (current.connectedId === null) {
        return;
      }

      gamepadCommandStateRef.current = {
        moveAt: 0,
        turnAt: 0,
        buttonTimes: {},
        previousButtons: {},
        connectedId: null
      };
      setGamepadState({
        connected: false,
        label: 'No Gamepad',
        mapping: null
      });
    }

    function tick(now) {
      const gamepads = navigator.getGamepads?.() ?? [];
      const activeGamepad = Array.from(gamepads).find(
        (gamepad) => gamepad && gamepad.connected
      );

      if (!activeGamepad) {
        setDisconnectedState();
        frameId = window.requestAnimationFrame(tick);
        return;
      }

      const commandState = gamepadCommandStateRef.current;
      if (commandState.connectedId !== activeGamepad.id) {
        commandState.connectedId = activeGamepad.id;
        commandState.buttonTimes = {};
        commandState.previousButtons = {};
        setGamepadState({
          connected: true,
          label: activeGamepad.id || 'Gamepad',
          mapping: activeGamepad.mapping || null
        });
      }

      const handlers = gamepadActionHandlersRef.current;
      if (!handlers) {
        frameId = window.requestAnimationFrame(tick);
        return;
      }

      const deadzone = activeConfig.robotics.gamepadDeadzone ?? 0.35;
      const repeatMs = activeConfig.robotics.gamepadRepeatMs ?? 180;
      const buttonRepeatMs = activeConfig.robotics.gamepadButtonRepeatMs ?? 240;
      const horizontalAxis = activeGamepad.axes?.[0] ?? 0;
      const verticalAxis = activeGamepad.axes?.[1] ?? 0;
      const dpadUp = Boolean(activeGamepad.buttons?.[12]?.pressed);
      const dpadDown = Boolean(activeGamepad.buttons?.[13]?.pressed);
      const dpadLeft = Boolean(activeGamepad.buttons?.[14]?.pressed);
      const dpadRight = Boolean(activeGamepad.buttons?.[15]?.pressed);
      const verticalIntent =
        dpadUp ? -1 : dpadDown ? 1 : Math.abs(verticalAxis) >= deadzone ? verticalAxis : 0;
      const horizontalIntent =
        dpadLeft ? -1 : dpadRight ? 1 : Math.abs(horizontalAxis) >= deadzone ? horizontalAxis : 0;

      if (verticalIntent <= -deadzone && now - commandState.moveAt >= repeatMs) {
        handlers.moveRobot('forward', {
          announce: false
        });
        commandState.moveAt = now;
      } else if (verticalIntent >= deadzone && now - commandState.moveAt >= repeatMs) {
        handlers.moveRobot('backward', {
          announce: false
        });
        commandState.moveAt = now;
      }

      if (horizontalIntent <= -deadzone && now - commandState.turnAt >= repeatMs) {
        handlers.moveRobot('turn-left', {
          announce: false
        });
        commandState.turnAt = now;
      } else if (horizontalIntent >= deadzone && now - commandState.turnAt >= repeatMs) {
        handlers.moveRobot('turn-right', {
          announce: false
        });
        commandState.turnAt = now;
      }

      const buttonActions = [
        {
          key: 'waypoint',
          index: 0,
          action: () => handlers.dropRobotWaypoint()
        },
        {
          key: 'clear-waypoint',
          index: 2,
          action: () => handlers.clearRobotWaypoint()
        },
        {
          key: 'clear-route',
          index: 1,
          action: () => handlers.clearRobotRoute()
        },
        {
          key: 'reset-pose',
          index: 3,
          action: () => handlers.resetRobotPose()
        },
        {
          key: 'prev-camera',
          index: 4,
          action: () => handlers.cycleRobotCamera('previous', { statusPrefix: 'Gamepad camera' })
        },
        {
          key: 'next-camera',
          index: 5,
          action: () => handlers.cycleRobotCamera('next', { statusPrefix: 'Gamepad camera' })
        }
      ];

      for (const buttonAction of buttonActions) {
        const isPressed = Boolean(activeGamepad.buttons?.[buttonAction.index]?.pressed);
        const wasPressed = Boolean(commandState.previousButtons[buttonAction.key]);
        const lastTriggeredAt = commandState.buttonTimes[buttonAction.key] ?? 0;

        if (isPressed && !wasPressed && now - lastTriggeredAt >= buttonRepeatMs) {
          buttonAction.action();
          commandState.buttonTimes[buttonAction.key] = now;
        }

        commandState.previousButtons[buttonAction.key] = isPressed;
      }

      frameId = window.requestAnimationFrame(tick);
    }

    frameId = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [
    activeConfig.robotics.gamepadButtonRepeatMs,
    activeConfig.robotics.gamepadDeadzone,
    activeConfig.robotics.gamepadRepeatMs,
    isOverlayMode,
    isRobotMode,
    isWalkMode
  ]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      const key = event.key.toLowerCase();

      if (key === photoKey) {
        togglePhotoMode();
        return;
      }

      if (key === liveKey) {
        toggleLiveMode();
        return;
      }

      if (key === robotKey) {
        toggleRobotMode();
        return;
      }

      if (key === guideKey) {
        setShowGuides((current) => !current);
        setStatusMessage('Guide overlay を切り替えました');
        return;
      }

      if (key === interactKey && isWalkMode) {
        if (!reticleTarget) {
          setStatusMessage(
            reticleCandidate
              ? `${reticleCandidate.label} に届きません。近づいてください`
              : '中央に interactable がありません'
          );
          return;
        }

        if (
          reticleTarget.kind === 'distortion-shard' ||
          reticleTarget.kind === 'dream-gate'
        ) {
          handleLoopItemActivate(reticleTarget);
          return;
        }

        handleHotspotActivate(reticleTarget);
        return;
      }

      if (key === walkKey) {
        if (isWalkMode) {
          exitWalkMode();
        } else {
          enterWalkMode();
        }
        return;
      }

      if (isRobotMode) {
        if (key === 'w' || key === 'arrowup') {
          event.preventDefault();
          moveRobot('forward');
          return;
        }

        if (key === 's' || key === 'arrowdown') {
          event.preventDefault();
          moveRobot('backward');
          return;
        }

        if (key === 'a' || key === 'arrowleft') {
          event.preventDefault();
          moveRobot('turn-left');
          return;
        }

        if (key === 'd' || key === 'arrowright') {
          event.preventDefault();
          moveRobot('turn-right');
          return;
        }

        if (key === waypointKey) {
          event.preventDefault();
          dropRobotWaypoint();
          return;
        }

        if (key === clearRouteKey) {
          event.preventDefault();
          clearRobotRoute();
          return;
        }
      }

      if (key === homeKey) {
        setSelectedPresetId(activeConfig.homePresetId);
        setStatusMessage('Home preset に戻りました');
        return;
      }

      if (key === 'escape' && activeModalItem) {
        setSelectedHotspotId(null);
        setActiveModalItem(null);
        setStatusMessage('Echo Note を閉じました');
        return;
      }

      if (key === 'escape' && isPointerLocked) {
        document.exitPointerLock?.();
        setStatusMessage('Pointer lock を解除しました');
        return;
      }

      if (key === captureKey) {
        const didCapture = downloadCanvasSnapshot();
        setStatusMessage(
          didCapture
            ? 'PNG snapshot を保存しました'
            : 'canvas がまだ準備できていません'
        );
        return;
      }

      if (key === '1' || key === '2' || key === '3') {
        const nextPreset = activeConfig.cameraPresets[Number(key) - 1];
        if (nextPreset) {
          setSelectedPresetId(nextPreset.id);
          setStatusMessage(`Camera preset: ${nextPreset.label}`);
        }
        return;
      }

      if (streamSceneKeys.includes(key)) {
        const nextStreamScene = resolvedStreamScenes[Number(key) - 4];
        if (nextStreamScene) {
          activateStreamScene(nextStreamScene);
        }
        return;
      }

      if (overlayPresetKeys.includes(key)) {
        const nextOverlayPreset = activeConfig.overlayPresets[Number(key) - 7];
        if (nextOverlayPreset) {
          activateOverlayPreset(nextOverlayPreset);
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    activeConfig.cameraPresets,
    activeConfig.homePresetId,
    activeConfig.overlayPresets,
    activeModalItem,
    isRobotMode,
    isPointerLocked,
    isWalkMode,
    reticleCandidate,
    reticleTarget,
    resolvedStreamScenes,
    selectedHotspotId
  ]);

  function handleCaptureClick() {
    const didCapture = downloadCanvasSnapshot();
    setStatusMessage(
      didCapture
        ? 'PNG snapshot を保存しました'
        : 'canvas がまだ準備できていません'
    );
  }

  function navigateToFragment(targetFragmentId) {
    if (!targetFragmentId) {
      return;
    }

    const nextConfig = resolveDreamwalkerConfig(targetFragmentId);
    setCurrentFragmentId(nextConfig.fragmentId);

    if (typeof window !== 'undefined') {
      window.location.hash = nextConfig.fragmentId;
    }
  }

  function activateStreamScene(streamScene) {
    if (!streamScene) {
      return;
    }

    if (isWalkMode) {
      exitWalkMode();
    }

    setMode('live');
    setSelectedStreamSceneId(streamScene.id);

    if (streamScene.presetId) {
      setSelectedPresetId(streamScene.presetId);
    }

    setStatusMessage(`Live Scene: ${streamScene.title}`);
  }

  function activateOverlayPreset(overlayPreset) {
    if (!overlayPreset) {
      return;
    }

    setSelectedOverlayPresetId(overlayPreset.id);
    setStatusMessage(`Overlay preset: ${overlayPreset.label}`);
  }

  async function copyLiveSceneJson() {
    try {
      await navigator.clipboard.writeText(liveScenePayloadJson);
      setStatusMessage('Live scene JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('clipboard へコピーできませんでした');
    }
  }

  function downloadLiveSceneJson() {
    downloadTextFile(
      buildSceneExportFileName(activeConfig.fragmentId, selectedStreamScene?.id),
      liveScenePayloadJson
    );
    setStatusMessage('Live scene JSON を保存しました');
  }

  function updateAssetWorkspaceField(field, value) {
    assetWorkspaceTouchedRef.current = true;
    setAssetWorkspaceImportError('');
    setAssetWorkspaceDraft((current) =>
      normalizeAssetManifest({
        ...current,
        [field]: value
      })
    );
  }

  function updateAssetWorkspaceFragmentField(fragmentId, field, value) {
    assetWorkspaceTouchedRef.current = true;
    setAssetWorkspaceImportError('');
    setAssetWorkspaceDraft((current) =>
      normalizeAssetManifest({
        ...current,
        fragments: {
          ...current.fragments,
          [fragmentId]: {
            ...(current.fragments?.[fragmentId] ?? {}),
            [field]: value
          }
        }
      })
    );
  }

  async function copyAssetWorkspaceJson() {
    try {
      await navigator.clipboard.writeText(assetWorkspaceJson);
      setStatusMessage('Asset workspace JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Asset workspace JSON をコピーできませんでした');
    }
  }

  function downloadAssetWorkspaceJson() {
    downloadTextFile('dreamwalker-live-asset-workspace.json', assetWorkspaceJson);
    setStatusMessage('Asset workspace JSON を保存しました');
  }

  function applyAssetWorkspaceManifest(nextManifest, sourceLabel) {
    assetWorkspaceTouchedRef.current = true;
    setAssetWorkspaceDraft(nextManifest);
    setAssetWorkspaceImportError('');
    setStatusMessage(`Asset workspace JSON を適用しました (${sourceLabel})`);
  }

  function applyAssetWorkspaceImportText() {
    try {
      const nextManifest = tryParseAssetManifestJson(assetWorkspaceImportText);
      applyAssetWorkspaceManifest(nextManifest, 'pasted JSON');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAssetWorkspaceImportError(message);
      setStatusMessage('Asset workspace JSON を適用できませんでした');
    }
  }

  async function handleAssetWorkspaceFileImport(event) {
    const [file] = Array.from(event.target.files ?? []);
    if (!file) {
      return;
    }

    try {
      const fileText = await file.text();
      const nextManifest = tryParseAssetManifestJson(fileText);
      setAssetWorkspaceImportText(JSON.stringify(nextManifest, null, 2));
      applyAssetWorkspaceManifest(nextManifest, file.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setAssetWorkspaceImportError(message);
      setStatusMessage(`Asset workspace file を読み込めませんでした: ${file.name}`);
    } finally {
      event.target.value = '';
    }
  }

  function saveAssetWorkspace() {
    if (typeof window === 'undefined') {
      return;
    }

    window.localStorage.setItem(assetWorkspaceStorageKey, assetWorkspaceJson);
    setAssetWorkspaceBaselineJson(JSON.stringify(assetWorkspaceDraft));
    setHasSavedAssetWorkspace(true);
    assetWorkspaceTouchedRef.current = false;
    setStatusMessage('Asset workspace を localStorage に保存しました');
  }

  function resetAssetWorkspace() {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(assetWorkspaceStorageKey);
    }

    const fallbackManifest = normalizeAssetManifest(
      assetManifestState.manifest ?? defaultAssetManifestTemplate
    );

    setAssetWorkspaceDraft(fallbackManifest);
    setAssetWorkspaceBaselineJson(JSON.stringify(fallbackManifest));
    setAssetWorkspaceImportText('');
    setAssetWorkspaceImportError('');
    setHasSavedAssetWorkspace(false);
    assetWorkspaceTouchedRef.current = false;
    setStatusMessage('Asset workspace を manifest/template 状態へ戻しました');
  }

  function updateSceneWorkspaceStreamScene(fragmentId, sceneId, updater) {
    setSceneWorkspaceImportError('');
    setSceneWorkspaceDraft((current) => {
      const currentFragment =
        current.fragments?.[fragmentId] ?? defaultSceneWorkspaceTemplate.fragments[fragmentId];
      const nextStreamScenes = (currentFragment.streamScenes ?? []).map((streamScene) =>
        streamScene.id === sceneId ? updater(streamScene) : streamScene
      );

      return normalizeSceneWorkspace({
        ...current,
        fragments: {
          ...current.fragments,
          [fragmentId]: {
            ...currentFragment,
            streamScenes: nextStreamScenes
          }
        }
      });
    });
  }

  function updateSceneWorkspaceSceneField(fragmentId, sceneId, field, value) {
    updateSceneWorkspaceStreamScene(fragmentId, sceneId, (streamScene) => ({
      ...streamScene,
      [field]: value
    }));
  }

  function updateSceneWorkspaceMemoField(fragmentId, sceneId, field, value) {
    updateSceneWorkspaceStreamScene(fragmentId, sceneId, (streamScene) => ({
      ...streamScene,
      overlayMemo: {
        ...(streamScene.overlayMemo ?? {}),
        [field]:
          field === 'items'
            ? value
                .split('\n')
                .map((item) => item.trim())
                .filter(Boolean)
            : value
      }
    }));
  }

  function updateSceneWorkspaceBrandingField(fragmentId, sceneId, field, value) {
    updateSceneWorkspaceStreamScene(fragmentId, sceneId, (streamScene) => ({
      ...streamScene,
      overlayBrandingOverrides: {
        ...(streamScene.overlayBrandingOverrides ?? {}),
        [field]: value
      }
    }));
  }

  async function copySceneWorkspaceJson() {
    try {
      await navigator.clipboard.writeText(sceneWorkspaceJson);
      setStatusMessage('Scene workspace JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Scene workspace JSON をコピーできませんでした');
    }
  }

  function downloadSceneWorkspaceJson() {
    downloadTextFile('dreamwalker-live-scene-workspace.json', sceneWorkspaceJson);
    setStatusMessage('Scene workspace JSON を保存しました');
  }

  function applySceneWorkspaceDraft(nextWorkspace, sourceLabel) {
    setSceneWorkspaceDraft(nextWorkspace);
    setSceneWorkspaceImportError('');
    setStatusMessage(`Scene workspace JSON を適用しました (${sourceLabel})`);
  }

  function applySceneWorkspaceImportText() {
    try {
      const nextWorkspace = tryParseSceneWorkspaceJson(sceneWorkspaceImportText);
      applySceneWorkspaceDraft(nextWorkspace, 'pasted JSON');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSceneWorkspaceImportError(message);
      setStatusMessage('Scene workspace JSON を適用できませんでした');
    }
  }

  async function handleSceneWorkspaceFileImport(event) {
    const [file] = Array.from(event.target.files ?? []);
    if (!file) {
      return;
    }

    try {
      const fileText = await file.text();
      const nextWorkspace = tryParseSceneWorkspaceJson(fileText);
      setSceneWorkspaceImportText(JSON.stringify(nextWorkspace, null, 2));
      applySceneWorkspaceDraft(nextWorkspace, file.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSceneWorkspaceImportError(message);
      setStatusMessage(`Scene workspace file を読み込めませんでした: ${file.name}`);
    } finally {
      event.target.value = '';
    }
  }

  function saveSceneWorkspace() {
    if (typeof window === 'undefined') {
      return;
    }

    window.localStorage.setItem(sceneWorkspaceStorageKey, sceneWorkspaceJson);
    setSceneWorkspaceBaselineJson(JSON.stringify(sceneWorkspaceDraft));
    setHasSavedSceneWorkspace(true);
    setStatusMessage('Scene workspace を localStorage に保存しました');
  }

  function resetSceneWorkspace() {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(sceneWorkspaceStorageKey);
    }

    setSceneWorkspaceDraft(defaultSceneWorkspaceTemplate);
    setSceneWorkspaceBaselineJson(JSON.stringify(defaultSceneWorkspaceTemplate));
    setSceneWorkspaceImportText('');
    setSceneWorkspaceImportError('');
    setHasSavedSceneWorkspace(false);
    setStatusMessage('Scene workspace を template 状態へ戻しました');
  }

  function applySemanticZoneDraft(nextPayload, sourceLabel) {
    const normalizedPayload = serializeSemanticZoneMap(buildSemanticZoneMap(nextPayload));

    setSemanticZoneWorkspaceDrafts((current) => ({
      ...current,
      [activeConfig.fragmentId]: normalizedPayload
    }));
    setSemanticZoneImportError('');
    if (sourceLabel !== 'editor') {
      setStatusMessage(`Semantic zone JSON を適用しました (${sourceLabel})`);
    }
  }

  function updateSemanticZoneRootField(field, value) {
    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        [field]: value
      },
      'editor'
    );
  }

  function updateSemanticZoneBoundsField(field, value) {
    const bounds = currentSemanticZonePayload.bounds ?? {};
    const normalizedValue = Number.isFinite(value) ? value : 0;
    let nextValue = normalizedValue;

    if (field === 'minX') {
      nextValue = Math.min(normalizedValue, Number(bounds.maxX ?? normalizedValue + 0.1) - 0.1);
    } else if (field === 'maxX') {
      nextValue = Math.max(normalizedValue, Number(bounds.minX ?? normalizedValue - 0.1) + 0.1);
    } else if (field === 'minZ') {
      nextValue = Math.min(normalizedValue, Number(bounds.maxZ ?? normalizedValue + 0.1) - 0.1);
    } else if (field === 'maxZ') {
      nextValue = Math.max(normalizedValue, Number(bounds.minZ ?? normalizedValue - 0.1) + 0.1);
    }

    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        bounds: {
          ...(currentSemanticZonePayload.bounds ?? {}),
          [field]: Number(nextValue.toFixed(2))
        }
      },
      'editor'
    );
  }

  function updateSemanticZoneEntry(zoneId, updater) {
    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: (currentSemanticZonePayload.zones ?? []).map((zone) =>
          zone.id === zoneId ? updater(zone) : zone
        )
      },
      'editor'
    );
  }

  function updateSemanticZoneField(zoneId, field, value) {
    updateSemanticZoneEntry(zoneId, (zone) => ({
      ...zone,
      [field]: value
    }));
  }

  function updateSemanticZoneCenterField(zoneId, axis, value) {
    updateSemanticZoneEntry(zoneId, (zone) => {
      const currentCenter = Array.isArray(zone.center) ? [...zone.center] : [0, 0, 0];
      const nextCenter = [currentCenter[0] ?? 0, currentCenter[1] ?? 0, currentCenter[2] ?? 0];
      nextCenter[axis] = value;
      return {
        ...zone,
        center: nextCenter
      };
    });
  }

  function updateSemanticZoneSizeField(zoneId, axis, value) {
    updateSemanticZoneEntry(zoneId, (zone) => {
      const currentSize = Array.isArray(zone.size) ? [...zone.size] : [1, 1];
      const nextSize = [Math.max(0.1, currentSize[0] ?? 1), Math.max(0.1, currentSize[1] ?? 1)];
      nextSize[axis] = Math.max(0.1, value);
      return {
        ...zone,
        size: nextSize
      };
    });
  }

  function updateSemanticZoneTags(zoneId, value) {
    updateSemanticZoneEntry(zoneId, (zone) => ({
      ...zone,
      tags: value
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean)
    }));
  }

  function updateSemanticZoneShape(zoneId, shape) {
    updateSemanticZoneEntry(zoneId, (zone) => ({
      ...zone,
      shape,
      size: shape === 'rect' ? zone.size ?? [2.5, 2.5] : zone.size,
      radius: shape === 'circle' ? Math.max(0.1, Number(zone.radius ?? 1.2)) : zone.radius
    }));
  }

  function addSemanticZoneEntryAtPosition(positionLike, options = {}) {
    const { sourceLabel = 'robot pose' } = options;
    const nextZone = buildDefaultSemanticZoneDraft(
      activeConfig,
      currentSemanticZonePayload.zones?.length ?? 0
    );
    nextZone.center = [positionLike[0], positionLike[1] ?? 0, positionLike[2]];

    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: [...(currentSemanticZonePayload.zones ?? []), nextZone]
      },
      sourceLabel
    );
  }

  function addSemanticZoneEntry() {
    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: [
          ...(currentSemanticZonePayload.zones ?? []),
          buildDefaultSemanticZoneDraft(activeConfig, currentSemanticZonePayload.zones?.length ?? 0)
        ]
      },
      'editor'
    );
  }

  function duplicateSemanticZoneEntry(zoneId) {
    const sourceZone = (currentSemanticZonePayload.zones ?? []).find((zone) => zone.id === zoneId);
    if (!sourceZone) {
      return;
    }

    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: [
          ...(currentSemanticZonePayload.zones ?? []),
          buildDuplicatedSemanticZoneDraft(sourceZone, currentSemanticZonePayload.zones?.length ?? 0)
        ]
      },
      `${sourceZone.label || sourceZone.id} duplicate`
    );
  }

  function addSemanticZonesFromRoute() {
    const nextZones = buildRouteSemanticZones(
      activeConfig,
      robotTrail,
      robotWaypoint,
      currentSemanticZonePayload.zones?.length ?? 0
    );

    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: [...(currentSemanticZonePayload.zones ?? []), ...nextZones]
      },
      'route batch'
    );
  }

  function fitSemanticZoneWorkspaceBounds() {
    applySemanticZoneDraft(
      fitSemanticZoneBounds(
        currentSemanticZonePayload,
        activeConfig.robotics.zoneBoundsPadding
      ),
      'fit bounds'
    );
  }

  function clearAllSemanticZones() {
    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: []
      },
      'clear all zones'
    );
  }

  function removeSemanticZoneEntry(zoneId) {
    applySemanticZoneDraft(
      {
        ...currentSemanticZonePayload,
        zones: (currentSemanticZonePayload.zones ?? []).filter((zone) => zone.id !== zoneId)
      },
      'editor'
    );
  }

  function setSemanticZoneCenterFromPosition(zoneId, positionLike, sourceLabel) {
    updateSemanticZoneEntry(zoneId, (zone) => ({
      ...zone,
      center: [positionLike[0], positionLike[1] ?? 0, positionLike[2]]
    }));
    setStatusMessage(`Semantic zone center を更新しました (${sourceLabel})`);
  }

  function assignRobotPoseToSemanticZone(zoneId) {
    setSemanticZoneCenterFromPosition(zoneId, robotPose.position, 'robot pose');
  }

  function assignWaypointToSemanticZone(zoneId) {
    if (!robotWaypoint) {
      return;
    }

    setSemanticZoneCenterFromPosition(zoneId, robotWaypoint.position, 'waypoint');
  }

  function moveRobotToSemanticZone(zoneId) {
    const zone = (currentSemanticZonePayload.zones ?? []).find((entry) => entry.id === zoneId);
    if (!zone) {
      return;
    }

    const center = Array.isArray(zone.center) ? zone.center : [0, 0, 0];
    const nextPose = {
      position: [center[0] ?? 0, center[1] ?? 0, center[2] ?? 0],
      yawDegrees: robotPose.yawDegrees
    };

    setRobotPose(nextPose);
    setRobotTrail(buildRobotTrailFromPose(nextPose));
    setStatusMessage(`Robot pose を ${zone.label || zone.id} に移動しました`);
  }

  async function copySemanticZoneWorkspaceJson() {
    try {
      await navigator.clipboard.writeText(semanticZoneWorkspaceJson);
      setStatusMessage('Semantic zone JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Semantic zone JSON をコピーできませんでした');
    }
  }

  function downloadSemanticZoneWorkspaceJson() {
    downloadTextFile(
      `dreamwalker-live-${activeConfig.fragmentId}-semantic-zones.json`,
      semanticZoneWorkspaceJson
    );
    setStatusMessage('Semantic zone JSON を保存しました');
  }

  function applySemanticZoneImportText() {
    try {
      const nextPayload = tryParseSemanticZoneJson(semanticZoneImportText);
      applySemanticZoneDraft(nextPayload, 'pasted JSON');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSemanticZoneImportError(message);
      setStatusMessage('Semantic zone JSON を適用できませんでした');
    }
  }

  async function handleSemanticZoneFileImport(event) {
    const [file] = Array.from(event.target.files ?? []);
    if (!file) {
      return;
    }

    try {
      const fileText = await file.text();
      const nextPayload = tryParseSemanticZoneJson(fileText);
      setSemanticZoneImportText(JSON.stringify(nextPayload, null, 2));
      applySemanticZoneDraft(nextPayload, file.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSemanticZoneImportError(message);
      setStatusMessage(`Semantic zone file を読み込めませんでした: ${file.name}`);
    } finally {
      event.target.value = '';
    }
  }

  function saveSemanticZoneWorkspace() {
    if (typeof window === 'undefined') {
      return;
    }

    const nextWorkspace = normalizeSemanticZoneWorkspaceMap({
      ...semanticZoneWorkspaceDrafts,
      [activeConfig.fragmentId]: currentSemanticZonePayload
    });

    window.localStorage.setItem(
      semanticZoneWorkspaceStorageKey,
      JSON.stringify(nextWorkspace)
    );
    setSemanticZoneWorkspaceDrafts(nextWorkspace);
    setSemanticZoneWorkspaceBaselineJson(JSON.stringify(nextWorkspace));
    setHasSavedSemanticZoneWorkspace(Object.keys(nextWorkspace).length > 0);
    setStatusMessage('Semantic zone workspace を localStorage に保存しました');
  }

  function resetSemanticZoneWorkspace() {
    const nextWorkspace = { ...semanticZoneWorkspaceDrafts };
    delete nextWorkspace[activeConfig.fragmentId];

    if (typeof window !== 'undefined') {
      if (Object.keys(nextWorkspace).length === 0) {
        window.localStorage.removeItem(semanticZoneWorkspaceStorageKey);
      } else {
        window.localStorage.setItem(
          semanticZoneWorkspaceStorageKey,
          JSON.stringify(nextWorkspace)
        );
      }
    }

    setSemanticZoneWorkspaceDrafts(nextWorkspace);
    setSemanticZoneWorkspaceBaselineJson(JSON.stringify(nextWorkspace));
    setSemanticZoneImportText('');
    setSemanticZoneImportError('');
    setHasSavedSemanticZoneWorkspace(Object.keys(nextWorkspace).length > 0);
    setStatusMessage('Semantic zone workspace を config 状態へ戻しました');
  }

  async function copyStudioBundleJson() {
    try {
      await navigator.clipboard.writeText(studioBundleJson);
      setStatusMessage('Studio bundle JSON を clipboard にコピーしました');
    } catch {
      setStatusMessage('Studio bundle JSON をコピーできませんでした');
    }
  }

  function downloadStudioBundleJson() {
    downloadTextFile('dreamwalker-live-studio-bundle.json', studioBundleJson);
    setStatusMessage('Studio bundle JSON を保存しました');
  }

  function persistStudioBundleShelf(nextShelf) {
    const normalizedShelf = nextShelf
      .map((entry, index) => normalizeStudioBundleShelfEntry(entry, index))
      .slice(0, 8);

    setStudioBundleShelf(normalizedShelf);

    if (typeof window !== 'undefined') {
      if (normalizedShelf.length === 0) {
        window.localStorage.removeItem(studioBundleShelfStorageKey);
      } else {
        window.localStorage.setItem(
          studioBundleShelfStorageKey,
          JSON.stringify(normalizedShelf)
        );
      }
    }
  }

  function saveStudioBundleSnapshot() {
    const snapshotLabel =
      studioBundleShelfLabel.trim() ||
      `${activeConfig.fragmentLabel} / ${selectedStreamScene?.label ?? 'Scene'}`;
    const snapshotBundle = normalizeStudioBundle(JSON.parse(studioBundleJson));
    const snapshotEntry = {
      id: `studio-bundle-${Date.now().toString(36)}`,
      label: snapshotLabel,
      bundle: snapshotBundle
    };

    persistStudioBundleShelf([
      snapshotEntry,
      ...studioBundleShelf.filter((entry) => entry.label !== snapshotLabel)
    ]);
    setStudioBundleShelfLabel('');
    setStatusMessage(`Studio bundle snapshot を保存しました: ${snapshotLabel}`);
  }

  function applyStudioBundleSnapshot(entry) {
    if (!entry) {
      return;
    }

    setStudioBundleImportText(JSON.stringify(entry.bundle, null, 2));
    applyStudioBundle(entry.bundle, `snapshot:${entry.label}`);
  }

  function downloadStudioBundleSnapshot(entry) {
    if (!entry) {
      return;
    }

    const safeLabel = entry.label.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    downloadTextFile(
      `dreamwalker-live-${safeLabel || 'studio-bundle'}.json`,
      JSON.stringify(entry.bundle, null, 2)
    );
    setStatusMessage(`Studio bundle snapshot を保存しました: ${entry.label}`);
  }

  function deleteStudioBundleSnapshot(entryId) {
    persistStudioBundleShelf(
      studioBundleShelf.filter((entry) => entry.id !== entryId)
    );
    setStatusMessage('Studio bundle snapshot を削除しました');
  }

  function clearStudioBundleShelf() {
    persistStudioBundleShelf([]);
    setStatusMessage('Studio bundle shelf を空にしました');
  }

  function buildStudioBundleLaunchUrl(bundleUrl) {
    if (typeof window === 'undefined') {
      return `?studioBundle=${encodeURIComponent(bundleUrl)}`;
    }

    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set(dreamwalkerConfig.studioBundle.queryParam, bundleUrl);
    nextUrl.searchParams.delete('overlay');
    nextUrl.hash = '';
    return nextUrl.toString();
  }

  function buildRobotRouteLaunchUrl(routeUrl) {
    if (typeof window === 'undefined') {
      return `?robotRoute=${encodeURIComponent(routeUrl)}`;
    }

    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set(dreamwalkerConfig.robotRoute.queryParam, routeUrl);
    nextUrl.searchParams.delete('overlay');
    nextUrl.hash = '';
    return nextUrl.toString();
  }

  function buildRobotMissionLaunchUrl(missionUrl) {
    if (typeof window === 'undefined') {
      return `?robotMission=${encodeURIComponent(missionUrl)}`;
    }

    const nextUrl = new URL(window.location.href);
    nextUrl.searchParams.set(dreamwalkerConfig.robotMission.queryParam, missionUrl);
    nextUrl.searchParams.delete(dreamwalkerConfig.robotRoute.queryParam);
    nextUrl.searchParams.delete('overlay');
    nextUrl.hash = '';
    return nextUrl.toString();
  }

  async function applyPublicStudioBundle(entry) {
    if (!entry?.url) {
      return;
    }

    try {
      const response = await fetch(entry.url, {
        cache: 'no-store',
        headers: {
          Accept: 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const bundle = await response.json();
      applyStudioBundle(normalizeStudioBundle(bundle), `catalog:${entry.label}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatusMessage(`Public studio bundle を適用できませんでした: ${message}`);
    }
  }

  function openPublicStudioBundle(entry) {
    if (!entry?.url || typeof window === 'undefined') {
      return;
    }

    window.location.assign(buildStudioBundleLaunchUrl(entry.url));
  }

  async function copyPublicStudioBundleLaunchUrl(entry) {
    if (!entry?.url) {
      return;
    }

    try {
      await navigator.clipboard.writeText(buildStudioBundleLaunchUrl(entry.url));
      setStatusMessage(`Bundle launch URL をコピーしました: ${entry.label}`);
    } catch {
      setStatusMessage('Bundle launch URL をコピーできませんでした');
    }
  }

  async function applyPublicRobotRoute(entry) {
    if (!entry?.url) {
      return;
    }

    try {
      const response = await fetch(entry.url, {
        cache: 'no-store',
        headers: {
          Accept: 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const route = normalizeRobotRoutePayload(await response.json());
      applyRobotRoutePayload(route, `catalog:${entry.label}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatusMessage(`Public robot route を適用できませんでした: ${message}`);
    }
  }

  async function applyPublicRobotMission(entry) {
    if (!entry?.url) {
      return;
    }

    try {
      const { mission, route } = await loadRobotMissionResource(entry.url);
      applyRobotMissionPayload(mission, route, `mission:${entry.label}`, entry.url);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatusMessage(`Public robot mission を適用できませんでした: ${message}`);
    }
  }

  function openPublicRobotRoute(entry) {
    if (!entry?.url || typeof window === 'undefined') {
      return;
    }

    window.location.assign(buildRobotRouteLaunchUrl(entry.url));
  }

  function openPublicRobotMission(entry) {
    if (!entry?.url || typeof window === 'undefined') {
      return;
    }

    window.location.assign(buildRobotMissionLaunchUrl(entry.url));
  }

  async function copyPublicRobotRouteLaunchUrl(entry) {
    if (!entry?.url) {
      return;
    }

    try {
      await navigator.clipboard.writeText(buildRobotRouteLaunchUrl(entry.url));
      setStatusMessage(`Robot route launch URL をコピーしました: ${entry.label}`);
    } catch {
      setStatusMessage('Robot route launch URL をコピーできませんでした');
    }
  }

  async function copyPublicRobotMissionLaunchUrl(entry) {
    if (!entry?.url) {
      return;
    }

    try {
      await navigator.clipboard.writeText(buildRobotMissionLaunchUrl(entry.url));
      setStatusMessage(`Robot mission launch URL をコピーしました: ${entry.label}`);
    } catch {
      setStatusMessage('Robot mission launch URL をコピーできませんでした');
    }
  }

  function applyStudioBundle(nextBundle, sourceLabel) {
    setAssetWorkspaceDraft(nextBundle.assetWorkspace);
    setSceneWorkspaceDraft(nextBundle.sceneWorkspace);
    setSemanticZoneWorkspaceDrafts(nextBundle.semanticZoneWorkspace ?? {});
    setAssetWorkspaceBaselineJson(JSON.stringify(nextBundle.assetWorkspace));
    setSceneWorkspaceBaselineJson(JSON.stringify(nextBundle.sceneWorkspace));
    setSemanticZoneWorkspaceBaselineJson(JSON.stringify(nextBundle.semanticZoneWorkspace ?? {}));
    setStudioBundleImportText(JSON.stringify(nextBundle, null, 2));
    setAssetWorkspaceImportError('');
    setSceneWorkspaceImportError('');
    setSemanticZoneImportError('');
    setStudioBundleImportError('');
    setHasSavedAssetWorkspace(false);
    setHasSavedSceneWorkspace(false);
    setHasSavedSemanticZoneWorkspace(false);
    assetWorkspaceTouchedRef.current = false;
    pendingStudioStateRef.current = nextBundle.state ?? null;

    if (
      nextBundle.state?.fragmentId &&
      nextBundle.state.fragmentId !== activeConfig.fragmentId
    ) {
      if (nextBundle.robotRoute) {
        applyRobotRoutePayload(nextBundle.robotRoute, `${sourceLabel}/robotRoute`);
      }
      navigateToFragment(nextBundle.state.fragmentId);
    } else {
      applyStudioState(nextBundle.state, activeConfig, resolvedStreamScenes);
      if (nextBundle.robotRoute) {
        applyRobotRoutePayload(nextBundle.robotRoute, `${sourceLabel}/robotRoute`);
      }
      pendingStudioStateRef.current = null;
    }

    setStatusMessage(`Studio bundle を適用しました (${sourceLabel})`);
  }

  function applyStudioBundleImportText() {
    try {
      const nextBundle = tryParseStudioBundleJson(studioBundleImportText);
      applyStudioBundle(nextBundle, 'pasted JSON');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStudioBundleImportError(message);
      setStatusMessage('Studio bundle JSON を適用できませんでした');
    }
  }

  async function handleStudioBundleFileImport(event) {
    const [file] = Array.from(event.target.files ?? []);
    if (!file) {
      return;
    }

    try {
      const fileText = await file.text();
      const nextBundle = tryParseStudioBundleJson(fileText);
      setStudioBundleImportText(JSON.stringify(nextBundle, null, 2));
      applyStudioBundle(nextBundle, file.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStudioBundleImportError(message);
      setStatusMessage(`Studio bundle file を読み込めませんでした: ${file.name}`);
    } finally {
      event.target.value = '';
    }
  }

  async function copyOverlayUrl() {
    try {
      await navigator.clipboard.writeText(overlayUrl);
      setStatusMessage('Overlay URL を clipboard にコピーしました');
    } catch {
      setStatusMessage('Overlay URL をコピーできませんでした');
    }
  }

  function openOverlayView() {
    if (typeof window === 'undefined') {
      return;
    }

    window.open(overlayUrl, '_blank', 'noopener,noreferrer');
    setStatusMessage('Overlay view を新しいタブで開きました');
  }

  function handleHotspotActivate(hotspot) {
    if (!hotspot) {
      return;
    }

    if (isPointerLocked) {
      document.exitPointerLock?.();
    }

    if (hotspot.presetId) {
      setSelectedPresetId(hotspot.presetId);
    }

    setSelectedHotspotId(hotspot.id);
    setActiveModalItem(hotspot);
    setStatusMessage(`${hotspot.label} を開きました`);
  }

  function handleLoopItemActivate(item) {
    if (!item) {
      return;
    }

    if (isPointerLocked) {
      document.exitPointerLock?.();
    }

    if (item.kind === 'distortion-shard') {
      if (collectedShardIds.includes(item.id)) {
        return;
      }

      const nextCollected = [...collectedShardIds, item.id];
      const nextRemaining = Math.max(0, totalShardCount - nextCollected.length);

      setCollectedShardIds(nextCollected);
      setStatusMessage(
        nextRemaining === 0
          ? `${item.label} を回収。DreamGate が開きました`
          : `${item.label} を回収 (${nextCollected.length}/${totalShardCount})`
      );
      return;
    }

    if (item.kind === 'dream-gate') {
      if (item.presetId) {
        setSelectedPresetId(item.presetId);
      }

      if (item.isGateUnlocked && item.targetFragmentId) {
        navigateToFragment(item.targetFragmentId);
        setStatusMessage(
          `DreamGate から ${item.targetFragmentLabel ?? item.targetFragmentId} へ移動しました`
        );
        return;
      }

      setActiveModalItem(item);
      setStatusMessage(
        item.isGateUnlocked
          ? 'DreamGate は開いています'
          : `DreamGate は閉ざされています。あと ${remainingShardCount} 個`
      );
    }
  }

  function handleMapPositionChange(nextPositionState) {
    const nextPosition = nextPositionState?.position;
    if (!nextPosition) {
      return;
    }

    const roundedPosition = {
      x: Number(Number(nextPosition.x).toFixed(2)),
      y: Number(Number(nextPosition.y).toFixed(2)),
      z: Number(Number(nextPosition.z).toFixed(2))
    };

    if (![roundedPosition.x, roundedPosition.y, roundedPosition.z].every(Number.isFinite)) {
      return;
    }

    setMapPositionState((current) => {
      const currentPosition = current.position;
      const source = nextPositionState.source ?? 'camera';
      const unchanged =
        current.source === source &&
        currentPosition &&
        currentPosition.x === roundedPosition.x &&
        currentPosition.y === roundedPosition.y &&
        currentPosition.z === roundedPosition.z;

      if (unchanged) {
        return current;
      }

      return {
        status: 'ready',
        source,
        position: roundedPosition,
        timestamp: nextPositionState.timestamp ?? Date.now()
      };
    });
  }

  function resetDreamState() {
    setCollectedShardIds([]);
    setActiveModalItem(null);
    setSelectedHotspotId(null);
    setStatusMessage('DreamWalker state をリセットしました');
  }

  if (isOverlayMode) {
    return <ObsOverlayView overlayState={overlayState} />;
  }

  return (
    <div className={`dreamwalker-shell mode-${mode} filter-${selectedFilter.id}`}>
      <div className="dreamwalker-stage">
        <Suspense
          fallback={
            <div className="empty-world-banner">
              <span className="empty-world-badge">Loading Scene</span>
              <p>DreamWalker stage を起動しています。</p>
            </div>
          }>
          <DreamwalkerScene
            worldConfig={activeWorldConfig}
            cameraMode={cameraMode}
            currentPreset={currentPreset}
            hotspots={activeConfig.hotspots}
            loopItems={loopItems}
            robotPoints={robotProjectionPoints}
            robotRoutePoints={robotRouteProjectionPoints}
            benchmarkRoutePoints={benchmarkRouteProjectionPoints}
            semanticZonePoints={semanticZoneProjectionPoints}
            semanticZoneSurfacePoints={semanticZoneSurfacePoints}
            roboticsCamera={roboticsCamera}
            onColliderStatusChange={setWalkColliderStatus}
            onPresetApplied={(label) => setStatusMessage(`Camera preset: ${label}`)}
            onHotspotsProjected={setProjectedHotspots}
            onLoopItemsProjected={setProjectedLoopItems}
            onRobotPointsProjected={setProjectedRobotPoints}
            onRobotRoutePointsProjected={setProjectedRobotRoutePoints}
            onBenchmarkRoutePointsProjected={setProjectedBenchmarkRoutePoints}
            onSemanticZonePointsProjected={setProjectedSemanticZonePoints}
            onSemanticZoneSurfacePointsProjected={setProjectedSemanticZoneSurfacePoints}
            onFrame={shouldStreamRobotFrames ? handleRobotFrame : undefined}
            onDepthFrame={shouldStreamRobotDepthFrames ? handleRobotDepthFrame : undefined}
            dynamicMapPositionTrackingEnabled={
              Boolean(dynamicMapLoadPlan.tileCatalog) && !selectedTileId
            }
            onMapPositionChange={handleMapPositionChange}
            onSplatStatusChange={setMapRuntimeState}
            splatUrl={effectiveSplatUrl}
          />
        </Suspense>

        {!effectiveSplatUrl ? (
          <div className="empty-world-banner">
            <span className="empty-world-badge">No Splat Configured</span>
            <p>
              `public/manifests/dreamwalker-live.assets.json` の `splatUrl` を埋めるか、
              `?assetManifest=` で別 manifest を渡すと DreamWalker world が表示されます。
            </p>
          </div>
        ) : null}

        {effectiveSplatUrl && !isWalkMode && !isRobotMode ? (
          <HotspotOverlay
            hotspots={visibleHotspots}
            onActivate={handleHotspotActivate}
          />
        ) : null}

        {effectiveSplatUrl && !isWalkMode && !isRobotMode ? (
          <HotspotOverlay
            hotspots={visibleLoopItems}
            onActivate={handleLoopItemActivate}
          />
        ) : null}

        {effectiveSplatUrl && isRobotMode && !isWalkMode ? (
          <RobotRouteOverlay points={visibleRobotRoutePoints} />
        ) : null}

        {effectiveSplatUrl && isRobotMode && !isWalkMode ? (
          <BenchmarkRouteOverlay
            benchmark={sim2realBenchmarkOverlay}
            points={visibleBenchmarkRoutePoints}
          />
        ) : null}

        {effectiveSplatUrl && isRobotMode && !isWalkMode ? (
          <SemanticZoneSurfaceOverlay zones={visibleSemanticZoneSurfaces} />
        ) : null}

        {effectiveSplatUrl && isRobotMode && !isWalkMode ? (
          <SemanticZoneOverlay points={visibleSemanticZonePoints} />
        ) : null}

        {effectiveSplatUrl && isRobotMode && !isWalkMode ? (
          <RoboticsOverlay points={visibleRobotPoints} />
        ) : null}

        {isPhotoMode && showGuides ? (
          <div className="photo-guides">
            <div className="photo-frame" style={guideStyle}>
              <div className="guide-lines guide-lines-x" />
              <div className="guide-lines guide-lines-y" />
            </div>
          </div>
        ) : null}

        {isLiveMode ? (
          <div
            className={`stream-safe-overlay overlay-layout-${selectedOverlayPreset.id}`}
            data-overlay-branding={resolvedLiveOverlayBranding.id}
            data-overlay-preset={selectedOverlayPreset.id}>
            <div className="safe-title">STREAM SAFE AREA</div>
            <OverlayStage
              overlayState={liveScenePayload}
              overlayBranding={resolvedLiveOverlayBranding}
              overlayPreset={selectedOverlayPreset}
              preview
            />
          </div>
        ) : null}

        {isWalkMode ? (
          <div className="walk-hud glass-panel">
            <span className="walk-badge">WALK MODE</span>
            <span>WASD: move</span>
            <span>Mouse: look</span>
            <span>Space: jump</span>
            <span>Shift: sprint</span>
            <span>F: interact</span>
            <span>{isPointerLocked ? 'Esc: unlock cursor' : 'Click: lock cursor'}</span>
          </div>
        ) : null}

        {isRobotMode && !isWalkMode ? (
          <div className="robotics-hud glass-panel">
            <span className="walk-badge">ROBOT MODE</span>
            <span>W / S: move</span>
            <span>A / D: turn</span>
            <span>V: waypoint</span>
            <span>C: clear route</span>
            <span>{selectedRobotCamera?.label ?? 'Robot Camera'}</span>
            <span>Bridge {robotBridgeStatusLabel}</span>
            <span>Pad {gamepadStatusLabel}</span>
            <span>{robotNodeLabel} / {robotTrailDistance} m</span>
            <span>{robotWaypoint ? `Waypoint ${robotWaypointDistance} m` : 'Waypoint none'}</span>
            <span>Zone {semanticZoneCurrentLabel}</span>
          </div>
        ) : null}

        {isWalkMode ? (
          <div className="reticle-overlay" aria-live="polite">
            <div
              className={`reticle-shell${reticleTarget ? ' active' : ''}${!reticleTarget && reticleCandidate ? ' distant' : ''}`}>
              <span className="reticle-dot" />
            </div>
            <div className="reticle-label">{getReticleHint(reticleTarget, reticleCandidate)}</div>
          </div>
        ) : null}
      </div>

      <header className="topbar">
        <div>
          <p className="eyebrow">Browser Prototype / {activeConfig.fragmentLabel}</p>
          <h1>{dreamwalkerConfig.appTitle}</h1>
          <p className="subtitle">{dreamwalkerConfig.subtitle}</p>
        </div>
        <div className="mode-switches">
          <button
            className={isWalkMode ? 'active walk-toggle' : 'walk-toggle'}
            onClick={() => {
              if (isWalkMode) {
                exitWalkMode();
              } else {
                enterWalkMode();
              }
            }}
            type="button">
            Walk
          </button>
          <button
            className={isRobotMode ? 'active robot-toggle' : 'robot-toggle'}
            onClick={toggleRobotMode}
            type="button">
            Robot
          </button>
          <button
            className={mode === 'explore' ? 'active' : ''}
            onClick={setExploreMode}
            type="button">
            Explore
          </button>
          <button
            className={isPhotoMode ? 'active' : ''}
            onClick={togglePhotoMode}
            type="button">
            Photo
          </button>
          <button
            className={isLiveMode ? 'active' : ''}
            onClick={toggleLiveMode}
            type="button">
            Live
          </button>
        </div>
      </header>

      <aside className="left-panel glass-panel">
        <h2>World Asset</h2>
        <p className="panel-value">{assetBundle.assetLabel}</p>
        <p className="panel-note">
          {assetBundle.worldNote || 'fragment ごとに Marble world を差し替えられる構成です。'}
        </p>
        <div className="state-grid">
          <div className="state-card">
            <span className="state-label">Splat</span>
            <strong>{splatAssetSourceLabel}</strong>
          </div>
          <div className="state-card">
            <span className="state-label">Collider</span>
            <strong>{colliderAssetSourceLabel}</strong>
          </div>
        </div>
        <div className="state-card">
          <span className="state-label">World Health</span>
          <div className="status-row">
            <HealthBadge health={activeWorldHealth} />
            <strong>
              {activeWorldHealth.status === 'ready'
                ? '配信投入可能'
                : activeWorldHealth.status === 'warning'
                  ? '要確認'
                  : '要修正'}
            </strong>
          </div>
          <p className={activeWorldHealth.status === 'error' ? 'panel-note panel-note-error' : 'panel-note'}>
            {activeWorldHealth.detail}
          </p>
        </div>

        <h2>Dynamic Map Loading</h2>
        <div className="state-card">
          <span className="state-label">Runtime Map</span>
          <div className="status-row">
            <HealthBadge health={dynamicMapRuntimeHealth} />
            <strong>{dynamicMapLoadPlan.active?.fragmentLabel ?? activeConfig.fragmentLabel}</strong>
          </div>
          <p className="panel-note">{dynamicMapRuntimeHealth.detail}</p>
          <p className="panel-note">
            {mapRuntimeState.url
              ? `runtime splat: ${mapRuntimeState.url}`
              : 'runtime splat 未設定'}
          </p>
        </div>
        <div className="state-grid">
          <div className="state-card">
            <span className="state-label">Active Fragment</span>
            <strong>{dynamicMapLoadPlan.active?.assetLabel ?? assetBundle.assetLabel}</strong>
            <span className="panel-note">
              {dynamicMapLoadPlan.active?.splatAssetKind ?? 'missing'} /{' '}
              {dynamicMapLoadPlan.active?.splatSource ?? 'missing'}
            </span>
          </div>
          <div className="state-card">
            <span className="state-label">Gate Candidate</span>
            <strong>{nextDynamicMapCandidate?.fragmentLabel ?? 'none'}</strong>
            <span className="panel-note">
              {nextDynamicMapCandidate
                ? `${nextDynamicMapCandidate.splatAssetKind} / ${nextDynamicMapCandidate.splatSource}`
                : 'gate 先 map なし'}
            </span>
          </div>
        </div>
        <div className="state-card">
          <span className="state-label">Tile Catalog</span>
          <div className="status-row">
            <HealthBadge health={tileCatalogHealth} />
            <strong>{dynamicMapLoadPlan.tileCatalog?.label ?? 'none'}</strong>
          </div>
          <p className="panel-note">
            {tileCatalogStatusLabel}
            {tileCatalogState.error ? `: ${tileCatalogState.error}` : ''}
          </p>
          <p className="panel-note">
            {dynamicMapLoadPlan.tileCatalog
              ? `${dynamicMapLoadPlan.tileCatalog.summary.readyTileCount} ready / ${dynamicMapLoadPlan.tileCatalog.summary.tileCount} tiles`
              : 'tile catalog 未接続'}
          </p>
          {activeDynamicMapTile ? (
            <p className="panel-note">
              Active tile: {activeDynamicMapTile.id}
              {selectedTileId ? ` / requested ${selectedTileId}` : ''}
            </p>
          ) : null}
          <p className="panel-note">
            Selection: {selectedTileId ? 'manual tileId' : 'auto position'}
          </p>
          <p className="panel-note">
            Preload limit: {dynamicMapTilePreloadLimit}
          </p>
          <p className="panel-note">
            Resident limit: {dynamicMapLoadPlan.maxResidentTiles}
          </p>
          <p className="panel-note">
            {effectiveMapPositionState.position
              ? `Position ${effectiveMapPositionState.source}: ${effectiveMapPositionState.position.x}, ${effectiveMapPositionState.position.y}, ${effectiveMapPositionState.position.z}`
              : 'position tracking 待機中'}
          </p>
          {dynamicMapLoadPlan.tilePreloadCandidates.length > 0 ? (
            <p className="panel-note">
              Preload tiles: {dynamicMapLoadPlan.tilePreloadCandidates
                .map((entry) => entry.tileId)
                .join(', ')}
            </p>
          ) : null}
          {dynamicMapLoadPlan.tileEvictionCandidates.length > 0 ? (
            <p className="panel-note">
              Evicted tiles: {dynamicMapLoadPlan.tileEvictionCandidates
                .map((entry) => entry.tileId)
                .join(', ')}
            </p>
          ) : null}
          {tileResidencySummary ? (
            <p className="panel-note">
              Tile residency: {tileResidencySummary}
            </p>
          ) : null}
          {tileResidencyPreview ? (
            <p className="panel-note">
              Tile residency list: {tileResidencyPreview}
            </p>
          ) : null}
          <div className="status-row">
            <HealthBadge health={tilePreloadHealth} />
            <strong>{tilePreloadState.mode}</strong>
          </div>
          <p className="panel-note">{tilePreloadHealth.detail}</p>
          {tilePreloadState.results.length > 0 ? (
            <p className="panel-note">
              Preload status: {tilePreloadState.results
                .map((result, index) =>
                  `${tilePreloadState.entries[index]?.tileId ?? result.fragmentLabel ?? result.label}: ${result.status}${result.cached ? ' cached' : ''}`
                )
                .join(' / ')}
            </p>
          ) : null}
          <p className="panel-note">
            {tileCatalogState.url || 'tileCatalog query 未指定'}
          </p>
        </div>
        <p className="panel-note">
          {effectiveSplatUrl ? `Active splat: ${effectiveSplatUrl}` : 'splat asset 未設定'}
        </p>
        {assetBundle.colliderMeshUrl ? (
          <p className="panel-note">Active collider: {assetBundle.colliderMeshUrl}</p>
        ) : null}
        {!assetBundle.hasConfiguredSplat && assetBundle.expectedSplatUrl ? (
          <p className="panel-note">推奨配置: {assetBundle.expectedSplatUrl}</p>
        ) : null}
        {!assetBundle.hasColliderMesh && assetBundle.expectedColliderMeshUrl ? (
          <p className="panel-note">推奨 collider: {assetBundle.expectedColliderMeshUrl}</p>
        ) : null}

        <h2>Asset Manifest</h2>
        <p className="panel-value">
          {assetManifestState.manifest?.label ?? 'Asset manifest optional'}
        </p>
        <p className="panel-note">
          {assetManifestStatusLabel}
          {assetManifestState.error ? `: ${assetManifestState.error}` : ''}
        </p>
        <p className="panel-note">
          {assetManifestState.url || 'assetManifest query 未指定'}
        </p>
        {assetManifestState.manifest?.note ? (
          <p className="panel-note">{assetManifestState.manifest.note}</p>
        ) : null}
        <p className="panel-note">
          `?assetManifest=/manifests/custom-world.json` で別 manifest を差し替えできます。
        </p>
        <p className="panel-note">
          {hasSavedAssetWorkspace
            ? '現在は localStorage の workspace が優先されています。'
            : 'workspace を保存していなければ、manifest file をそのまま参照します。'}
        </p>

        <h2>Studio Bundle Source</h2>
        <p className="panel-value">
          {studioBundleState.bundle?.label ?? 'Studio bundle optional'}
        </p>
        <p className="panel-note">
          {studioBundleStatusLabel}
          {studioBundleState.error ? `: ${studioBundleState.error}` : ''}
        </p>
        <p className="panel-note">
          {studioBundleState.url || 'studioBundle query 未指定'}
        </p>
        {studioBundleState.bundle?.note ? (
          <p className="panel-note">{studioBundleState.bundle.note}</p>
        ) : null}
        <p className="panel-note">
          `?studioBundle=/studio-bundles/dreamwalker-live.sample.json` で配信セットをそのまま起動できます。
        </p>

        <h2>Public Studio Bundles</h2>
        <p className="panel-value">
          {publicStudioBundleCatalog?.label ?? 'Public bundle catalog optional'}
        </p>
        <p className="panel-note">
          {studioBundleCatalogStatusLabel}
          {studioBundleCatalogState.error ? `: ${studioBundleCatalogState.error}` : ''}
        </p>
        <p className="panel-note">
          {studioBundleCatalogState.url || 'studioBundleCatalog query 未指定'}
        </p>
        {publicStudioBundleCatalog?.note ? (
          <p className="panel-note">{publicStudioBundleCatalog.note}</p>
        ) : null}
        {publicStudioBundleCatalog?.bundles?.length ? (
          <div className="button-stack">
            {publicStudioBundleCatalog.bundles.map((entry) => (
              <div key={entry.id} className="state-card">
                <span className="state-label">{entry.label}</span>
                <div className="status-row">
                  <strong>{entry.fragmentId || 'DreamWalker Live'}</strong>
                  <HealthBadge health={publicStudioBundleHealthMap[entry.id]} />
                </div>
                {entry.description ? (
                  <p className="panel-note">{entry.description}</p>
                ) : null}
                {publicStudioBundleHealthMap[entry.id]?.detail ? (
                  <p
                    className={
                      publicStudioBundleHealthMap[entry.id].status === 'error'
                        ? 'panel-note panel-note-error'
                        : 'panel-note'
                    }>
                    {publicStudioBundleHealthMap[entry.id].detail}
                  </p>
                ) : null}
                <div className="chip-list">
                  <button
                    className="chip"
                    disabled={publicStudioBundleHealthMap[entry.id]?.status === 'error'}
                    onClick={() => applyPublicStudioBundle(entry)}
                    type="button">
                    Apply
                  </button>
                  <button
                    className="chip"
                    disabled={publicStudioBundleHealthMap[entry.id]?.status === 'error'}
                    onClick={() => openPublicStudioBundle(entry)}
                    type="button">
                    Launch
                  </button>
                  <button
                    className="chip"
                    onClick={() => copyPublicStudioBundleLaunchUrl(entry)}
                    type="button">
                    Copy URL
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="panel-note">
            `public/studio-bundles/index.json` に bundle を並べると、ここから直接選べます。
          </p>
        )}

        <h2>Public Robot Missions</h2>
        <p className="panel-value">
          {publicRobotMissionCatalog?.label ?? 'Public robot mission catalog optional'}
        </p>
        <p className="panel-note">
          {robotMissionCatalogStatusLabel}
          {robotMissionCatalogState.error ? `: ${robotMissionCatalogState.error}` : ''}
        </p>
        <p className="panel-note">
          {robotMissionCatalogState.url || 'robotMissionCatalog query 未指定'}
        </p>
        <p className="panel-note">
          {robotMissionStatusLabel}
          {robotMissionState.error ? `: ${robotMissionState.error}` : ''}
        </p>
        <p className="panel-note">
          {robotMissionState.url || 'robotMission query 未指定'}
        </p>
        {robotMissionState.mission?.routeUrl ? (
          <p className="panel-note">Mission route: {robotMissionState.mission.routeUrl}</p>
        ) : null}
        {robotMissionState.mission?.zoneMapUrl ? (
          <p className="panel-note">Mission zones: {robotMissionState.mission.zoneMapUrl}</p>
        ) : null}
        {robotMissionState.mission?.cameraPresetId ? (
          <p className="panel-note">Mission preset: {robotMissionState.mission.cameraPresetId}</p>
        ) : null}
        {robotMissionState.mission?.robotCameraId ? (
          <p className="panel-note">Mission robot camera: {robotMissionState.mission.robotCameraId}</p>
        ) : null}
        {robotMissionState.mission?.streamSceneId ? (
          <p className="panel-note">Mission stream scene: {robotMissionState.mission.streamSceneId}</p>
        ) : null}
        {robotMissionState.mission?.startupMode ? (
          <p className="panel-note">Mission startup mode: {robotMissionState.mission.startupMode}</p>
        ) : null}
        {activeRobotMissionHealth?.detail ? (
          <p
            className={
              activeRobotMissionHealth.status === 'error'
                ? 'panel-note panel-note-error'
                : 'panel-note'
            }>
            {activeRobotMissionHealth.label}: {activeRobotMissionHealth.detail}
          </p>
        ) : null}
        {publicRobotMissionCatalog?.note ? (
          <p className="panel-note">{publicRobotMissionCatalog.note}</p>
        ) : null}
        {publicRobotMissionCatalog?.missions?.length ? (
          <div className="button-stack">
            {publicRobotMissionCatalog.missions.map((entry) => (
              <div key={entry.id} className="state-card">
                <span className="state-label">{entry.label}</span>
                <div className="status-row">
                  <strong>{entry.fragmentId || 'Robot Mission'}</strong>
                  <div className="status-row-badges">
                    <HealthBadge health={publicRobotMissionHealthMap[entry.id]} />
                    {entry.accent ? (
                      <span className="chip" style={{ borderColor: entry.accent, color: entry.accent }}>
                        mission
                      </span>
                    ) : null}
                  </div>
                </div>
                {entry.description ? (
                  <p className="panel-note">{entry.description}</p>
                ) : null}
                {publicRobotMissionHealthMap[entry.id]?.detail ? (
                  <p
                    className={
                      publicRobotMissionHealthMap[entry.id].status === 'error'
                        ? 'panel-note panel-note-error'
                        : 'panel-note'
                    }>
                    {publicRobotMissionHealthMap[entry.id].detail}
                  </p>
                ) : null}
                <div className="chip-list">
                  <button
                    className="chip"
                    disabled={publicRobotMissionHealthMap[entry.id]?.status === 'error'}
                    onClick={() => applyPublicRobotMission(entry)}
                    type="button">
                    Apply
                  </button>
                  <button
                    className="chip"
                    disabled={publicRobotMissionHealthMap[entry.id]?.status === 'error'}
                    onClick={() => openPublicRobotMission(entry)}
                    type="button">
                    Launch
                  </button>
                  <button
                    className="chip"
                    onClick={() => copyPublicRobotMissionLaunchUrl(entry)}
                    type="button">
                    Copy URL
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="panel-note">
            `public/robot-missions/index.json` に mission manifest を並べると、ここから直接選べます。
          </p>
        )}

        <h2>Public Robot Routes</h2>
        <p className="panel-value">
          {publicRobotRouteCatalog?.label ?? 'Public robot route catalog optional'}
        </p>
        <p className="panel-note">
          {robotRouteCatalogStatusLabel}
          {robotRouteCatalogState.error ? `: ${robotRouteCatalogState.error}` : ''}
        </p>
        <p className="panel-note">
          {robotRouteCatalogState.url || 'robotRouteCatalog query 未指定'}
        </p>
        <p className="panel-note">
          {robotRouteStatusLabel}
          {robotRouteState.error ? `: ${robotRouteState.error}` : ''}
        </p>
        <p className="panel-note">
          {robotRouteState.url || 'robotRoute query 未指定'}
        </p>
        <p className="panel-note">
          Robot route playback: {robotRoutePlaybackConfig.enabled
            ? `${robotRoutePlaybackConfig.intervalMs} ms${robotRoutePlaybackConfig.loop ? ' / loop' : ''}`
            : 'off'}
        </p>
        {activeRobotRouteHealth?.detail ? (
          <p
            className={
              activeRobotRouteHealth.status === 'error'
                ? 'panel-note panel-note-error'
                : 'panel-note'
            }>
            {activeRobotRouteHealth.label}: {activeRobotRouteHealth.detail}
          </p>
        ) : null}
        {publicRobotRouteCatalog?.note ? (
          <p className="panel-note">{publicRobotRouteCatalog.note}</p>
        ) : null}
        {publicRobotRouteCatalog?.routes?.length ? (
          <div className="button-stack">
            {publicRobotRouteCatalog.routes.map((entry) => (
              <div key={entry.id} className="state-card">
                <span className="state-label">{entry.label}</span>
                <div className="status-row">
                  <strong>{entry.fragmentId || 'Robot Route'}</strong>
                  <div className="status-row-badges">
                    <HealthBadge health={publicRobotRouteHealthMap[entry.id]} />
                    {entry.accent ? (
                      <span className="chip" style={{ borderColor: entry.accent, color: entry.accent }}>
                        preset
                      </span>
                    ) : null}
                  </div>
                </div>
                {entry.description ? (
                  <p className="panel-note">{entry.description}</p>
                ) : null}
                {publicRobotRouteHealthMap[entry.id]?.detail ? (
                  <p
                    className={
                      publicRobotRouteHealthMap[entry.id].status === 'error'
                        ? 'panel-note panel-note-error'
                        : 'panel-note'
                    }>
                    {publicRobotRouteHealthMap[entry.id].detail}
                  </p>
                ) : null}
                <div className="chip-list">
                  <button
                    className="chip"
                    disabled={publicRobotRouteHealthMap[entry.id]?.status === 'error'}
                    onClick={() => applyPublicRobotRoute(entry)}
                    type="button">
                    Apply
                  </button>
                  <button
                    className="chip"
                    disabled={publicRobotRouteHealthMap[entry.id]?.status === 'error'}
                    onClick={() => openPublicRobotRoute(entry)}
                    type="button">
                    Launch
                  </button>
                  <button className="chip" onClick={() => copyPublicRobotRouteLaunchUrl(entry)} type="button">
                    Copy URL
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="panel-note">
            `public/robot-routes/index.json` に route preset を並べると、ここから直接選べます。
          </p>
        )}

        <h2>Asset Workspace</h2>
        <p className="panel-value">
          {assetWorkspaceModeLabel}
          {isAssetWorkspaceDirty ? ' / Unsaved Draft' : ''}
        </p>
        <p className="panel-note">
          入力値はこの画面ですぐ preview に反映されます。`Save Asset Workspace` で永続化されます。
        </p>
        <div className="chip-list">
          {Object.values(dreamwalkerConfig.fragments).map((fragment) => (
            <button
              key={fragment.fragmentId}
              className={fragment.fragmentId === activeConfig.fragmentId ? 'chip active' : 'chip'}
              onClick={() => navigateToFragment(fragment.fragmentId)}
              type="button">
              {fragment.fragmentLabel}
            </button>
          ))}
        </div>
        <div className="button-stack">
          <button className="primary-button" onClick={saveAssetWorkspace} type="button">
            Save Asset Workspace
          </button>
          <button className="ghost-button" onClick={copyAssetWorkspaceJson} type="button">
            Copy Asset Workspace JSON
          </button>
          <button className="ghost-button" onClick={downloadAssetWorkspaceJson} type="button">
            Download Asset Workspace JSON
          </button>
          <button className="ghost-button" onClick={resetAssetWorkspace} type="button">
            Reset Asset Workspace
          </button>
        </div>
        <div className="button-stack">
          <button
            className="ghost-button"
            onClick={() => assetWorkspaceFileInputRef.current?.click()}
            type="button">
            Import Asset Workspace File
          </button>
          <input
            ref={assetWorkspaceFileInputRef}
            accept="application/json,.json"
            className="manifest-file-input"
            onChange={handleAssetWorkspaceFileImport}
            type="file"
          />
          <button className="ghost-button" onClick={applyAssetWorkspaceImportText} type="button">
            Apply Pasted Asset Workspace JSON
          </button>
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="workspace-import-json">
            Asset Workspace JSON Import
          </label>
          <textarea
            id="workspace-import-json"
            className="manifest-textarea manifest-textarea-compact"
            onChange={(event) => {
              setAssetWorkspaceImportText(event.target.value);
              setAssetWorkspaceImportError('');
            }}
            placeholder='{"fragments":{"residency":{"label":"Residency Live"}}}'
            value={assetWorkspaceImportText}
          />
          {assetWorkspaceImportError ? (
            <p className="panel-note panel-note-error">
              Import Error: {assetWorkspaceImportError}
            </p>
          ) : (
            <p className="panel-note">
              paste 後に `Apply Pasted Asset Workspace JSON` を押すと draft に反映されます。
            </p>
          )}
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="workspace-manifest-label">
            Workspace Manifest Label
          </label>
          <input
            id="workspace-manifest-label"
            className="manifest-input"
            onChange={(event) => updateAssetWorkspaceField('label', event.target.value)}
            type="text"
            value={assetWorkspaceDraft.label ?? ''}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="workspace-world-label">
            Workspace World Label
          </label>
          <input
            id="workspace-world-label"
            className="manifest-input"
            onChange={(event) =>
              updateAssetWorkspaceFragmentField(
                activeConfig.fragmentId,
                'label',
                event.target.value
              )
            }
            type="text"
            value={currentAssetWorkspaceFragment.label ?? ''}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="workspace-splat-url">
            Workspace Splat URL
          </label>
          <input
            id="workspace-splat-url"
            className="manifest-input"
            onChange={(event) =>
              updateAssetWorkspaceFragmentField(
                activeConfig.fragmentId,
                'splatUrl',
                event.target.value
              )
            }
            placeholder={currentAssetWorkspaceFragment.expectedSplatUrl ?? ''}
            type="text"
            value={currentAssetWorkspaceFragment.splatUrl ?? ''}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="workspace-collider-url">
            Workspace Collider URL
          </label>
          <input
            id="workspace-collider-url"
            className="manifest-input"
            onChange={(event) =>
              updateAssetWorkspaceFragmentField(
                activeConfig.fragmentId,
                'colliderMeshUrl',
                event.target.value
              )
            }
            placeholder={currentAssetWorkspaceFragment.expectedColliderMeshUrl ?? ''}
            type="text"
            value={currentAssetWorkspaceFragment.colliderMeshUrl ?? ''}
          />
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="workspace-world-note">
            Workspace World Note
          </label>
          <textarea
            id="workspace-world-note"
            className="manifest-textarea"
            onChange={(event) =>
              updateAssetWorkspaceFragmentField(
                activeConfig.fragmentId,
                'worldNote',
                event.target.value
              )
            }
            value={currentAssetWorkspaceFragment.worldNote ?? ''}
          />
        </div>

        <h2>Studio Bundle</h2>
        <p className="panel-note">
          Asset Workspace / Scene Workspace / Semantic Zone Workspace / Robot Route と、現在の fragment / scene / preset 状態をまとめて持ち出すための bundle です。
        </p>
        <div className="button-stack">
          <button className="primary-button" onClick={copyStudioBundleJson} type="button">
            Copy Studio Bundle JSON
          </button>
          <button className="ghost-button" onClick={downloadStudioBundleJson} type="button">
            Download Studio Bundle JSON
          </button>
          <button className="ghost-button" onClick={saveStudioBundleSnapshot} type="button">
            Save Studio Bundle Snapshot
          </button>
          <button className="ghost-button" onClick={clearStudioBundleShelf} type="button">
            Clear Studio Bundle Shelf
          </button>
          <button
            className="ghost-button"
            onClick={() => studioBundleFileInputRef.current?.click()}
            type="button">
            Import Studio Bundle File
          </button>
          <input
            ref={studioBundleFileInputRef}
            accept="application/json,.json"
            className="manifest-file-input"
            onChange={handleStudioBundleFileImport}
            type="file"
          />
          <button className="ghost-button" onClick={applyStudioBundleImportText} type="button">
            Apply Pasted Studio Bundle JSON
          </button>
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="studio-bundle-json-import">
            Studio Bundle JSON Import
          </label>
          <textarea
            id="studio-bundle-json-import"
            className="manifest-textarea manifest-textarea-compact"
            onChange={(event) => {
              setStudioBundleImportText(event.target.value);
              setStudioBundleImportError('');
            }}
            placeholder='{"state":{"streamSceneId":"window-talk"},"assetWorkspace":{"fragments":{"residency":{"label":"Studio Residency"}}},"sceneWorkspace":{"fragments":{"residency":{"streamScenes":[{"id":"window-talk","title":"Studio Window Talk"}]}}}}'
            value={studioBundleImportText}
          />
          {studioBundleImportError ? (
            <p className="panel-note panel-note-error">
              Import Error: {studioBundleImportError}
            </p>
          ) : (
            <p className="panel-note">
              bundle を適用すると asset / scene / semantic zone / robot route の各 draft が更新され、scene state も可能な範囲で復元されます。
            </p>
          )}
        </div>
        <div className="field-group">
          <label className="field-label" htmlFor="studio-bundle-label">
            Studio Bundle Label
          </label>
          <input
            id="studio-bundle-label"
            className="manifest-input"
            onChange={(event) => setStudioBundleShelfLabel(event.target.value)}
            placeholder={`${activeConfig.fragmentLabel} / ${selectedStreamScene?.label ?? 'Scene'}`}
            type="text"
            value={studioBundleShelfLabel}
          />
        </div>
        {studioBundleShelf.length > 0 ? (
          <>
            <h2>Studio Bundle Shelf</h2>
            <div className="button-stack">
              {studioBundleShelf.map((entry) => (
                <div key={entry.id} className="state-card">
                  <span className="state-label">{entry.label}</span>
                  <strong>
                    {entry.bundle.state.fragmentId} / {entry.bundle.state.streamSceneId ?? 'scene'}
                  </strong>
                  <p className="panel-note">
                    {entry.bundle.assetWorkspace.fragments?.[entry.bundle.state.fragmentId]?.label ??
                      entry.bundle.label}
                  </p>
                  <div className="chip-list">
                    <button className="chip" onClick={() => applyStudioBundleSnapshot(entry)} type="button">
                      Apply
                    </button>
                    <button className="chip" onClick={() => downloadStudioBundleSnapshot(entry)} type="button">
                      Download
                    </button>
                    <button className="chip" onClick={() => deleteStudioBundleSnapshot(entry.id)} type="button">
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : null}

        <h2>World</h2>
        <p className="panel-value">
          {effectiveSplatUrl || 'splat asset 未設定'}
        </p>
        <p className="panel-note">
          Browser 公開用には Marble の raw export をそのまま使わず、
          軽量化した `.sog` を置く前提です。
        </p>
        {isUsingDemoSplat ? (
          <p className="panel-note">
            現在は PlayCanvas 公式の demo splat を fallback 表示しています。
          </p>
        ) : null}

        <h2>Fragment</h2>
        <p className="panel-value">{activeConfig.fragmentLabel}</p>
        <p className="panel-note">URL hash と gate で chapter を切り替えます。</p>

        <h2>Walk Collider</h2>
        <p className="panel-value">{walkColliderLabel}</p>
        <p className="panel-note">
          {activeWorldConfig.colliderMeshUrl
            ? activeWorldConfig.colliderMeshUrl
            : 'GLB 未設定。walk は proxy floor で動作します。'}
        </p>
        <p className="panel-note">
          {walkColliderStatus.mode === 'error'
            ? `GLB collider の読み込みに失敗したため proxy floor を使用中です: ${walkColliderStatus.error}`
            : walkColliderStatus.mode === 'mesh'
              ? 'Marble の collider mesh を floor 判定に使っています。'
              : walkColliderStatus.mode === 'idle'
                ? 'Walk Mode に入るまで collider GLB は読み込みません。初回表示を軽くするための on-demand 読み込みです。'
              : walkColliderStatus.mode === 'loading'
                ? 'GLB collider を読み込み中です。完了までは proxy floor で維持します。'
                : 'splat 表示とは別に、歩行判定は軽量な proxy collider で処理しています。'}
        </p>

        <h2>Dream State</h2>
        <div className="state-grid">
          <div className="state-card">
            <span className="state-label">Shards</span>
            <strong>{collectedShardCount} / {totalShardCount}</strong>
          </div>
          <div className="state-card">
            <span className="state-label">Gate</span>
            <strong>{isGateUnlocked ? 'Open' : `Locked (${remainingShardCount})`}</strong>
          </div>
        </div>
        <button className="ghost-button compact-button" onClick={resetDreamState} type="button">
          Reset Dream State
        </button>
        <p className="panel-note">
          shard 回収状態は browser の localStorage に保存しています。
        </p>
        <p className="panel-note">
          Walk は splat そのものではなく proxy collider を床に使うハイブリッド方式です。
        </p>

        <h2>Camera Mode</h2>
        <div className="chip-list">
          <button
            className={cameraMode === 'orbit' && !isRobotMode ? 'chip active' : 'chip'}
            onClick={() => {
              exitWalkMode();
              setMode('explore');
            }}
            type="button">
            Orbit
          </button>
          <button
            className={isWalkMode ? 'chip active' : 'chip'}
            onClick={() => {
              enterWalkMode();
            }}
            type="button">
            Walk
          </button>
          <button
            className={isRobotMode ? 'chip active' : 'chip'}
            onClick={toggleRobotMode}
            type="button">
            Robot
          </button>
        </div>
        <p className="panel-note">
          {isWalkMode
            ? 'Walk Mode は grounded FPS に切り替わり、hotspot overlay は隠れます。'
            : isRobotMode
              ? 'Robot Mode は orbit camera を robot front / chase / top view に切り替えて追従させます。'
              : 'Orbit mode は写真と配信の構図確認向けです。'}
        </p>

        <h2>Camera Presets</h2>
        <div className="chip-list">
          {activeConfig.cameraPresets.map((preset, index) => (
            <button
              key={preset.id}
              className={preset.id === selectedPresetId ? 'chip active' : 'chip'}
              onClick={() => {
                setSelectedPresetId(preset.id);
                setStatusMessage(`Camera preset: ${preset.label}`);
              }}
              type="button">
              {index + 1}. {preset.label}
            </button>
          ))}
        </div>
        <p className="panel-note">{currentPreset.description}</p>

        <h2>Dream Filter</h2>
        <div className="chip-list">
          {activeConfig.dreamFilters.map((filter) => (
            <button
              key={filter.id}
              className={filter.id === selectedFilterId ? 'chip active' : 'chip'}
              onClick={() => setSelectedFilterId(filter.id)}
              type="button">
              {filter.label}
            </button>
          ))}
        </div>
        <p className="panel-note">{selectedFilter.description}</p>
      </aside>

      <aside className="right-panel glass-panel">
        <h2>{isRobotMode ? 'Robot Mode' : isPhotoMode ? 'Photo Mode' : isLiveMode ? 'Live Mode' : 'Hotkeys'}</h2>

        {isRobotMode ? (
          <>
            <p className="panel-note">
              Gaussian Splat world を robot teleop / camera / waypoint sandbox として使う最小モードです。main stage は robot camera view を追従します。
            </p>
            <div className="state-grid">
              <div className="state-card">
                <span className="state-label">Pose</span>
                <strong>
                  x {robotPoseSummary.x} / z {robotPoseSummary.z}
                </strong>
              </div>
              <div className="state-card">
                <span className="state-label">Heading</span>
                <strong>{robotPoseSummary.yaw} deg</strong>
              </div>
              <div className="state-card">
                <span className="state-label">Route</span>
                <strong>{robotNodeLabel} / {robotTrailDistance} m</strong>
              </div>
              <div className="state-card">
                <span className="state-label">Bridge</span>
                <strong>{robotBridgeStatusLabel}</strong>
              </div>
              <div className="state-card">
                <span className="state-label">Gamepad</span>
                <strong>{gamepadStatusLabel}</strong>
              </div>
              <div className="state-card">
                <span className="state-label">Zone Map</span>
                <strong>
                  {effectiveSemanticZoneMap
                    ? `${semanticZoneCount} zones`
                    : semanticZoneStatusLabel}
                </strong>
              </div>
              <div className="state-card">
                <span className="state-label">Current Zone</span>
                <strong>{semanticZoneCurrentLabel}</strong>
              </div>
            </div>
            <div className="state-card robot-camera-panel">
              <span className="state-label">Front Camera Panel</span>
              <strong>{selectedRobotCamera?.label ?? 'Robot Camera'}</strong>
              <p className="panel-note">
                pose y {robotPoseSummary.y} / waypoint {robotWaypointDistance ? `${robotWaypointDistance} m ahead` : 'none'}
              </p>
              <p className="panel-note">
                gamepad {gamepadState.label}{gamepadState.mapping ? ` / ${gamepadState.mapping}` : ''}
              </p>
              <p className="panel-note">
                bridge {robotBridgeConfig.enabled ? robotBridgeConfig.url : robotBridgeDefaultUrl}
              </p>
              <p className="panel-note">
                inbound {robotBridgeState.lastInboundType ?? 'none'} / outbound {robotBridgeState.lastOutboundType ?? 'none'}
              </p>
              {robotBridgeState.error ? (
                <p className="panel-note panel-note-error">
                  Bridge Error: {robotBridgeState.error}
                </p>
              ) : null}
            </div>
            <Sim2RealPanel
              config={sim2realConfig}
              fragmentId={activeConfig.fragmentId}
              fragmentLabel={activeConfig.fragmentLabel}
              onBenchmarkOverlayChange={setSim2RealBenchmarkOverlay}
              onStatusMessage={setStatusMessage}
              robotPose={robotPose}
              robotTrail={robotTrail}
              robotWaypoint={robotWaypoint}
            />
            <div className="state-card robot-zone-panel">
              <span className="state-label">Semantic Zone Panel</span>
              <strong>
                {semanticZoneCount > 0 ? semanticZoneCurrentLabel : semanticZoneStatusLabel}
              </strong>
              <p className="panel-note">
                {semanticZoneState.url || 'zone map optional'}
              </p>
              <p className="panel-note">
                {semanticZoneWorkspaceModeLabel}
                {isSemanticZoneWorkspaceDirty ? ' / Unsaved Draft' : ''}
              </p>
              <p className="panel-note">
                cost {semanticZoneCostLabel}
                {semanticZoneSummary.tags.length
                  ? ` / tags ${semanticZoneSummary.tags.join(', ')}`
                  : ''}
              </p>
              {semanticZoneState.error ? (
                <p className="panel-note panel-note-error">
                  Zone Error: {semanticZoneState.error}
                </p>
              ) : null}
              {effectiveSemanticZoneMap?.zones?.length ? (
                <div className="chip-list">
                  {effectiveSemanticZoneMap.zones.map((zone) => {
                    const isActiveZone = semanticZoneHits.some((hit) => hit.id === zone.id);

                    return (
                      <span
                        key={zone.id}
                        className={isActiveZone ? 'chip active semantic-zone-chip' : 'chip semantic-zone-chip'}>
                        {zone.label}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>
            <SemanticNavPanel
              activeZoneIds={semanticZoneHits.map((zone) => zone.id)}
              benchmarkOverlay={sim2realBenchmarkOverlay}
              robotPose={robotPose}
              robotTrail={robotTrail}
              waypoint={robotWaypoint}
              zoneMap={effectiveSemanticZoneMap}
            />
            <div className="state-card robot-zone-panel">
              <span className="state-label">Semantic Zone Workspace</span>
              <strong>{activeConfig.fragmentLabel}</strong>
              <p className="panel-note">
                bounds / zone shape / cost / tags を編集すると、その場で overlay と current zone 判定に反映されます。
              </p>
              <div className="button-stack">
                <button className="primary-button" onClick={saveSemanticZoneWorkspace} type="button">
                  Save Zone Workspace
                </button>
                <button className="ghost-button" onClick={copySemanticZoneWorkspaceJson} type="button">
                  Copy Zone JSON
                </button>
                <button className="ghost-button" onClick={downloadSemanticZoneWorkspaceJson} type="button">
                  Download Zone JSON
                </button>
                <button className="ghost-button" onClick={resetSemanticZoneWorkspace} type="button">
                  Reset Zone Workspace
                </button>
                <button
                  className="ghost-button"
                  onClick={() => semanticZoneFileInputRef.current?.click()}
                  type="button">
                  Import Zone File
                </button>
                <input
                  ref={semanticZoneFileInputRef}
                  accept="application/json,.json"
                  className="manifest-file-input"
                  onChange={handleSemanticZoneFileImport}
                  type="file"
                />
                <button className="ghost-button" onClick={applySemanticZoneImportText} type="button">
                  Apply Pasted Zone JSON
                </button>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="semantic-zone-import">
                  Semantic Zone JSON Import
                </label>
                <textarea
                  id="semantic-zone-import"
                  className="manifest-textarea manifest-textarea-compact"
                  onChange={(event) => {
                    setSemanticZoneImportText(event.target.value);
                    setSemanticZoneImportError('');
                  }}
                  placeholder='{"bounds":{"minX":-6,"maxX":6,"minZ":0,"maxZ":12},"zones":[]}'
                  value={semanticZoneImportText}
                />
                {semanticZoneImportError ? (
                  <p className="panel-note panel-note-error">
                    Import Error: {semanticZoneImportError}
                  </p>
                ) : (
                  <p className="panel-note">
                    current fragment の zone map だけを import します。
                  </p>
                )}
              </div>
              <div className="field-grid-two">
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-frame">
                    Frame ID
                  </label>
                  <input
                    id="semantic-zone-frame"
                    className="manifest-input"
                    onChange={(event) => updateSemanticZoneRootField('frameId', event.target.value)}
                    type="text"
                    value={currentSemanticZonePayload.frameId ?? ''}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-resolution">
                    Resolution
                  </label>
                  <input
                    id="semantic-zone-resolution"
                    className="manifest-input"
                    min="0.05"
                    onChange={(event) =>
                      updateSemanticZoneRootField('resolution', Math.max(0.05, Number(event.target.value) || 0.5))
                    }
                    step="0.05"
                    type="number"
                    value={currentSemanticZonePayload.resolution ?? 0.5}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-default-cost">
                    Default Cost
                  </label>
                  <input
                    id="semantic-zone-default-cost"
                    className="manifest-input"
                    max="100"
                    min="0"
                    onChange={(event) =>
                      updateSemanticZoneRootField(
                        'defaultCost',
                        Math.max(0, Math.min(100, Number(event.target.value) || 0))
                      )
                    }
                    step="1"
                    type="number"
                    value={currentSemanticZonePayload.defaultCost ?? 0}
                  />
                </div>
              </div>
              <div className="field-grid-two">
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-min-x">
                    Bounds Min X
                  </label>
                  <input
                    id="semantic-zone-min-x"
                    className="manifest-input"
                    onChange={(event) =>
                      updateSemanticZoneBoundsField('minX', Number(event.target.value) || 0)
                    }
                    step="0.1"
                    type="number"
                    value={currentSemanticZonePayload.bounds?.minX ?? 0}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-max-x">
                    Bounds Max X
                  </label>
                  <input
                    id="semantic-zone-max-x"
                    className="manifest-input"
                    onChange={(event) =>
                      updateSemanticZoneBoundsField('maxX', Number(event.target.value) || 0)
                    }
                    step="0.1"
                    type="number"
                    value={currentSemanticZonePayload.bounds?.maxX ?? 0}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-min-z">
                    Bounds Min Z
                  </label>
                  <input
                    id="semantic-zone-min-z"
                    className="manifest-input"
                    onChange={(event) =>
                      updateSemanticZoneBoundsField('minZ', Number(event.target.value) || 0)
                    }
                    step="0.1"
                    type="number"
                    value={currentSemanticZonePayload.bounds?.minZ ?? 0}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label" htmlFor="semantic-zone-max-z">
                    Bounds Max Z
                  </label>
                  <input
                    id="semantic-zone-max-z"
                    className="manifest-input"
                    onChange={(event) =>
                      updateSemanticZoneBoundsField('maxZ', Number(event.target.value) || 0)
                    }
                    step="0.1"
                    type="number"
                    value={currentSemanticZonePayload.bounds?.maxZ ?? 0}
                  />
                </div>
              </div>
              <div className="button-stack">
                <button className="ghost-button" onClick={addSemanticZoneEntry} type="button">
                  Add Zone
                </button>
                <button
                  className="ghost-button"
                  onClick={() => addSemanticZoneEntryAtPosition(robotPose.position, { sourceLabel: 'robot pose add' })}
                  type="button">
                  Add Zone At Robot
                </button>
                <button
                  className="ghost-button"
                  disabled={!robotWaypoint}
                  onClick={() =>
                    robotWaypoint
                      ? addSemanticZoneEntryAtPosition(robotWaypoint.position, { sourceLabel: 'waypoint add' })
                      : undefined
                  }
                  type="button">
                  Add Zone At Waypoint
                </button>
                <button className="ghost-button" onClick={addSemanticZonesFromRoute} type="button">
                  Add Zones From Route
                </button>
                <button className="ghost-button" onClick={fitSemanticZoneWorkspaceBounds} type="button">
                  Fit Bounds To Zones
                </button>
                <button className="ghost-button" onClick={clearAllSemanticZones} type="button">
                  Clear All Zones
                </button>
              </div>
              <div className="zone-editor-stack">
                {(currentSemanticZonePayload.zones ?? []).map((zone) => (
                  <div key={zone.id} className="state-card zone-editor-card">
                    <div className="status-row">
                      <strong>{zone.label || zone.id}</strong>
                      <div className="chip-list">
                        <button
                          className="chip"
                          onClick={() => assignRobotPoseToSemanticZone(zone.id)}
                          type="button">
                          {'Zone <- Robot'}
                        </button>
                        <button
                          className="chip"
                          disabled={!robotWaypoint}
                          onClick={() => assignWaypointToSemanticZone(zone.id)}
                          type="button">
                          {'Zone <- Waypoint'}
                        </button>
                        <button
                          className="chip"
                          onClick={() => moveRobotToSemanticZone(zone.id)}
                          type="button">
                          {'Robot -> Zone'}
                        </button>
                        <button
                          className="chip"
                          onClick={() => duplicateSemanticZoneEntry(zone.id)}
                          type="button">
                          Duplicate
                        </button>
                        <button
                          className="chip"
                          onClick={() => removeSemanticZoneEntry(zone.id)}
                          type="button">
                          Delete
                        </button>
                      </div>
                    </div>
                    <div className="field-grid-two">
                      <div className="field-group">
                        <label className="field-label" htmlFor={`semantic-zone-id-${zone.id}`}>
                          Zone ID
                        </label>
                        <input
                          id={`semantic-zone-id-${zone.id}`}
                          className="manifest-input"
                          onChange={(event) => updateSemanticZoneField(zone.id, 'id', event.target.value)}
                          type="text"
                          value={zone.id ?? ''}
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label" htmlFor={`semantic-zone-label-${zone.id}`}>
                          Label
                        </label>
                        <input
                          id={`semantic-zone-label-${zone.id}`}
                          className="manifest-input"
                          onChange={(event) => updateSemanticZoneField(zone.id, 'label', event.target.value)}
                          type="text"
                          value={zone.label ?? ''}
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label" htmlFor={`semantic-zone-shape-${zone.id}`}>
                          Shape
                        </label>
                        <select
                          id={`semantic-zone-shape-${zone.id}`}
                          className="manifest-input"
                          onChange={(event) => updateSemanticZoneShape(zone.id, event.target.value)}
                          value={zone.shape ?? 'rect'}>
                          <option value="rect">rect</option>
                          <option value="circle">circle</option>
                        </select>
                      </div>
                      <div className="field-group">
                        <label className="field-label" htmlFor={`semantic-zone-cost-${zone.id}`}>
                          Cost
                        </label>
                        <input
                          id={`semantic-zone-cost-${zone.id}`}
                          className="manifest-input"
                          max="100"
                          min="0"
                          onChange={(event) =>
                            updateSemanticZoneField(
                              zone.id,
                              'cost',
                              Math.max(0, Math.min(100, Number(event.target.value) || 0))
                            )
                          }
                          step="1"
                          type="number"
                          value={zone.cost ?? 0}
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label" htmlFor={`semantic-zone-center-x-${zone.id}`}>
                          Center X
                        </label>
                        <input
                          id={`semantic-zone-center-x-${zone.id}`}
                          className="manifest-input"
                          onChange={(event) =>
                            updateSemanticZoneCenterField(zone.id, 0, Number(event.target.value) || 0)
                          }
                          step="0.1"
                          type="number"
                          value={zone.center?.[0] ?? 0}
                        />
                      </div>
                      <div className="field-group">
                        <label className="field-label" htmlFor={`semantic-zone-center-z-${zone.id}`}>
                          Center Z
                        </label>
                        <input
                          id={`semantic-zone-center-z-${zone.id}`}
                          className="manifest-input"
                          onChange={(event) =>
                            updateSemanticZoneCenterField(zone.id, 2, Number(event.target.value) || 0)
                          }
                          step="0.1"
                          type="number"
                          value={zone.center?.[2] ?? 0}
                        />
                      </div>
                      {zone.shape === 'rect' ? (
                        <>
                          <div className="field-group">
                            <label className="field-label" htmlFor={`semantic-zone-size-x-${zone.id}`}>
                              Size X
                            </label>
                            <input
                              id={`semantic-zone-size-x-${zone.id}`}
                              className="manifest-input"
                              min="0.1"
                              onChange={(event) =>
                                updateSemanticZoneSizeField(
                                  zone.id,
                                  0,
                                  Math.max(0.1, Number(event.target.value) || 0.1)
                                )
                              }
                              step="0.1"
                              type="number"
                              value={zone.size?.[0] ?? 1}
                            />
                          </div>
                          <div className="field-group">
                            <label className="field-label" htmlFor={`semantic-zone-size-z-${zone.id}`}>
                              Size Z
                            </label>
                            <input
                              id={`semantic-zone-size-z-${zone.id}`}
                              className="manifest-input"
                              min="0.1"
                              onChange={(event) =>
                                updateSemanticZoneSizeField(
                                  zone.id,
                                  1,
                                  Math.max(0.1, Number(event.target.value) || 0.1)
                                )
                              }
                              step="0.1"
                              type="number"
                              value={zone.size?.[1] ?? 1}
                            />
                          </div>
                        </>
                      ) : (
                        <div className="field-group">
                          <label className="field-label" htmlFor={`semantic-zone-radius-${zone.id}`}>
                            Radius
                          </label>
                          <input
                            id={`semantic-zone-radius-${zone.id}`}
                            className="manifest-input"
                            min="0.1"
                            onChange={(event) =>
                              updateSemanticZoneField(
                                zone.id,
                                'radius',
                                Math.max(0.1, Number(event.target.value) || 0.1)
                              )
                            }
                            step="0.1"
                            type="number"
                            value={zone.radius ?? 1}
                          />
                        </div>
                      )}
                    </div>
                    <div className="field-group">
                      <label className="field-label" htmlFor={`semantic-zone-tags-${zone.id}`}>
                        Tags
                      </label>
                      <input
                        id={`semantic-zone-tags-${zone.id}`}
                        className="manifest-input"
                        onChange={(event) => updateSemanticZoneTags(zone.id, event.target.value)}
                        type="text"
                        value={Array.isArray(zone.tags) ? zone.tags.join(', ') : ''}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="chip-list">
              {activeConfig.robotics.cameras.map((camera) => (
                <button
                  key={camera.id}
                  className={camera.id === selectedRobotCameraId ? 'chip active' : 'chip'}
                  onClick={() => {
                    setSelectedRobotCameraId(camera.id);
                    setStatusMessage(`Robot camera: ${camera.label}`);
                  }}
                  type="button">
                  {camera.label}
                </button>
              ))}
            </div>
            <div className="robotics-control-grid">
              <button className="ghost-button" onClick={() => moveRobot('forward')} type="button">
                Forward
              </button>
              <button className="ghost-button" onClick={() => moveRobot('turn-left')} type="button">
                Turn Left
              </button>
              <button className="ghost-button" onClick={() => moveRobot('turn-right')} type="button">
                Turn Right
              </button>
              <button className="ghost-button" onClick={() => moveRobot('backward')} type="button">
                Backward
              </button>
            </div>
            <div className="button-stack">
              <button className="primary-button" onClick={dropRobotWaypoint} type="button">
                Drop Waypoint
              </button>
              <button className="ghost-button" onClick={clearRobotWaypoint} type="button">
                Clear Waypoint
              </button>
              <button className="ghost-button" onClick={clearRobotRoute} type="button">
                Clear Route
              </button>
              <button className="ghost-button" onClick={reconnectRobotBridge} type="button">
                Reconnect Bridge
              </button>
              <button className="ghost-button" onClick={resetRobotPose} type="button">
                Reset Robot Pose
              </button>
            </div>
            <div className="state-card robot-route-export-panel">
              <span className="state-label">Route Export</span>
              <strong>{robotNodeLabel} / {robotTrailDistance} m</strong>
              <p className="panel-note">
                {activeConfig.fragmentLabel} / {robotRoutePayload.frameId}
              </p>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-route-shelf-label">
                  Route Snapshot Label
                </label>
                <input
                  id="robot-route-shelf-label"
                  className="manifest-input"
                  onChange={(event) => setRobotRouteShelfLabel(event.target.value)}
                  placeholder={robotRouteShelfSummaryLabel}
                  type="text"
                  value={robotRouteShelfLabel}
                />
              </div>
              <div className="button-stack">
                <button className="primary-button" onClick={saveRobotRouteSnapshot} type="button">
                  Save Route Snapshot
                </button>
                <button className="ghost-button" onClick={copyRobotRouteJson} type="button">
                  Copy Route JSON
                </button>
                <button className="ghost-button" onClick={downloadRobotRouteJson} type="button">
                  Download Route JSON
                </button>
                <button
                  className="ghost-button"
                  onClick={() => robotRouteFileInputRef.current?.click()}
                  type="button">
                  Import Route File
                </button>
                <input
                  ref={robotRouteFileInputRef}
                  accept="application/json,.json"
                  className="manifest-file-input"
                  onChange={handleRobotRouteFileImport}
                  type="file"
                />
                <button className="ghost-button" onClick={applyRobotRouteImportText} type="button">
                  Apply Pasted Route JSON
                </button>
                <button className="ghost-button" onClick={clearRobotRouteShelf} type="button">
                  Clear Route Shelf
                </button>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-route-json">
                  Robot Route JSON
                </label>
                <textarea
                  id="robot-route-json"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotRoutePayloadJson}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-route-import">
                  Robot Route JSON Import
                </label>
                <textarea
                  id="robot-route-import"
                  className="manifest-textarea manifest-textarea-compact"
                  onChange={(event) => {
                    setRobotRouteImportText(event.target.value);
                    setRobotRouteImportError('');
                  }}
                  placeholder='{"pose":{"position":[0,0,5.8],"yawDegrees":0},"route":[[0,0,5.8]],"waypoint":{"position":[0,0,8.6]}}'
                  value={robotRouteImportText}
                />
                {robotRouteImportError ? (
                  <p className="panel-note panel-note-error">
                    Import Error: {robotRouteImportError}
                  </p>
                ) : (
                  <p className="panel-note">
                    pose / waypoint / route をまとめて適用します。
                  </p>
                )}
              </div>
              {robotRouteShelf.length > 0 ? (
                <div className="state-list">
                  {robotRouteShelf.map((entry) => (
                    <div key={entry.id} className="state-card">
                      <div className="status-row">
                        <strong>{entry.label}</strong>
                        <div className="status-row-badges">
                          <HealthBadge health={robotRouteShelfHealthMap[entry.id]} />
                          <span className="chip">{entry.route.fragmentLabel || entry.route.fragmentId || 'Route'}</span>
                        </div>
                      </div>
                      <p className="panel-note">
                        {entry.route.route.length} nodes
                        {entry.route.waypoint ? ' / waypoint' : ' / no waypoint'}
                      </p>
                      {robotRouteShelfHealthMap[entry.id]?.detail ? (
                        <p className="panel-note">
                          {robotRouteShelfHealthMap[entry.id].detail}
                        </p>
                      ) : null}
                      <div className="chip-list">
                        <button className="chip" onClick={() => applyRobotRouteSnapshot(entry)} type="button">
                          Apply
                        </button>
                        <button className="chip" onClick={() => downloadRobotRouteSnapshot(entry)} type="button">
                          Download
                        </button>
                        <button className="chip" onClick={() => deleteRobotRouteSnapshot(entry.id)} type="button">
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
            <div className="state-card robot-route-export-panel">
              <span className="state-label">Mission Export</span>
              <div className="status-row">
                <strong>
                  {robotMissionPayload.label}
                </strong>
                <div className="status-row-badges">
                  <HealthBadge health={robotMissionDraftBundleHealth} />
                </div>
              </div>
              <p className="panel-note">
                {robotMissionPayload.startupMode} / preset {robotMissionPayload.cameraPresetId || 'none'} / robot camera {robotMissionPayload.robotCameraId || 'none'}
              </p>
              <p className="panel-note">
                route {robotMissionPayload.routeUrl} / zone {robotMissionPayload.zoneMapUrl || 'none'}
              </p>
              {robotMissionDraftBundleHealth?.detail ? (
                <p
                  className={
                    robotMissionDraftBundleHealth.status === 'error'
                      ? 'panel-note panel-note-error'
                      : 'panel-note'
                  }>
                  {robotMissionDraftBundleHealth.label}: {robotMissionDraftBundleHealth.detail}
                </p>
              ) : null}
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-id">
                  Mission ID
                </label>
                <input
                  id="robot-mission-id"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField('id', event.target.value)
                  }
                  placeholder={robotMissionExportId}
                  type="text"
                  value={robotMissionPayload.id}
                />
                <p className="panel-note">
                  publish filename と mission manifest id に使います。
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-label">
                  Mission Label
                </label>
                <input
                  id="robot-mission-label"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField('label', event.target.value)
                  }
                  placeholder={`${activeConfig.fragmentLabel} Robot Mission`}
                  type="text"
                  value={robotMissionPayload.label}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-description">
                  Mission Description
                </label>
                <textarea
                  id="robot-mission-description"
                  className="manifest-textarea manifest-textarea-compact"
                  onChange={(event) =>
                    updateRobotMissionDraftField(
                      'description',
                      event.target.value
                    )
                  }
                  placeholder={`${activeConfig.fragmentLabel} robot mission snapshot`}
                  value={robotMissionPayload.description}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-accent">
                  Mission Accent
                </label>
                <input
                  id="robot-mission-accent"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField('accent', event.target.value)
                  }
                  placeholder="#85e3e1"
                  type="text"
                  value={robotMissionPayload.accent}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-route-label">
                  Route Label
                </label>
                <input
                  id="robot-route-label"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftRouteField('label', event.target.value)
                  }
                  placeholder={`${activeConfig.fragmentLabel} Route Snapshot`}
                  type="text"
                  value={robotRoutePayload.label}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-route-description">
                  Route Description
                </label>
                <textarea
                  id="robot-route-description"
                  className="manifest-textarea manifest-textarea-compact"
                  onChange={(event) =>
                    updateRobotMissionDraftRouteField(
                      'description',
                      event.target.value
                    )
                  }
                  placeholder={`${activeConfig.fragmentLabel} robot route snapshot`}
                  value={robotRoutePayload.description}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-route-accent">
                  Route Accent
                </label>
                <input
                  id="robot-route-accent"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftRouteField('accent', event.target.value)
                  }
                  placeholder="#85e3e1"
                  type="text"
                  value={robotRoutePayload.accent}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-fragment-id">
                  Mission Fragment ID
                </label>
                <input
                  id="robot-mission-fragment-id"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField('fragmentId', event.target.value)
                  }
                  placeholder={activeConfig.fragmentId}
                  type="text"
                  value={robotMissionPayload.fragmentId}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-fragment-label">
                  Mission Fragment Label
                </label>
                <input
                  id="robot-mission-fragment-label"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField(
                      'fragmentLabel',
                      event.target.value
                    )
                  }
                  placeholder={activeConfig.fragmentLabel}
                  type="text"
                  value={robotMissionPayload.fragmentLabel}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-route-id">
                  Mission Route ID
                </label>
                <input
                  id="robot-mission-route-id"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftRouteId(event.target.value)
                  }
                  placeholder={
                    extractRobotRouteIdFromUrl(robotMissionPayload.routeUrl) ||
                    buildMissionSlug(robotRoutePayload.label, `${activeConfig.fragmentId}-route`)
                  }
                  type="text"
                  value={extractRobotRouteIdFromUrl(robotMissionPayload.routeUrl)}
                />
                <p className="panel-note">
                  route publish file 名を決めます。
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-route-url">
                  Mission Route URL
                </label>
                <input
                  id="robot-mission-route-url"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField('routeUrl', event.target.value)
                  }
                  placeholder={`/robot-routes/${robotMissionExportId}.json`}
                  type="text"
                  value={robotMissionPayload.routeUrl}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-zone-map-url">
                  Mission Zone Map URL
                </label>
                <input
                  id="robot-mission-zone-map-url"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftField('zoneMapUrl', event.target.value)
                  }
                  placeholder={`/manifests/robotics-${activeConfig.fragmentId}.zones.json`}
                  type="text"
                  value={robotMissionPayload.zoneMapUrl}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-world-asset-label">
                  Mission World Asset Label
                </label>
                <input
                  id="robot-mission-world-asset-label"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftWorldField(
                      'assetLabel',
                      event.target.value
                    )
                  }
                  placeholder={currentRobotWorldContext.assetLabel || assetBundle.assetLabel}
                  type="text"
                  value={robotMissionPayload.world.assetLabel}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-world-frame-id">
                  Mission World Frame ID
                </label>
                <input
                  id="robot-mission-world-frame-id"
                  className="manifest-input"
                  onChange={(event) =>
                    updateRobotMissionDraftWorldField('frameId', event.target.value)
                  }
                  placeholder={currentRobotWorldContext.frameId || 'dreamwalker_map'}
                  type="text"
                  value={robotMissionPayload.world.frameId}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-draft-bundle-shelf-label">
                  Draft Snapshot Label
                </label>
                <input
                  id="robot-mission-draft-bundle-shelf-label"
                  className="manifest-input"
                  onChange={(event) =>
                    setRobotMissionDraftBundleShelfLabel(event.target.value)
                  }
                  placeholder={robotMissionDraftBundleShelfSummaryLabel}
                  type="text"
                  value={robotMissionDraftBundleShelfLabel}
                />
              </div>
              <div className="button-stack">
                <button
                  className="primary-button"
                  onClick={saveRobotMissionDraftBundleSnapshot}
                  type="button">
                  Save Draft Snapshot
                </button>
                <button className="ghost-button" onClick={copyRobotMissionJson} type="button">
                  Copy Mission JSON
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionJson} type="button">
                  Download Mission JSON
                </button>
                <button
                  className="ghost-button"
                  onClick={copyPublishedRobotMissionJson}
                  type="button">
                  Copy Published Preview
                </button>
                <button
                  className="ghost-button"
                  onClick={downloadPublishedRobotMissionJson}
                  type="button">
                  Download Published Preview
                </button>
                <button className="ghost-button" onClick={copyRobotMissionLaunchUrl} type="button">
                  Copy Launch
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionLaunchUrl} type="button">
                  Download Launch
                </button>
                <button className="ghost-button" onClick={copyRobotMissionPreflightSummary} type="button">
                  Copy Preflight
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionPreflightSummary} type="button">
                  Download Preflight
                </button>
                <button className="ghost-button" onClick={copyRobotMissionPublishReport} type="button">
                  Copy Publish Report
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionPublishReport} type="button">
                  Download Publish Report
                </button>
                <button className="ghost-button" onClick={copyRobotMissionValidateCommand} type="button">
                  Copy Validate
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionValidateCommand} type="button">
                  Download Validate
                </button>
                <button className="ghost-button" onClick={copyRobotMissionReleaseCommand} type="button">
                  Copy Release
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionReleaseCommand} type="button">
                  Download Release
                </button>
                <button className="ghost-button" onClick={copyRobotMissionDraftBundleJson} type="button">
                  Copy Draft Bundle
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionDraftBundleJson} type="button">
                  Download Draft Bundle
                </button>
                <button className="ghost-button" onClick={copyRobotMissionPublishCommand} type="button">
                  Copy Publish Command
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionPublishCommand} type="button">
                  Download Publish Command
                </button>
                <button className="ghost-button" onClick={copyRobotMissionArtifactPack} type="button">
                  Copy Artifact Pack
                </button>
                <button className="ghost-button" onClick={downloadRobotMissionArtifactPack} type="button">
                  Download Artifact Pack
                </button>
                <button
                  className="ghost-button"
                  onClick={() => openRobotMissionDraftBundleFilePicker('apply')}
                  type="button">
                  Import Draft Bundle File
                </button>
                <button
                  className="ghost-button"
                  onClick={() => openRobotMissionDraftBundleFilePicker('shelf')}
                  type="button">
                  Import Draft Bundle File To Shelf
                </button>
                <input
                  ref={robotMissionDraftBundleFileInputRef}
                  accept="application/json,.json"
                  className="manifest-file-input"
                  onChange={handleRobotMissionDraftBundleFileImport}
                  type="file"
                />
                <button
                  className="ghost-button"
                  onClick={applyRobotMissionDraftBundleImportText}
                  type="button">
                  Apply Pasted Draft Bundle
                </button>
                <button
                  className="ghost-button"
                  onClick={applyRobotMissionDraftBundleImportTextToShelf}
                  type="button">
                  Apply Pasted Draft Bundle To Shelf
                </button>
                <button
                  className="ghost-button"
                  onClick={clearRobotMissionDraftBundleShelf}
                  type="button">
                  Clear Draft Shelf
                </button>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-json">
                  Robot Mission JSON
                </label>
                <textarea
                  id="robot-mission-json"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionPayloadJson}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="published-robot-mission-json">
                  Published Mission Preview JSON
                </label>
                <textarea
                  id="published-robot-mission-json"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={publishedRobotMissionPayloadJson}
                />
                <p className="panel-note">
                  `publish:robot-mission --bundle` を default option で流した時の public mission manifest preview です。
                </p>
                <p className="panel-note">
                  published file {publishedRobotMissionFileName}
                </p>
                <p className="panel-note">
                  preflight fragment {publishedRobotMissionPreview.fragmentId} / route id {publishedRobotMissionPreview.routeId} / mission id {publishedRobotMissionPreview.missionId}
                </p>
                <p className="panel-note">
                  route file {publishedRobotMissionPreview.routeFileName}
                </p>
                <p className="panel-note">
                  world asset {publishedRobotMissionPreview.payload.world.assetLabel || 'none'} / frame {publishedRobotMissionPreview.payload.world.frameId || 'none'} / zone {publishedRobotMissionPreview.payload.zoneMapUrl || 'none'}
                </p>
                <p className="panel-note">
                  preflight mission {publishedRobotMissionPreview.payload.label || 'none'} / desc {publishedRobotMissionPreview.payload.description || 'none'} / fragment label {publishedRobotMissionPreview.payload.fragmentLabel || 'none'}
                </p>
                <p className="panel-note">
                  preflight route meta {publishedRobotMissionPreview.routeLabel} / accent {publishedRobotMissionPreview.routeAccent} / startup {publishedRobotMissionPreview.payload.startupMode} / preset {publishedRobotMissionPreview.payload.cameraPresetId || 'none'} / robot camera {publishedRobotMissionPreview.payload.robotCameraId || 'none'} / scene {publishedRobotMissionPreview.payload.streamSceneId || 'none'}
                </p>
                <p className="panel-note">
                  preflight route desc {publishedRobotMissionPreview.routeDescription} / preset label {publishedRobotMissionPreview.cameraPresetLabel} / robot camera label {publishedRobotMissionPreview.robotCameraLabel} / scene label {publishedRobotMissionPreview.streamSceneLabel}
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-draft-bundle-json">
                  Mission Draft Bundle JSON
                </label>
                <textarea
                  id="robot-mission-draft-bundle-json"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionDraftBundleJson}
                />
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-artifact-pack-json">
                  Mission Artifact Pack JSON
                </label>
                <textarea
                  id="robot-mission-artifact-pack-json"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionArtifactPackJson}
                />
                <p className="panel-note">
                  `publish:robot-mission --bundle` の正規入力です。
                </p>
                <p className="panel-note">
                  artifact file {robotMissionArtifactPackFileName}
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-publish-report-json">
                  Publish Report JSON
                </label>
                <textarea
                  id="robot-mission-publish-report-json"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionPublishReportJson}
                />
                <p className="panel-note">
                  CLI 側 `publish:robot-mission --report-output` と同じ schema の preview です。
                </p>
                <p className="panel-note">
                  report file {robotMissionPublishReportFileName}
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-draft-bundle-import">
                  Mission Draft Bundle Import
                </label>
                <textarea
                  id="robot-mission-draft-bundle-import"
                  className="manifest-textarea manifest-textarea-compact"
                  onChange={(event) => {
                    setRobotMissionDraftBundleImportText(event.target.value);
                    setRobotMissionDraftBundleImportError('');
                  }}
                  placeholder='{"fragmentId":"residency","mission":{...},"route":{...},"zones":{...}}'
                  value={robotMissionDraftBundleImportText}
                />
                {robotMissionDraftBundleImportError ? (
                  <p className="panel-note panel-note-error">
                    Import Error: {robotMissionDraftBundleImportError}
                  </p>
                ) : (
                  <p className="panel-note">
                    draft bundle か artifact pack から mission / route / zones をまとめて preview 適用します。
                  </p>
                )}
              </div>
              {robotMissionDraftBundleImportPreview?.artifactPack ? (
                <>
                  <p className="panel-note">
                    artifact pack preview {robotMissionDraftBundleImportPreview.artifactPack.label} / {robotMissionDraftBundleImportPreview.artifactPack.fileCount} files
                  </p>
                  {robotMissionDraftBundleImportPreview.artifactPack.preflightSummary.content ? (
                    <div className="field-group">
                      <label className="field-label" htmlFor="robot-mission-import-artifact-preflight">
                        Import Artifact Preflight
                      </label>
                      <textarea
                        id="robot-mission-import-artifact-preflight"
                        className="manifest-textarea manifest-textarea-compact"
                        readOnly
                        value={robotMissionDraftBundleImportPreview.artifactPack.preflightSummary.content}
                      />
                      <p className="panel-note">
                        artifact preflight file {robotMissionDraftBundleImportPreview.artifactPack.preflightSummary.fileName || 'none'}
                      </p>
                    </div>
                  ) : null}
                  {robotMissionDraftBundleImportPreview.artifactPack.publishReport.content ? (
                    <div className="field-group">
                      <label className="field-label" htmlFor="robot-mission-import-artifact-report">
                        Import Artifact Publish Report
                      </label>
                      <textarea
                        id="robot-mission-import-artifact-report"
                        className="manifest-textarea manifest-textarea-compact"
                        readOnly
                        value={robotMissionDraftBundleImportPreview.artifactPack.publishReport.content}
                      />
                      <p className="panel-note">
                        artifact report file {robotMissionDraftBundleImportPreview.artifactPack.publishReport.fileName || 'none'}
                      </p>
                    </div>
                  ) : null}
                </>
              ) : null}
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-validate-command">
                  Mission Validate Command
                </label>
                <textarea
                  id="robot-mission-validate-command"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionValidateCommand}
                />
                <p className="panel-note">
                  `validate:robot-bundle` で artifact pack 単体の preflight を先に確認するための command です。
                </p>
                <p className="panel-note">
                  validate file {robotMissionArtifactPackFileName.replace(/\.json$/i, '.validate-command.txt')}
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-release-command">
                  Mission Release Command
                </label>
                <textarea
                  id="robot-mission-release-command"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionReleaseCommand}
                />
                <p className="panel-note">
                  `validate:robot-bundle` と `publish:robot-mission` を順番に回すための release command です。
                </p>
                <p className="panel-note">
                  local artifact pack を使う時は `.preflight.txt` と `.publish-report.json` を同じディレクトリへ自動出力します。
                </p>
                <p className="panel-note">
                  release file {robotMissionArtifactPackFileName.replace(/\.json$/i, '.release-command.txt')}
                </p>
              </div>
              <div className="field-group">
                <label className="field-label" htmlFor="robot-mission-publish-command">
                  Mission Publish Command
                </label>
                <textarea
                  id="robot-mission-publish-command"
                  className="manifest-textarea manifest-textarea-compact"
                  readOnly
                  value={robotMissionPublishCommand}
                />
              </div>
              <p className="panel-note">
                current route / zone source / camera preset / robot camera / stream scene を含む authoring 用 JSON と publish command です。public 配置前の draft として使います。
              </p>
              <p className="panel-note">
                draft file {robotMissionDraftBundleFileName}
              </p>
              <p className="panel-note">
                artifact file {robotMissionArtifactPackFileName}
              </p>
              {robotMissionDraftBundleShelf.length > 0 ? (
                <div className="state-list">
                  {robotMissionDraftBundleShelf.map((entry) => {
                    const snapshotPreview = buildPublishedRobotMissionPreviewFromBundle(
                      entry.bundle,
                      activeConfig
                    );
                    const snapshotConfig = snapshotPreview.config;
                    const snapshotHealth =
                      robotMissionDraftBundleShelfHealthMap[entry.id] ?? null;

                    return (
                      <div key={entry.id} className="state-card">
                        <div className="status-row">
                          <strong>{entry.label}</strong>
                          <div className="status-row-badges">
                            <HealthBadge health={snapshotHealth} />
                            <span className="chip">
                              {entry.bundle.fragmentLabel ||
                                entry.bundle.fragmentId ||
                                'Mission'}
                            </span>
                            <span className="chip">
                              {entry.bundle.mission.startupMode || 'robot'}
                            </span>
                          </div>
                        </div>
                        <p className="panel-note">
                          {entry.bundle.mission.label || 'Untitled Mission'}
                        </p>
                        {snapshotHealth?.detail ? (
                          <p
                            className={
                              snapshotHealth.status === 'error'
                                ? 'panel-note panel-note-error'
                                : 'panel-note'
                            }>
                            {snapshotHealth.label}: {snapshotHealth.detail}
                          </p>
                        ) : null}
                        <p className="panel-note">
                          {entry.bundle.route.route.length} nodes
                          {entry.bundle.zones
                            ? ` / ${entry.bundle.zones.zones.length} zones`
                            : ' / no zones'}
                        </p>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-label-${entry.id}`}>
                            Snapshot Label
                          </label>
                          <input
                            id={`robot-mission-snapshot-label-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotLabel(
                                entry.id,
                                event.target.value
                              )
                            }
                            type="text"
                            value={entry.label}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-id-${entry.id}`}>
                            Snapshot Mission ID
                          </label>
                          <input
                            id={`robot-mission-snapshot-id-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'id',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.id}
                            type="text"
                            value={entry.bundle.mission.id}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-mission-label-${entry.id}`}>
                            Snapshot Mission Label
                          </label>
                          <input
                            id={`robot-mission-snapshot-mission-label-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'label',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.label}
                            type="text"
                            value={entry.bundle.mission.label}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-description-${entry.id}`}>
                            Snapshot Mission Description
                          </label>
                          <textarea
                            id={`robot-mission-snapshot-description-${entry.id}`}
                            className="manifest-textarea manifest-textarea-compact"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'description',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.description}
                            value={entry.bundle.mission.description}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-accent-${entry.id}`}>
                            Snapshot Mission Accent
                          </label>
                          <input
                            id={`robot-mission-snapshot-accent-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'accent',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.accent}
                            type="text"
                            value={entry.bundle.mission.accent}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-route-label-${entry.id}`}>
                            Snapshot Route Label
                          </label>
                          <input
                            id={`robot-mission-snapshot-route-label-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotRouteField(
                                entry.id,
                                'label',
                                event.target.value
                              )
                            }
                            placeholder={entry.bundle.route.label}
                            type="text"
                            value={entry.bundle.route.label}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-route-description-${entry.id}`}>
                            Snapshot Route Description
                          </label>
                          <textarea
                            id={`robot-mission-snapshot-route-description-${entry.id}`}
                            className="manifest-textarea manifest-textarea-compact"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotRouteField(
                                entry.id,
                                'description',
                                event.target.value
                              )
                            }
                            placeholder={entry.bundle.route.description}
                            value={entry.bundle.route.description}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-route-accent-${entry.id}`}>
                            Snapshot Route Accent
                          </label>
                          <input
                            id={`robot-mission-snapshot-route-accent-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotRouteField(
                                entry.id,
                                'accent',
                                event.target.value
                              )
                            }
                            placeholder={entry.bundle.route.accent}
                            type="text"
                            value={entry.bundle.route.accent}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-fragment-id-${entry.id}`}>
                            Snapshot Mission Fragment ID
                          </label>
                          <input
                            id={`robot-mission-snapshot-fragment-id-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'fragmentId',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.fragmentId}
                            type="text"
                            value={entry.bundle.mission.fragmentId}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-fragment-label-${entry.id}`}>
                            Snapshot Mission Fragment Label
                          </label>
                          <input
                            id={`robot-mission-snapshot-fragment-label-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'fragmentLabel',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.fragmentLabel}
                            type="text"
                            value={entry.bundle.mission.fragmentLabel}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-route-id-${entry.id}`}>
                            Snapshot Mission Route ID
                          </label>
                          <input
                            id={`robot-mission-snapshot-route-id-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionRouteId(
                                entry.id,
                                event.target.value
                              )
                            }
                            placeholder={
                              extractRobotRouteIdFromUrl(snapshotPreview.payload.routeUrl) ||
                              buildMissionSlug(
                                entry.bundle.route.label,
                                `${entry.bundle.fragmentId}-route`
                              )
                            }
                            type="text"
                            value={extractRobotRouteIdFromUrl(entry.bundle.mission.routeUrl)}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-route-url-${entry.id}`}>
                            Snapshot Mission Route URL
                          </label>
                          <input
                            id={`robot-mission-snapshot-route-url-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'routeUrl',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.routeUrl}
                            type="text"
                            value={entry.bundle.mission.routeUrl}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-zone-map-url-${entry.id}`}>
                            Snapshot Zone Map URL
                          </label>
                          <input
                            id={`robot-mission-snapshot-zone-map-url-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'zoneMapUrl',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.zoneMapUrl}
                            type="text"
                            value={entry.bundle.mission.zoneMapUrl}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-world-asset-label-${entry.id}`}>
                            Snapshot World Asset Label
                          </label>
                          <input
                            id={`robot-mission-snapshot-world-asset-label-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionWorldField(
                                entry.id,
                                'assetLabel',
                                event.target.value
                              )
                            }
                            placeholder={
                              snapshotPreview.payload.world.assetLabel ||
                              snapshotConfig.fragmentLabel
                            }
                            type="text"
                            value={entry.bundle.mission.world.assetLabel}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-world-frame-id-${entry.id}`}>
                            Snapshot World Frame ID
                          </label>
                          <input
                            id={`robot-mission-snapshot-world-frame-id-${entry.id}`}
                            className="manifest-input"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionWorldField(
                                entry.id,
                                'frameId',
                                event.target.value
                              )
                            }
                            placeholder={snapshotPreview.payload.world.frameId || 'dreamwalker_map'}
                            type="text"
                            value={entry.bundle.mission.world.frameId}
                          />
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-startup-mode-${entry.id}`}>
                            Snapshot Startup Mode
                          </label>
                          <select
                            id={`robot-mission-snapshot-startup-mode-${entry.id}`}
                            className="manifest-select"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'startupMode',
                                event.target.value
                              )
                            }
                            value={entry.bundle.mission.startupMode}>
                            <option value="">Default</option>
                            {robotMissionStartupModes.map((startupMode) => (
                              <option key={startupMode} value={startupMode}>
                                {startupMode}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-camera-preset-${entry.id}`}>
                            Snapshot Camera Preset
                          </label>
                          <select
                            id={`robot-mission-snapshot-camera-preset-${entry.id}`}
                            className="manifest-select"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'cameraPresetId',
                                event.target.value
                              )
                            }
                            value={entry.bundle.mission.cameraPresetId}>
                            <option value="">Default</option>
                            {snapshotConfig.cameraPresets.map((preset) => (
                              <option key={preset.id} value={preset.id}>
                                {preset.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-robot-camera-${entry.id}`}>
                            Snapshot Robot Camera
                          </label>
                          <select
                            id={`robot-mission-snapshot-robot-camera-${entry.id}`}
                            className="manifest-select"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'robotCameraId',
                                event.target.value
                              )
                            }
                            value={entry.bundle.mission.robotCameraId}>
                            <option value="">Default</option>
                            {snapshotConfig.robotics.cameras.map((camera) => (
                              <option key={camera.id} value={camera.id}>
                                {camera.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="field-group">
                          <label
                            className="field-label"
                            htmlFor={`robot-mission-snapshot-stream-scene-${entry.id}`}>
                            Snapshot Stream Scene
                          </label>
                          <select
                            id={`robot-mission-snapshot-stream-scene-${entry.id}`}
                            className="manifest-select"
                            onChange={(event) =>
                              updateRobotMissionDraftBundleSnapshotMissionField(
                                entry.id,
                                'streamSceneId',
                                event.target.value
                              )
                            }
                            value={entry.bundle.mission.streamSceneId}>
                            <option value="">Default</option>
                            {snapshotConfig.streamScenes.map((streamScene) => (
                              <option key={streamScene.id} value={streamScene.id}>
                                {streamScene.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <p className="panel-note">
                          draft file {buildRobotMissionDraftBundleFileName(
                            entry.bundle,
                            activeConfig.fragmentId
                          )}
                        </p>
                        <p className="panel-note">
                          artifact file {buildRobotMissionArtifactPackFileName(
                            entry.bundle,
                            activeConfig.fragmentId
                          )}
                        </p>
                        <p className="panel-note">
                          published file {snapshotPreview.fileName}
                        </p>
                        <p className="panel-note">
                          preflight fragment {snapshotPreview.fragmentId} / route id {snapshotPreview.routeId} / mission id {snapshotPreview.missionId}
                        </p>
                        <p className="panel-note">
                          world asset {snapshotPreview.payload.world.assetLabel || 'none'} / frame {snapshotPreview.payload.world.frameId || 'none'} / zone {snapshotPreview.payload.zoneMapUrl || 'none'}
                        </p>
                        <p className="panel-note">
                          preflight mission {snapshotPreview.payload.label || 'none'} / desc {snapshotPreview.payload.description || 'none'} / fragment label {snapshotPreview.payload.fragmentLabel || 'none'}
                        </p>
                        <p className="panel-note">
                          preflight route meta {snapshotPreview.routeLabel} / accent {snapshotPreview.routeAccent} / startup {snapshotPreview.payload.startupMode} / preset {snapshotPreview.payload.cameraPresetId || 'none'} / robot camera {snapshotPreview.payload.robotCameraId || 'none'} / scene {snapshotPreview.payload.streamSceneId || 'none'}
                        </p>
                        <p className="panel-note">
                          preflight route desc {snapshotPreview.routeDescription} / preset label {snapshotPreview.cameraPresetLabel} / robot camera label {snapshotPreview.robotCameraLabel} / scene label {snapshotPreview.streamSceneLabel}
                        </p>
                        <p className="panel-note">
                          launch {snapshotPreview.payload.launchUrl}
                        </p>
                        <p className="panel-note">
                          fragment {snapshotPreview.payload.fragmentId || 'none'} / label {snapshotPreview.payload.fragmentLabel || 'none'}
                        </p>
                        <p className="panel-note">
                          route file {(extractRobotRouteIdFromUrl(snapshotPreview.payload.routeUrl) || 'none')}.json
                        </p>
                        <p className="panel-note">
                          route {snapshotPreview.payload.routeUrl || 'none'} / zone {snapshotPreview.payload.zoneMapUrl || 'none'}
                        </p>
                        <p className="panel-note">
                          route meta {entry.bundle.route.label || 'none'} / accent {entry.bundle.route.accent || 'none'}
                        </p>
                        <p className="panel-note">
                          accent {snapshotPreview.payload.accent || 'none'}
                        </p>
                        <p className="panel-note">
                          world {snapshotPreview.payload.world.assetLabel || 'none'} / frame {snapshotPreview.payload.world.frameId || 'none'}
                        </p>
                        <p className="panel-note">
                          effective {snapshotPreview.payload.startupMode} / preset {snapshotPreview.payload.cameraPresetId || 'none'} / robot camera {snapshotPreview.payload.robotCameraId || 'none'} / scene {snapshotPreview.payload.streamSceneId || 'none'}
                        </p>
                        <div className="chip-list">
                          <button
                            className="chip"
                            onClick={() => applyRobotMissionDraftBundleSnapshot(entry)}
                            type="button">
                            Apply
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotBundleJson(entry)
                            }
                            type="button">
                            Copy Bundle
                          </button>
                          <button
                            className="chip"
                            onClick={() => downloadRobotMissionDraftBundleSnapshot(entry)}
                            type="button">
                            Download Bundle
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotMissionJson(entry)
                            }
                            type="button">
                            Copy Mission
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotMissionJson(entry)
                            }
                            type="button">
                            Download Mission
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotPublishedPreview(entry)
                            }
                            type="button">
                            Copy Preview
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotPublishedPreview(entry)
                            }
                            type="button">
                            Download Preview
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotLaunchUrl(entry)
                            }
                            type="button">
                            Copy Launch
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotPreflightSummary(entry)
                            }
                            type="button">
                            Copy Preflight
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotPublishReport(entry)
                            }
                            type="button">
                            Copy Report
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotLaunchUrl(entry)
                            }
                            type="button">
                            Download Launch
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotPublishReport(entry)
                            }
                            type="button">
                            Download Report
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotPreflightSummary(entry)
                            }
                            type="button">
                            Download Preflight
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotValidateCommand(entry)
                            }
                            type="button">
                            Copy Validate
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotReleaseCommand(entry)
                            }
                            type="button">
                            Copy Release
                          </button>
                          <button
                            className="chip"
                            onClick={() => copyRobotMissionDraftBundleSnapshotPublishCommand(entry)}
                            type="button">
                            Copy Publish
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotValidateCommand(entry)
                            }
                            type="button">
                            Download Validate
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotReleaseCommand(entry)
                            }
                            type="button">
                            Download Release
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotPublishCommand(entry)
                            }
                            type="button">
                            Download Publish
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              copyRobotMissionDraftBundleSnapshotArtifactPack(entry)
                            }
                            type="button">
                            Copy Artifacts
                          </button>
                          <button
                            className="chip"
                            onClick={() =>
                              downloadRobotMissionDraftBundleSnapshotArtifactPack(entry)
                            }
                            type="button">
                            Download Artifacts
                          </button>
                          <button
                            className="chip"
                            onClick={() => deleteRobotMissionDraftBundleSnapshot(entry.id)}
                            type="button">
                            Delete
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </div>
            <ul className="plain-list">
              <li>R: Robot Mode toggle</li>
              <li>W / A / S / D: teleop step</li>
              <li>Arrow Keys: teleop step</li>
              <li>V: Drop Waypoint</li>
              <li>C: Clear Route</li>
              <li>Gamepad: left stick move/turn, A waypoint, X clear waypoint, B clear route, Y reset pose, LB/RB camera</li>
              <li>Robot Bridge: `?robotBridge=1` or `?robotBridgeUrl=ws://...`</li>
              <li>1 / 2 / 3: world camera preset に戻る</li>
            </ul>
          </>
        ) : isPhotoMode ? (
          <>
            <div className="chip-list">
              {activeConfig.photoRatios.map((ratio) => (
                <button
                  key={ratio.id}
                  className={ratio.id === selectedRatioId ? 'chip active' : 'chip'}
                  onClick={() => setSelectedRatioId(ratio.id)}
                  type="button">
                  {ratio.label}
                </button>
              ))}
            </div>
            <button className="primary-button" onClick={handleCaptureClick} type="button">
              Save PNG
            </button>
            <button
              className="ghost-button"
              onClick={() => setShowGuides((current) => !current)}
              type="button">
              {showGuides ? 'Hide Guides' : 'Show Guides'}
            </button>
            <p className="panel-note">
              撮影ガイドは DOM overlay なので PNG には焼き込まれません。
            </p>
          </>
        ) : isLiveMode ? (
          <>
            <p className="panel-note">
              VTuber 本体はまず OBS 合成前提。ここは背景ステージと chapter ごとの定点カメラ管理を担当します。
            </p>
            <div className="chip-list">
              {resolvedStreamScenes.map((streamScene, index) => (
                <button
                  key={streamScene.id}
                  className={streamScene.id === selectedStreamScene?.id ? 'chip active' : 'chip'}
                  onClick={() => activateStreamScene(streamScene)}
                  type="button">
                  {streamSceneKeys[index] ? `${streamSceneKeys[index]}. ${streamScene.label}` : streamScene.label}
                </button>
              ))}
            </div>
            <p className="panel-note">
              {selectedStreamScene
                ? `${selectedStreamScene.title}: ${selectedStreamScene.topic}`
                : 'stream scene を選ぶと camera preset と overlay の文言が切り替わります。'}
            </p>
            <div className="chip-list">
              {activeConfig.overlayPresets.map((overlayPreset, index) => (
                <button
                  key={overlayPreset.id}
                  className={overlayPreset.id === selectedOverlayPreset.id ? 'chip active' : 'chip'}
                  onClick={() => activateOverlayPreset(overlayPreset)}
                  type="button">
                  {overlayPresetKeys[index]
                    ? `${overlayPresetKeys[index]}. ${overlayPreset.label}`
                    : overlayPreset.label}
                </button>
              ))}
            </div>
            <p className="panel-note">{selectedOverlayPreset.description}</p>
            <h2>Stream Scene Workspace</h2>
            <p className="status-label">
              {sceneWorkspaceModeLabel}
              {isSceneWorkspaceDirty ? ' / Unsaved Draft' : ''}
            </p>
            <p className="panel-note">
              stream scene の title / topic / memo / branding をこの場で編集できます。保存すると次回も同じ scene で始められます。
            </p>
            <div className="button-stack">
              <button className="primary-button" onClick={saveSceneWorkspace} type="button">
                Save Scene Workspace
              </button>
              <button className="ghost-button" onClick={copySceneWorkspaceJson} type="button">
                Copy Scene Workspace JSON
              </button>
              <button className="ghost-button" onClick={downloadSceneWorkspaceJson} type="button">
                Download Scene Workspace JSON
              </button>
              <button className="ghost-button" onClick={resetSceneWorkspace} type="button">
                Reset Scene Workspace
              </button>
            </div>
            <div className="button-stack">
              <button
                className="ghost-button"
                onClick={() => sceneWorkspaceFileInputRef.current?.click()}
                type="button">
                Import Scene Workspace File
              </button>
              <input
                ref={sceneWorkspaceFileInputRef}
                accept="application/json,.json"
                className="manifest-file-input"
                onChange={handleSceneWorkspaceFileImport}
                type="file"
              />
              <button className="ghost-button" onClick={applySceneWorkspaceImportText} type="button">
                Apply Pasted Scene Workspace JSON
              </button>
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-json-import">
                Scene Workspace JSON Import
              </label>
              <textarea
                id="scene-workspace-json-import"
                className="manifest-textarea manifest-textarea-compact"
                onChange={(event) => {
                  setSceneWorkspaceImportText(event.target.value);
                  setSceneWorkspaceImportError('');
                }}
                placeholder='{"fragments":{"residency":{"streamScenes":[{"id":"window-talk","title":"Long Use Talk"}]}}}'
                value={sceneWorkspaceImportText}
              />
              {sceneWorkspaceImportError ? (
                <p className="panel-note panel-note-error">
                  Import Error: {sceneWorkspaceImportError}
                </p>
              ) : (
                <p className="panel-note">
                  paste 後に `Apply Pasted Scene Workspace JSON` を押すと draft に反映されます。
                </p>
              )}
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-label">
                Scene Label
              </label>
              <input
                id="scene-workspace-label"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceSceneField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'label',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.label ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-title">
                Scene Workspace Title
              </label>
              <input
                id="scene-workspace-title"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceSceneField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'title',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.title ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-topic">
                Scene Topic
              </label>
              <input
                id="scene-workspace-topic"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceSceneField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'topic',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.topic ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-preset">
                Scene Camera Preset
              </label>
              <select
                id="scene-workspace-preset"
                className="manifest-input"
                onChange={(event) => {
                  updateSceneWorkspaceSceneField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'presetId',
                    event.target.value
                  );
                  setSelectedPresetId(event.target.value);
                }}
                value={selectedStreamScene?.presetId ?? activeConfig.cameraPresets[0]?.id ?? ''}>
                {activeConfig.cameraPresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-memo-title">
                Scene Memo Title
              </label>
              <input
                id="scene-workspace-memo-title"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceMemoField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'title',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.overlayMemo?.title ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-memo-items">
                Scene Memo Items
              </label>
              <textarea
                id="scene-workspace-memo-items"
                className="manifest-textarea"
                onChange={(event) =>
                  updateSceneWorkspaceMemoField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'items',
                    event.target.value
                  )
                }
                value={normalizeOverlayMemoItems(selectedStreamScene?.overlayMemo?.items).join('\n')}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-memo-footer">
                Scene Memo Footer
              </label>
              <input
                id="scene-workspace-memo-footer"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceMemoField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'footer',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.overlayMemo?.footer ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-badge">
                Overlay Badge
              </label>
              <input
                id="scene-workspace-badge"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceBrandingField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'badge',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.overlayBrandingOverrides?.badge ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-strapline">
                Overlay Strapline
              </label>
              <input
                id="scene-workspace-strapline"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceBrandingField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'strapline',
                    event.target.value
                  )
                }
                type="text"
                value={selectedStreamScene?.overlayBrandingOverrides?.strapline ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-accent">
                Overlay Accent
              </label>
              <input
                id="scene-workspace-accent"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceBrandingField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'accent',
                    event.target.value
                  )
                }
                placeholder={activeConfig.overlayBranding.accent}
                type="text"
                value={selectedStreamScene?.overlayBrandingOverrides?.accent ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-highlight">
                Overlay Highlight
              </label>
              <input
                id="scene-workspace-highlight"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceBrandingField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'highlight',
                    event.target.value
                  )
                }
                placeholder={activeConfig.overlayBranding.highlight}
                type="text"
                value={selectedStreamScene?.overlayBrandingOverrides?.highlight ?? ''}
              />
            </div>
            <div className="field-group">
              <label className="field-label" htmlFor="scene-workspace-glow">
                Overlay Glow
              </label>
              <input
                id="scene-workspace-glow"
                className="manifest-input"
                onChange={(event) =>
                  updateSceneWorkspaceBrandingField(
                    activeConfig.fragmentId,
                    selectedStreamScene?.id,
                    'glow',
                    event.target.value
                  )
                }
                placeholder={activeConfig.overlayBranding.glow}
                type="text"
                value={selectedStreamScene?.overlayBrandingOverrides?.glow ?? ''}
              />
            </div>
            <div className="chip-list">
              <button className="primary-button" onClick={copyLiveSceneJson} type="button">
                Copy Scene JSON
              </button>
              <button className="ghost-button" onClick={downloadLiveSceneJson} type="button">
                Download Scene JSON
              </button>
              <button className="ghost-button" onClick={copyOverlayUrl} type="button">
                Copy Overlay URL
              </button>
              <button className="ghost-button" onClick={openOverlayView} type="button">
                Open Overlay View
              </button>
            </div>
            <pre className="json-preview" aria-label="Live Scene JSON">
              <code>{liveScenePayloadJson}</code>
            </pre>
            <ul className="plain-list">
              <li>Left zone: avatar overlay 想定</li>
              <li>Upper zone: title / schedule card</li>
              <li>Right zone: chat or topic memo</li>
              <li>H: home spot へ戻る</li>
              <li>4 / 5 / 6: stream scene</li>
              <li>7 / 8 / 9: overlay preset</li>
              <li>Overlay transport: {overlayTransportLabel}</li>
              <li>Overlay view: {relayConfig.enabled ? '`/overlay.html?relay=1`' : '`/overlay.html`'}</li>
              {relayConfig.enabled ? <li>Relay URL: {relayConfig.url}</li> : null}
              <li>X: Walk Mode toggle</li>
            </ul>
          </>
        ) : (
          <ul className="plain-list">
            <li>R: Robot Mode</li>
            <li>X: Walk Mode</li>
            <li>F: Interact</li>
            <li>Space: Jump</li>
            <li>4 / 5 / 6: Live Scenes</li>
            <li>7 / 8 / 9: Overlay Presets</li>
            <li>P: Photo Mode</li>
            <li>L: Live Mode</li>
            <li>1 / 2 / 3: Camera Presets</li>
            <li>V: Robot Waypoint</li>
            <li>G: Guide Toggle</li>
            <li>K: PNG Capture</li>
            <li>H: Home Preset</li>
          </ul>
        )}

        <h2>Fragment Hotspots</h2>
        <div className="chip-list">
          {activeConfig.hotspots.map((hotspot) => (
            <button
              key={hotspot.id}
              className={hotspot.id === selectedHotspotId ? 'chip active' : 'chip'}
              disabled={isWalkMode || isRobotMode}
              onClick={() => handleHotspotActivate(hotspot)}
              type="button">
              {hotspot.label}
            </button>
          ))}
        </div>
        {isWalkMode || isRobotMode ? (
          <p className="panel-note">
            {isRobotMode
              ? 'Robot Mode 中は robot marker と teleop を優先するため hotspot の画面ピンを隠しています。'
              : 'Walk 中は world 側の視認を優先するため hotspot の画面ピンを隠しています。'}
          </p>
        ) : null}

        <h2>Collectibles</h2>
        <div className="chip-list">
          {activeConfig.shards.map((shard) => (
            <button
              key={shard.id}
              className={collectedShardIds.includes(shard.id) ? 'chip active' : 'chip'}
              disabled={isWalkMode || isRobotMode || collectedShardIds.includes(shard.id)}
              onClick={() => handleLoopItemActivate(shard)}
              type="button">
              {collectedShardIds.includes(shard.id) ? `${shard.label} Collected` : shard.label}
            </button>
          ))}
          <button
            className={isGateUnlocked ? 'chip active' : 'chip'}
            disabled={isWalkMode || isRobotMode}
            onClick={() => handleLoopItemActivate(gateCard)}
            type="button">
            {gateCard.label}
          </button>
        </div>
      </aside>

      <footer className="statusbar glass-panel">
        <span className="status-label">STATUS</span>
        <span>{statusMessage}</span>
      </footer>

      <EchoNoteModal
        hotspot={activeModalItem}
        onClose={() => {
          setActiveModalItem(null);
          setSelectedHotspotId(null);
          setStatusMessage('Echo Note を閉じました');
        }}
      />
    </div>
  );
}
