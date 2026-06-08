import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  dreamwalkerConfig,
  resolveDreamwalkerConfig,
  resolveWorldAssetBundle
} from '../src/app-config.js';
import {
  buildWorldAssetHealth,
  normalizeLocalAssetPath,
  resolveBundleWorldHealth
} from '../src/studio-health.js';
import { summarizeRouteZoneCoverage } from '../src/robot-route-analysis.js';
import { buildSemanticZoneMap } from '../src/semantic-zones.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const defaultPublicRoot = path.join(appRoot, 'public');
let activePublicRoot = defaultPublicRoot;
let activeManifestInput = dreamwalkerConfig.assetManifest.defaultUrl;
let activeStudioBundleCatalogInput = dreamwalkerConfig.studioBundleCatalog.defaultUrl;
let activeRobotRouteCatalogInput = dreamwalkerConfig.robotRouteCatalog.defaultUrl;
let activeRobotMissionCatalogInput = dreamwalkerConfig.robotMissionCatalog.defaultUrl;
let activeMissionOnly = false;

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function isRemoteUrl(value) {
  return typeof value === 'string' && /^https?:\/\//i.test(value.trim());
}

function toAbsolutePath(inputPath, baseDir = process.cwd()) {
  if (!inputPath) {
    return '';
  }

  return path.isAbsolute(inputPath)
    ? inputPath
    : path.resolve(baseDir, inputPath);
}

function parseArgs(argv) {
  const args = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token.startsWith('--')) {
      continue;
    }

    const key = token.slice(2);
    if (key === 'help' || key === 'mission-only') {
      args[key] = true;
      continue;
    }

    const nextToken = argv[index + 1];
    if (!nextToken || nextToken.startsWith('--')) {
      throw new Error(`値が必要です: --${key}`);
    }

    args[key] = nextToken;
    index += 1;
  }

  return args;
}

function printUsage() {
  console.log(`Dynamic Map Viewer studio validation

Usage:
  node ./tools/validate-studio-assets.mjs
  node ./tools/validate-studio-assets.mjs --public-root /tmp/dreamwalker-public --mission-only

Options:
  --public-root <dir>           custom public root. default: apps/dreamwalker-web/public
  --asset-manifest <file|url>   custom asset manifest input.
  --bundle-catalog <file|url>   custom studio bundle catalog input.
  --robot-route-catalog <file|url> custom robot route catalog input.
  --robot-mission-catalog <file|url> custom robot mission catalog input.
  --mission-only                validate robot routes + zone maps only.
  --help                        show this message.
`);
}

function formatScope(scope) {
  return scope.padEnd(32, ' ');
}

function toPublicFilePath(assetUrl) {
  const localPath = normalizeLocalAssetPath(assetUrl);

  if (!localPath) {
    return '';
  }

  return path.join(activePublicRoot, localPath.replace(/^\/+/, ''));
}

async function fileExists(filePath) {
  if (!filePath) {
    return undefined;
  }

  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFromPublicUrl(assetUrl) {
  const filePath = toPublicFilePath(assetUrl);

  if (!filePath) {
    throw new Error(`local public path に解決できません: ${assetUrl}`);
  }

  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function readJsonInput(input) {
  if (isRemoteUrl(input)) {
    const response = await fetch(input);

    if (!response.ok) {
      throw new Error(`download に失敗しました: ${response.status} ${response.statusText}`);
    }

    return JSON.parse(await response.text());
  }

  if (hasNonEmptyString(input)) {
    const normalizedInput = String(input).trim();

    if (path.isAbsolute(normalizedInput)) {
      if (await fileExists(normalizedInput)) {
        const raw = await readFile(normalizedInput, 'utf8');
        return JSON.parse(raw);
      }

      if (normalizedInput.startsWith('/')) {
        return readJsonFromPublicUrl(normalizedInput);
      }
    }
  }

  const raw = await readFile(toAbsolutePath(input), 'utf8');
  return JSON.parse(raw);
}

function normalizeRoutePosition(positionLike) {
  if (!Array.isArray(positionLike) || positionLike.length < 3) {
    return null;
  }

  const position = positionLike
    .slice(0, 3)
    .map((value) => (typeof value === 'number' ? value : Number(value)));
  return position.every((value) => Number.isFinite(value)) ? position : null;
}

function normalizeRoutePose(poseLike) {
  if (!poseLike || typeof poseLike !== 'object') {
    return null;
  }

  const position = normalizeRoutePosition(poseLike.position);
  const yawDegrees = Number(poseLike.yawDegrees);

  if (!position || !Number.isFinite(yawDegrees)) {
    return null;
  }

  return {
    position,
    yawDegrees
  };
}

function normalizeRouteWorld(worldLike, fallbackLike = {}) {
  const world = worldLike && typeof worldLike === 'object' ? worldLike : {};
  const fallback =
    fallbackLike && typeof fallbackLike === 'object' ? fallbackLike : {};

  return {
    fragmentId: hasNonEmptyString(world.fragmentId)
      ? world.fragmentId.trim()
      : hasNonEmptyString(fallback.fragmentId)
        ? fallback.fragmentId.trim()
        : '',
    fragmentLabel: hasNonEmptyString(world.fragmentLabel)
      ? world.fragmentLabel.trim()
      : hasNonEmptyString(fallback.fragmentLabel)
        ? fallback.fragmentLabel.trim()
        : '',
    assetLabel: hasNonEmptyString(world.assetLabel) ? world.assetLabel.trim() : '',
    manifestLabel: hasNonEmptyString(world.manifestLabel) ? world.manifestLabel.trim() : '',
    splatUrl: hasNonEmptyString(world.splatUrl) ? world.splatUrl.trim() : '',
    colliderMeshUrl: hasNonEmptyString(world.colliderMeshUrl)
      ? world.colliderMeshUrl.trim()
      : '',
    frameId: hasNonEmptyString(world.frameId)
      ? world.frameId.trim()
      : hasNonEmptyString(fallback.frameId)
        ? fallback.frameId.trim()
        : 'dreamwalker_map',
    zoneMapUrl: hasNonEmptyString(world.zoneMapUrl) ? world.zoneMapUrl.trim() : '',
    usesDemoFallback: Boolean(world.usesDemoFallback)
  };
}

function normalizeRobotRoute(routeLike) {
  const route = routeLike && typeof routeLike === 'object' ? routeLike : {};
  const pose = normalizeRoutePose(route.pose);
  const points = Array.isArray(route.route)
    ? route.route.map((position) => normalizeRoutePosition(position)).filter(Boolean)
    : [];

  if (!pose && points.length === 0) {
    throw new Error('robot route must contain pose or route');
  }

  const normalizedPose = pose ?? {
    position: [...points[points.length - 1]],
    yawDegrees: 0
  };
  const normalizedPoints =
    points.length > 0 ? points : [[...normalizedPose.position]];
  const world = normalizeRouteWorld(route.world, {
    fragmentId: route.fragmentId,
    fragmentLabel: route.fragmentLabel,
    frameId: route.frameId
  });

  return {
    label: hasNonEmptyString(route.label) ? route.label.trim() : '',
    fragmentId: world.fragmentId,
    fragmentLabel: world.fragmentLabel,
    frameId: world.frameId,
    pose: normalizedPose,
    route: normalizedPoints,
    world
  };
}

function normalizeRobotMission(missionLike) {
  const mission = missionLike && typeof missionLike === 'object' ? missionLike : {};
  const world = mission.world && typeof mission.world === 'object' ? mission.world : {};
  const startupMode = hasNonEmptyString(mission.startupMode)
    ? mission.startupMode.trim()
    : '';

  return {
    id: hasNonEmptyString(mission.id) ? mission.id.trim() : '',
    label: hasNonEmptyString(mission.label) ? mission.label.trim() : '',
    description: hasNonEmptyString(mission.description) ? mission.description.trim() : '',
    fragmentId: hasNonEmptyString(mission.fragmentId) ? mission.fragmentId.trim() : '',
    fragmentLabel: hasNonEmptyString(mission.fragmentLabel) ? mission.fragmentLabel.trim() : '',
    accent: hasNonEmptyString(mission.accent) ? mission.accent.trim() : '',
    routeUrl: hasNonEmptyString(mission.routeUrl) ? mission.routeUrl.trim() : '',
    zoneMapUrl: hasNonEmptyString(mission.zoneMapUrl) ? mission.zoneMapUrl.trim() : '',
    launchUrl: hasNonEmptyString(mission.launchUrl) ? mission.launchUrl.trim() : '',
    cameraPresetId: hasNonEmptyString(mission.cameraPresetId) ? mission.cameraPresetId.trim() : '',
    robotCameraId: hasNonEmptyString(mission.robotCameraId) ? mission.robotCameraId.trim() : '',
    streamSceneId: hasNonEmptyString(mission.streamSceneId) ? mission.streamSceneId.trim() : '',
    startupMode,
    world: {
      assetLabel: hasNonEmptyString(world.assetLabel) ? world.assetLabel.trim() : '',
      frameId: hasNonEmptyString(world.frameId) ? world.frameId.trim() : ''
    }
  };
}

function mergeStatus(currentStatus, nextStatus) {
  const rank = {
    ready: 0,
    warning: 1,
    error: 2
  };

  return (rank[nextStatus] ?? 0) > (rank[currentStatus] ?? 0)
    ? nextStatus
    : currentStatus;
}

async function validateWorldAsset(scope, fragmentId, workspaceLike) {
  const activeConfig = resolveDreamwalkerConfig(fragmentId);
  const assetBundle = resolveWorldAssetBundle(activeConfig, workspaceLike);
  const splatExists = await fileExists(toPublicFilePath(assetBundle.splatUrl));
  const colliderExists = await fileExists(toPublicFilePath(assetBundle.colliderMeshUrl));
  const health = buildWorldAssetHealth(assetBundle, {
    splatExists,
    colliderExists
  });

  return {
    scope,
    status: health.status,
    message: `${health.label}: ${health.detail}`
  };
}

async function validateZoneMap(scope, zoneUrl, options = {}) {
  if (!zoneUrl) {
    return {
      scope,
      status: 'warning',
      message: 'zone map URL が未設定です'
    };
  }

  try {
    const payload = await readJsonFromPublicUrl(zoneUrl);
    const zoneMap = buildSemanticZoneMap(payload);
    const expectedFragmentId =
      options && hasNonEmptyString(options.fragmentId) ? options.fragmentId.trim() : '';
    let status = 'ready';
    const notes = [
      `${zoneMap.zones.length} zones`,
      `frame ${zoneMap.frameId}`,
      `resolution ${zoneMap.resolution}`
    ];

    if (expectedFragmentId && !zoneUrl.includes(expectedFragmentId)) {
      status = 'warning';
      notes.push(`path does not mention fragmentId=${expectedFragmentId}`);
    }

    return {
      scope,
      status,
      message: notes.join(' / ')
    };
  } catch (error) {
    return {
      scope,
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    };
  }
}

async function validateManifest() {
  const findings = [];
  const manifestUrl = activeManifestInput;

  try {
    const manifest = await readJsonInput(manifestUrl);
    findings.push({
      scope: 'manifest',
      status: 'ready',
      message: `${manifest.label ?? 'Asset Manifest'} を読み込みました`
    });

    for (const fragmentId of Object.keys(dreamwalkerConfig.fragments)) {
      findings.push(
        await validateWorldAsset(`manifest:${fragmentId}`, fragmentId, manifest)
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    findings.push({
      scope: 'manifest',
      status: 'error',
      message
    });
  }

  return findings;
}

async function validateRobotRoute(scope, routeLike, entryLike = null) {
  try {
    const route = normalizeRobotRoute(routeLike);
    let status = 'ready';
    const notes = [];
    const entryFragmentId =
      entryLike && hasNonEmptyString(entryLike.fragmentId)
        ? entryLike.fragmentId.trim()
        : '';
    const routeFragmentId = route.world.fragmentId || route.fragmentId;
    const routeLabel = route.label || routeFragmentId || 'Robot Route';
    const hasWorldMetadata = Boolean(
      route.world.assetLabel ||
        route.world.manifestLabel ||
        route.world.splatUrl ||
        route.world.colliderMeshUrl
    );

    if (!hasWorldMetadata) {
      status = mergeStatus(status, 'warning');
      notes.push('world metadata が無いため legacy route 扱いです');
    }

    if (entryFragmentId && routeFragmentId && entryFragmentId !== routeFragmentId) {
      status = mergeStatus(status, 'warning');
      notes.push(`catalog fragmentId=${entryFragmentId} / route fragmentId=${routeFragmentId}`);
    }

    if (routeFragmentId && !dreamwalkerConfig.fragments[routeFragmentId]) {
      status = mergeStatus(status, 'warning');
      notes.push(`unknown fragmentId: ${routeFragmentId}`);
    }

    const localSplatExists = await fileExists(toPublicFilePath(route.world.splatUrl));
    if (route.world.splatUrl && normalizeLocalAssetPath(route.world.splatUrl) && localSplatExists === false) {
      status = mergeStatus(status, 'error');
      notes.push(`missing local splat: ${route.world.splatUrl}`);
    }

    const localColliderExists = await fileExists(toPublicFilePath(route.world.colliderMeshUrl));
    if (
      route.world.colliderMeshUrl &&
      normalizeLocalAssetPath(route.world.colliderMeshUrl) &&
      localColliderExists === false
    ) {
      status = mergeStatus(status, 'warning');
      notes.push(`missing local collider: ${route.world.colliderMeshUrl}`);
    }

    if (route.world.zoneMapUrl) {
      if (normalizeLocalAssetPath(route.world.zoneMapUrl)) {
        try {
          const zonePayload = await readJsonFromPublicUrl(route.world.zoneMapUrl);
          const zoneMap = buildSemanticZoneMap(zonePayload);
          const routeCoverage = summarizeRouteZoneCoverage(route, zoneMap);

          if (route.frameId !== zoneMap.frameId) {
            status = mergeStatus(status, 'warning');
            notes.push(`zone frame drift: route=${route.frameId} / zone=${zoneMap.frameId}`);
          }

          if (routeFragmentId && !route.world.zoneMapUrl.includes(routeFragmentId)) {
            status = mergeStatus(status, 'warning');
            notes.push(`zone path does not mention fragmentId=${routeFragmentId}`);
          }

          if (routeCoverage.outsideBoundsCount > 0) {
            status = mergeStatus(status, 'warning');
            notes.push(`route leaves zone bounds at ${routeCoverage.outsideBoundsCount} node(s)`);
          }

          if (routeCoverage.hazardNodeCount > 0) {
            status = mergeStatus(status, 'warning');
            notes.push(`hazard nodes ${routeCoverage.hazardNodeCount}/${route.route.length}`);
          }

          notes.push(
            `zoneMap ${path.basename(route.world.zoneMapUrl)} / ${zoneMap.zones.length} zones / coverage ${routeCoverage.hitNodeCount}/${route.route.length} / maxCost ${routeCoverage.maxCost}`
          );

          if (routeCoverage.labels.length > 0) {
            notes.push(`zones ${routeCoverage.labels.slice(0, 3).join(', ')}`);
          }
        } catch (error) {
          status = mergeStatus(status, 'warning');
          notes.push(
            `zone map invalid: ${error instanceof Error ? error.message : String(error)}`
          );
        }
      } else {
        notes.push(`zoneMap remote: ${route.world.zoneMapUrl}`);
      }
    }

    const baseLabel =
      status === 'ready'
        ? 'Route Ready'
        : status === 'warning'
          ? hasWorldMetadata
            ? 'Route Warning'
            : 'Legacy Route'
          : 'Route Error';
    const worldLabel =
      route.world.assetLabel || route.world.fragmentLabel || routeLabel;
    const messageParts = [`${routeLabel} / ${worldLabel} / frame ${route.frameId}`];

    if (notes.length > 0) {
      messageParts.push(...notes);
    }

    return {
      scope,
      status,
      message: `${baseLabel}: ${messageParts.join(' / ')}`
    };
  } catch (error) {
    return {
      scope,
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    };
  }
}

async function validateRobotMission(scope, missionLike, entryLike = null, missionSourceUrl = '') {
  try {
    const mission = normalizeRobotMission(missionLike);
    let status = 'ready';
    const notes = [];
    const entryFragmentId =
      entryLike && hasNonEmptyString(entryLike.fragmentId)
        ? entryLike.fragmentId.trim()
        : '';
    const missionFragmentId = mission.fragmentId;
    const missionLabel = mission.label || mission.id || missionFragmentId || 'Robot Mission';
    const hasKnownFragment =
      missionFragmentId && Object.prototype.hasOwnProperty.call(dreamwalkerConfig.fragments, missionFragmentId);
    const fragmentConfig = hasKnownFragment
      ? resolveDreamwalkerConfig(missionFragmentId)
      : null;

    if (!mission.routeUrl) {
      status = mergeStatus(status, 'error');
      notes.push('routeUrl が空です');
    }

    if (!mission.zoneMapUrl) {
      status = mergeStatus(status, 'warning');
      notes.push('zoneMapUrl が空です');
    }

    if (entryFragmentId && missionFragmentId && entryFragmentId !== missionFragmentId) {
      status = mergeStatus(status, 'warning');
      notes.push(`catalog fragmentId=${entryFragmentId} / mission fragmentId=${missionFragmentId}`);
    }

    if (missionFragmentId && !hasKnownFragment) {
      status = mergeStatus(status, 'warning');
      notes.push(`unknown fragmentId=${missionFragmentId}`);
    }

    if (fragmentConfig) {
      if (
        mission.cameraPresetId &&
        !fragmentConfig.cameraPresets.some((preset) => preset.id === mission.cameraPresetId)
      ) {
        status = mergeStatus(status, 'warning');
        notes.push(`invalid cameraPresetId=${mission.cameraPresetId}`);
      }

      if (
        mission.robotCameraId &&
        !fragmentConfig.robotics.cameras.some((camera) => camera.id === mission.robotCameraId)
      ) {
        status = mergeStatus(status, 'warning');
        notes.push(`invalid robotCameraId=${mission.robotCameraId}`);
      }

      if (
        mission.streamSceneId &&
        !fragmentConfig.streamScenes.some((scene) => scene.id === mission.streamSceneId)
      ) {
        status = mergeStatus(status, 'warning');
        notes.push(`invalid streamSceneId=${mission.streamSceneId}`);
      }
    }

    if (
      mission.startupMode &&
      !['explore', 'live', 'photo', 'robot'].includes(mission.startupMode)
    ) {
      status = mergeStatus(status, 'warning');
      notes.push(`invalid startupMode=${mission.startupMode}`);
    }

    let route = null;
    let zoneMap = null;

    if (mission.routeUrl) {
      const localRouteExists = await fileExists(toPublicFilePath(mission.routeUrl));

      if (normalizeLocalAssetPath(mission.routeUrl) && localRouteExists === false) {
        status = mergeStatus(status, 'error');
        notes.push(`missing local route: ${mission.routeUrl}`);
      } else if (localRouteExists !== false) {
        try {
          route = normalizeRobotRoute(await readJsonFromPublicUrl(mission.routeUrl));
          notes.push(`route ${path.basename(mission.routeUrl)} loaded`);

          if (missionFragmentId && route.fragmentId && missionFragmentId !== route.fragmentId) {
            status = mergeStatus(status, 'warning');
            notes.push(`route fragment drift: mission=${missionFragmentId} / route=${route.fragmentId}`);
          }

          if (mission.world.frameId && route.frameId !== mission.world.frameId) {
            status = mergeStatus(status, 'warning');
            notes.push(`route frame drift: mission=${mission.world.frameId} / route=${route.frameId}`);
          }
        } catch (error) {
          status = mergeStatus(status, 'error');
          notes.push(`route invalid: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }

    if (mission.zoneMapUrl) {
      const localZoneExists = await fileExists(toPublicFilePath(mission.zoneMapUrl));

      if (normalizeLocalAssetPath(mission.zoneMapUrl) && localZoneExists === false) {
        status = mergeStatus(status, 'error');
        notes.push(`missing local zoneMap: ${mission.zoneMapUrl}`);
      } else if (localZoneExists !== false) {
        try {
          zoneMap = buildSemanticZoneMap(await readJsonFromPublicUrl(mission.zoneMapUrl));
          notes.push(`zoneMap ${path.basename(mission.zoneMapUrl)} / ${zoneMap.zones.length} zones`);

          if (missionFragmentId && !mission.zoneMapUrl.includes(missionFragmentId)) {
            status = mergeStatus(status, 'warning');
            notes.push(`zone path does not mention fragmentId=${missionFragmentId}`);
          }

          if (mission.world.frameId && zoneMap.frameId !== mission.world.frameId) {
            status = mergeStatus(status, 'warning');
            notes.push(`zone frame drift: mission=${mission.world.frameId} / zone=${zoneMap.frameId}`);
          }
        } catch (error) {
          status = mergeStatus(status, 'error');
          notes.push(`zone invalid: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }

    if (route && zoneMap) {
      const routeCoverage = summarizeRouteZoneCoverage(route, zoneMap);
      notes.push(`coverage ${routeCoverage.hitNodeCount}/${route.route.length} / maxCost ${routeCoverage.maxCost}`);

      if (routeCoverage.outsideBoundsCount > 0) {
        status = mergeStatus(status, 'warning');
        notes.push(`outside bounds ${routeCoverage.outsideBoundsCount}`);
      }

      if (routeCoverage.hazardNodeCount > 0) {
        status = mergeStatus(status, 'warning');
        notes.push(`hazard nodes ${routeCoverage.hazardNodeCount}/${route.route.length}`);
      }

      if (routeCoverage.labels.length > 0) {
        notes.push(`zones ${routeCoverage.labels.slice(0, 3).join(', ')}`);
      }
    }

    if (mission.launchUrl) {
      notes.push(`launch ${mission.launchUrl}`);

      try {
        const parsedLaunchUrl = new URL(mission.launchUrl, 'https://dreamwalker.invalid');
        const launchMissionUrl = hasNonEmptyString(parsedLaunchUrl.searchParams.get('robotMission'))
          ? parsedLaunchUrl.searchParams.get('robotMission').trim()
          : '';
        const launchRouteUrl = hasNonEmptyString(parsedLaunchUrl.searchParams.get('robotRoute'))
          ? parsedLaunchUrl.searchParams.get('robotRoute').trim()
          : '';

        if (missionSourceUrl) {
          if (launchMissionUrl) {
            if (launchMissionUrl !== missionSourceUrl) {
              status = mergeStatus(status, 'warning');
              notes.push(`launch mission drift: ${launchMissionUrl}`);
            }
          } else if (launchRouteUrl) {
            status = mergeStatus(status, 'warning');
            notes.push('launch uses legacy robotRoute query');

            if (mission.routeUrl && launchRouteUrl !== mission.routeUrl) {
              status = mergeStatus(status, 'warning');
              notes.push(`launch route drift: ${launchRouteUrl}`);
            }
          } else {
            status = mergeStatus(status, 'warning');
            notes.push('launch query missing robotMission');
          }
        } else if (!launchMissionUrl && launchRouteUrl) {
          status = mergeStatus(status, 'warning');
          notes.push('launch uses legacy robotRoute query');
        }
      } catch (error) {
        status = mergeStatus(status, 'warning');
        notes.push(`launch invalid: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    if (mission.cameraPresetId) {
      notes.push(`preset ${mission.cameraPresetId}`);
    }

    if (mission.robotCameraId) {
      notes.push(`robotCamera ${mission.robotCameraId}`);
    }

    if (mission.streamSceneId) {
      notes.push(`streamScene ${mission.streamSceneId}`);
    }

    if (mission.startupMode) {
      notes.push(`mode ${mission.startupMode}`);
    }

    const baseLabel =
      status === 'ready'
        ? 'Mission Ready'
        : status === 'warning'
          ? 'Mission Warning'
          : 'Mission Error';
    const worldLabel = mission.world.assetLabel || mission.fragmentLabel || missionLabel;

    return {
      scope,
      status,
      message: `${baseLabel}: ${missionLabel} / ${worldLabel}${notes.length > 0 ? ` / ${notes.join(' / ')}` : ''}`
    };
  } catch (error) {
    return {
      scope,
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    };
  }
}

async function validateCatalog() {
  const findings = [];
  const catalogUrl = activeStudioBundleCatalogInput;
  const validatedBundleUrls = new Set();

  try {
    const catalog = await readJsonInput(catalogUrl);
    const entries = Array.isArray(catalog.bundles) ? catalog.bundles : [];

    findings.push({
      scope: 'catalog',
      status: 'ready',
      message: `${catalog.label ?? 'Studio Bundle Catalog'} を読み込みました`
    });

    for (const [index, entry] of entries.entries()) {
      const label = typeof entry.label === 'string' && entry.label.trim()
        ? entry.label.trim()
        : `catalog-entry-${index + 1}`;
      const scope = `catalog:${label}`;
      const bundleUrl = typeof entry.url === 'string' ? entry.url.trim() : '';

      if (!bundleUrl) {
        findings.push({
          scope,
          status: 'error',
          message: 'bundle URL が空です'
        });
        continue;
      }

      validatedBundleUrls.add(bundleUrl);

      try {
        const bundle = await readJsonFromPublicUrl(bundleUrl);
        const splatExists = await fileExists(
          toPublicFilePath(
            resolveBundleWorldHealth(bundle, entry).assetBundle.splatUrl
          )
        );
        const colliderExists = await fileExists(
          toPublicFilePath(
            resolveBundleWorldHealth(bundle, entry).assetBundle.colliderMeshUrl
          )
        );
        const health = resolveBundleWorldHealth(bundle, entry, {
          splatExists,
          colliderExists
        });

        findings.push({
          scope,
          status: health.status,
          message: `${health.label}: ${health.detail}`
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        findings.push({
          scope,
          status: 'error',
          message
        });
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    findings.push({
      scope: 'catalog',
      status: 'error',
      message
    });
  }

  return {
    findings,
    validatedBundleUrls
  };
}

async function validateStandaloneBundles(validatedBundleUrls) {
  const findings = [];
  const bundleDir = path.join(activePublicRoot, 'studio-bundles');

  try {
    const entries = await readdir(bundleDir, { withFileTypes: true });
    const bundleFiles = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.json') && entry.name !== 'index.json')
      .map((entry) => `/studio-bundles/${entry.name}`)
      .filter((bundleUrl) => !validatedBundleUrls.has(bundleUrl));

    for (const bundleUrl of bundleFiles) {
      const label = path.basename(bundleUrl);
      const scope = `bundle:${label}`;

      try {
        const bundle = await readJsonFromPublicUrl(bundleUrl);
        const health = resolveBundleWorldHealth(bundle, null, {
          splatExists: await fileExists(
            toPublicFilePath(resolveBundleWorldHealth(bundle).assetBundle.splatUrl)
          ),
          colliderExists: await fileExists(
            toPublicFilePath(resolveBundleWorldHealth(bundle).assetBundle.colliderMeshUrl)
          )
        });

        findings.push({
          scope,
          status: health.status,
          message: `${health.label}: ${health.detail}`
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        findings.push({
          scope,
          status: 'error',
          message
        });
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    findings.push({
      scope: 'bundles',
      status: 'error',
      message
    });
  }

  return findings;
}

async function validateRobotRouteCatalog() {
  const findings = [];
  const catalogUrl = activeRobotRouteCatalogInput;
  const validatedRouteUrls = new Set();

  try {
    const catalog = await readJsonInput(catalogUrl);
    const entries = Array.isArray(catalog.routes) ? catalog.routes : [];

    findings.push({
      scope: 'robot-route-catalog',
      status: 'ready',
      message: `${catalog.label ?? 'Robot Route Catalog'} を読み込みました`
    });

    for (const [index, entry] of entries.entries()) {
      const label = hasNonEmptyString(entry.label)
        ? entry.label.trim()
        : `route-entry-${index + 1}`;
      const scope = `robot-route:${label}`;
      const routeUrl = hasNonEmptyString(entry.url) ? entry.url.trim() : '';

      if (!routeUrl) {
        findings.push({
          scope,
          status: 'error',
          message: 'route URL が空です'
        });
        continue;
      }

      validatedRouteUrls.add(routeUrl);

      try {
        const route = await readJsonFromPublicUrl(routeUrl);
        findings.push(await validateRobotRoute(scope, route, entry));
      } catch (error) {
        findings.push({
          scope,
          status: 'error',
          message: error instanceof Error ? error.message : String(error)
        });
      }
    }
  } catch (error) {
    findings.push({
      scope: 'robot-route-catalog',
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }

  return {
    findings,
    validatedRouteUrls
  };
}

async function validateStandaloneRobotRoutes(validatedRouteUrls) {
  const findings = [];
  const routeDir = path.join(activePublicRoot, 'robot-routes');

  try {
    const entries = await readdir(routeDir, { withFileTypes: true });
    const routeFiles = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.json') && entry.name !== 'index.json')
      .map((entry) => `/robot-routes/${entry.name}`)
      .filter((routeUrl) => !validatedRouteUrls.has(routeUrl));

    for (const routeUrl of routeFiles) {
      const label = path.basename(routeUrl);
      const scope = `robot-route-file:${label}`;

      try {
        const route = await readJsonFromPublicUrl(routeUrl);
        findings.push(await validateRobotRoute(scope, route));
      } catch (error) {
        findings.push({
          scope,
          status: 'error',
          message: error instanceof Error ? error.message : String(error)
        });
      }
    }
  } catch (error) {
    findings.push({
      scope: 'robot-routes',
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }

  return findings;
}

async function validateRobotMissionCatalog() {
  const findings = [];
  const catalogUrl = activeRobotMissionCatalogInput;
  const validatedMissionUrls = new Set();

  try {
    const catalog = await readJsonInput(catalogUrl);
    const entries = Array.isArray(catalog.missions) ? catalog.missions : [];

    findings.push({
      scope: 'robot-mission-catalog',
      status: 'ready',
      message: `${catalog.label ?? 'Robot Mission Catalog'} を読み込みました`
    });

    for (const [index, entry] of entries.entries()) {
      const label = hasNonEmptyString(entry.label)
        ? entry.label.trim()
        : `mission-entry-${index + 1}`;
      const scope = `robot-mission:${label}`;
      const missionUrl = hasNonEmptyString(entry.url) ? entry.url.trim() : '';

      if (!missionUrl) {
        findings.push({
          scope,
          status: 'error',
          message: 'mission URL が空です'
        });
        continue;
      }

      validatedMissionUrls.add(missionUrl);

      try {
        const mission = await readJsonInput(missionUrl);
        findings.push(await validateRobotMission(scope, mission, entry, missionUrl));
      } catch (error) {
        findings.push({
          scope,
          status: 'error',
          message: error instanceof Error ? error.message : String(error)
        });
      }
    }
  } catch (error) {
    findings.push({
      scope: 'robot-mission-catalog',
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }

  return {
    findings,
    validatedMissionUrls
  };
}

async function validateStandaloneRobotMissions(validatedMissionUrls) {
  const findings = [];
  const missionDir = path.join(activePublicRoot, 'robot-missions');

  try {
    const entries = await readdir(missionDir, { withFileTypes: true });
    const missionFiles = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.json') && entry.name !== 'index.json')
      .map((entry) => `/robot-missions/${entry.name}`)
      .filter((missionUrl) => !validatedMissionUrls.has(missionUrl));

    for (const missionUrl of missionFiles) {
      const label = path.basename(missionUrl);
      const scope = `robot-mission-file:${label}`;

      try {
        const mission = await readJsonInput(missionUrl);
        findings.push(await validateRobotMission(scope, mission, null, missionUrl));
      } catch (error) {
        findings.push({
          scope,
          status: 'error',
          message: error instanceof Error ? error.message : String(error)
        });
      }
    }
  } catch (error) {
    findings.push({
      scope: 'robot-missions',
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }

  return findings;
}

async function validateConfiguredZoneMaps() {
  const findings = [];
  const configuredUrls = new Set();

  for (const fragmentId of Object.keys(dreamwalkerConfig.fragments)) {
    const activeConfig = resolveDreamwalkerConfig(fragmentId);
    const zoneUrl = activeConfig.robotics?.semanticZoneMapUrl ?? '';

    if (zoneUrl) {
      configuredUrls.add(zoneUrl);
    }

    findings.push(
      await validateZoneMap(`zone-map:${fragmentId}`, zoneUrl, { fragmentId })
    );
  }

  return {
    findings,
    configuredUrls
  };
}

async function validateStandaloneZoneMaps(configuredUrls) {
  const findings = [];
  const manifestsDir = path.join(activePublicRoot, 'manifests');

  try {
    const entries = await readdir(manifestsDir, { withFileTypes: true });
    const zoneFiles = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith('.zones.json'))
      .map((entry) => `/manifests/${entry.name}`)
      .filter((zoneUrl) => !configuredUrls.has(zoneUrl));

    for (const zoneUrl of zoneFiles) {
      findings.push(
        await validateZoneMap(`zone-map-file:${path.basename(zoneUrl)}`, zoneUrl)
      );
    }
  } catch (error) {
    findings.push({
      scope: 'zone-maps',
      status: 'error',
      message: error instanceof Error ? error.message : String(error)
    });
  }

  return findings;
}

function printFindings(findings) {
  const icons = {
    ready: 'OK',
    warning: 'WARN',
    error: 'ERR'
  };

  findings.forEach((finding) => {
    console.log(`${icons[finding.status] ?? 'INFO'}  ${formatScope(finding.scope)} ${finding.message}`);
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  activePublicRoot = toAbsolutePath(args['public-root'] ?? defaultPublicRoot);
  activeManifestInput = hasNonEmptyString(args['asset-manifest'])
    ? args['asset-manifest'].trim()
    : dreamwalkerConfig.assetManifest.defaultUrl;
  activeStudioBundleCatalogInput = hasNonEmptyString(args['bundle-catalog'])
    ? args['bundle-catalog'].trim()
    : dreamwalkerConfig.studioBundleCatalog.defaultUrl;
  activeRobotRouteCatalogInput = hasNonEmptyString(args['robot-route-catalog'])
    ? args['robot-route-catalog'].trim()
    : dreamwalkerConfig.robotRouteCatalog.defaultUrl;
  activeRobotMissionCatalogInput = hasNonEmptyString(args['robot-mission-catalog'])
    ? args['robot-mission-catalog'].trim()
    : dreamwalkerConfig.robotMissionCatalog.defaultUrl;
  activeMissionOnly = Boolean(args['mission-only']);

  const manifestFindings = activeMissionOnly ? [] : await validateManifest();
  const { findings: catalogFindings, validatedBundleUrls } = activeMissionOnly
    ? { findings: [], validatedBundleUrls: new Set() }
    : await validateCatalog();
  const standaloneBundleFindings = activeMissionOnly
    ? []
    : await validateStandaloneBundles(validatedBundleUrls);
  const { findings: robotRouteCatalogFindings, validatedRouteUrls } =
    await validateRobotRouteCatalog();
  const standaloneRobotRouteFindings = await validateStandaloneRobotRoutes(validatedRouteUrls);
  const { findings: robotMissionCatalogFindings, validatedMissionUrls } =
    await validateRobotMissionCatalog();
  const standaloneRobotMissionFindings = await validateStandaloneRobotMissions(validatedMissionUrls);
  const { findings: zoneMapFindings, configuredUrls } = activeMissionOnly
    ? { findings: [], configuredUrls: new Set() }
    : await validateConfiguredZoneMaps();
  const standaloneZoneMapFindings = await validateStandaloneZoneMaps(configuredUrls);
  const findings = [
    ...manifestFindings,
    ...catalogFindings,
    ...standaloneBundleFindings,
    ...robotRouteCatalogFindings,
    ...standaloneRobotRouteFindings,
    ...robotMissionCatalogFindings,
    ...standaloneRobotMissionFindings,
    ...zoneMapFindings,
    ...standaloneZoneMapFindings
  ];

  printFindings(findings);

  const errorCount = findings.filter((finding) => finding.status === 'error').length;
  const warningCount = findings.filter((finding) => finding.status === 'warning').length;

  console.log('');
  console.log(`Summary: ${errorCount} error(s), ${warningCount} warning(s)`);

  if (errorCount > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
