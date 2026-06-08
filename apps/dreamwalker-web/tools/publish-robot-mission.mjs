import { mkdtemp, readFile, rm, writeFile, mkdir } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolveDreamwalkerConfig } from '../src/app-config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const robotMissionArtifactPackProtocolId = 'dreamwalker-robot-mission-artifact-pack/v1';

function printUsage() {
  console.log(`Dynamic Map Viewer robot mission publish

Usage:
  node ./tools/publish-robot-mission.mjs --route ./public/robot-routes/residency-window-loop.json --fragment residency --route-id residency-patrol --force
  node ./tools/publish-robot-mission.mjs --bundle ./downloads/dreamwalker-live-residency-robot-mission-draft-bundle.json --force
  node ./tools/publish-robot-mission.mjs --bundle ./downloads/dreamwalker-live-residency-robot-mission-draft-bundle.artifact-pack.json --force

Options:
  --bundle <file|url>           Mission Draft Bundle JSON または Artifact Pack JSON. route / zones / mission をまとめて読み込みます。
  --route <file|url>            route JSON. required unless --bundle.
  --fragment <id>               target fragment id. required unless bundle に fragmentId がある場合.
  --zones <file|url>            zone JSON source. optional.
  --tune-zones                  run tune:robot-zones before route publish.
  --route-id <id>               output route id.
  --route-label <text>          output route label.
  --description <text>          route catalog description.
  --accent <hex>                route catalog accent.
  --mission-id <id>             output mission id. default: route id.
  --mission-label <text>        output mission label. default: route label + " Mission"
  --mission-description <text>  output mission description.
  --mission-accent <hex>        output mission accent.
  --camera-preset <id>          startup camera preset id. default: fragment default.
  --robot-camera <id>           startup robot camera id. default: fragment default.
  --stream-scene <id>           startup stream scene id. optional.
  --startup-mode <id>           startup mode. default: robot.
  --mission-path <file>         custom output mission file path.
  --mission-catalog-path <file> custom mission catalog path.
  --preflight-output <file>     write publish preflight summary text.
  --report-output <file>        write machine-readable publish report JSON.
  --route-path <file>           custom output route file path.
  --zone-path <file>            custom output zone file path.
  --public-root <dir>           custom public root.
  --catalog-path <file>         custom robot route catalog path.
  --asset-manifest <file>       custom asset manifest path.
  --corridor-padding <n>        pass through to tune:robot-zones.
  --bounds-padding <n>          pass through to tune:robot-zones.
  --cost <n>                    pass through to tune:robot-zones.
  --label-prefix <text>         pass through to tune:robot-zones.
  --include-hazard-review       pass through to tune:robot-zones.
  --hazard-cost <n>             pass through to tune:robot-zones.
  --hazard-label-prefix <text>  pass through to tune:robot-zones.
  --include-bounds-review       pass through to tune:robot-zones.
  --bounds-cost <n>             pass through to tune:robot-zones.
  --bounds-label-prefix <text>  pass through to tune:robot-zones.
  --merge-bounds                pass through to tune:robot-zones.
  --frame-id <id>               override frame id for staged zones.
  --resolution <value>          override resolution for staged zones.
  --default-cost <value>        override default cost for staged zones.
  --dry-run                     do not write final files.
  --force                       overwrite existing final files.
  --validate                    run validate:studio after publish when using default public root.
  --keep-temp                   keep intermediate temp files.
  --help                        show this message.
`);
}

function parseArgs(argv) {
  const args = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token.startsWith('--')) {
      continue;
    }

    const key = token.slice(2);
    if (
      key === 'help' ||
      key === 'dry-run' ||
      key === 'force' ||
      key === 'validate' ||
      key === 'keep-temp' ||
      key === 'tune-zones' ||
      key === 'merge-bounds' ||
      key === 'include-hazard-review' ||
      key === 'include-bounds-review'
    ) {
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

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function toAbsolutePath(inputPath, baseDir = process.cwd()) {
  if (!inputPath) {
    return '';
  }

  return path.isAbsolute(inputPath)
    ? inputPath
    : path.resolve(baseDir, inputPath);
}

function isRemoteUrl(value) {
  return typeof value === 'string' && /^https?:\/\//i.test(value.trim());
}

function toPosixPath(filePath) {
  return filePath.split(path.sep).join('/');
}

function buildPublicUrl(publicRoot, filePath) {
  const relativePath = path.relative(publicRoot, filePath);

  if (relativePath.startsWith('..')) {
    throw new Error(`public root の外に出ています: ${filePath}`);
  }

  return `/${toPosixPath(relativePath)}`;
}

function pushFlag(args, key, enabled) {
  if (enabled) {
    args.push(`--${key}`);
  }
}

function pushOption(args, key, value) {
  if (hasNonEmptyString(value)) {
    args.push(`--${key}`, String(value).trim());
  }
}

async function readJsonInput(input) {
  if (isRemoteUrl(input)) {
    const response = await fetch(input);

    if (!response.ok) {
      throw new Error(`download に失敗しました: ${response.status} ${response.statusText}`);
    }

    return JSON.parse(await response.text());
  }

  const raw = await readFile(toAbsolutePath(input), 'utf8');
  return JSON.parse(raw);
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function normalizeBundleInput(bundleLike) {
  if (
    isRecord(bundleLike) &&
    hasNonEmptyString(bundleLike.protocol) &&
    bundleLike.protocol.trim() === robotMissionArtifactPackProtocolId
  ) {
    const files = Array.isArray(bundleLike.files) ? bundleLike.files : [];
    const draftBundleEntry = files.find(
      (entry) =>
        isRecord(entry) &&
        hasNonEmptyString(entry.kind) &&
        entry.kind.trim() === 'draft-bundle'
    );

    if (!draftBundleEntry) {
      throw new Error('--bundle の artifact pack には draft-bundle entry が必要です。');
    }

    const draftBundleContent =
      typeof draftBundleEntry.content === 'string'
        ? JSON.parse(draftBundleEntry.content)
        : draftBundleEntry.content;

    if (!isRecord(draftBundleContent)) {
      throw new Error('--bundle の artifact pack draft-bundle content が不正です。');
    }

    return {
      bundle: draftBundleContent,
      artifactPack: {
        label: hasNonEmptyString(bundleLike.label) ? bundleLike.label.trim() : '',
        fileCount: files.length
      }
    };
  }

  return {
    bundle: bundleLike,
    artifactPack: null
  };
}

function sanitizeId(value, fallback) {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return normalized || fallback;
}

function extractRobotRouteIdFromUrl(routeUrlLike) {
  const normalizedRouteUrl = hasNonEmptyString(routeUrlLike)
    ? routeUrlLike.trim()
    : '';

  if (!normalizedRouteUrl) {
    return '';
  }

  const routeMatch = normalizedRouteUrl.match(/\/robot-routes\/([^/?#]+)\.json(?:[?#].*)?$/i);
  if (!routeMatch?.[1]) {
    return '';
  }

  return sanitizeId(routeMatch[1], '');
}

function normalizeStartupMode(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  return ['explore', 'live', 'photo', 'robot'].includes(normalized)
    ? normalized
    : '';
}

function inferStemFromSource(source) {
  if (!hasNonEmptyString(source)) {
    return '';
  }

  if (isRemoteUrl(source)) {
    const url = new URL(source);
    return path.basename(url.pathname, path.extname(url.pathname));
  }

  const normalizedSource = String(source).trim();
  return path.basename(normalizedSource, path.extname(normalizedSource));
}

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function writeText(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, value, 'utf8');
}

async function materializeJsonInput(tempRoot, fileName, value) {
  const filePath = path.join(tempRoot, fileName);
  await writeJson(filePath, value);
  return filePath;
}

function upsertCatalogEntry(collectionKey, catalog, nextEntry, defaults) {
  const nextItems = Array.isArray(catalog[collectionKey]) ? [...catalog[collectionKey]] : [];
  const existingIndex = nextItems.findIndex(
    (entry) => entry.id === nextEntry.id || entry.url === nextEntry.url
  );

  if (existingIndex >= 0) {
    nextItems[existingIndex] = {
      ...nextItems[existingIndex],
      ...nextEntry
    };
  } else {
    nextItems.push(nextEntry);
  }

  return {
    ...catalog,
    version: catalog.version ?? 1,
    label: catalog.label ?? defaults.label,
    note: catalog.note ?? defaults.note,
    [collectionKey]: nextItems
  };
}

function runNodeScript(scriptName, args) {
  const scriptPath = path.join(__dirname, scriptName);
  const result = spawnSync(process.execPath, [scriptPath, ...args], {
    stdio: 'inherit'
  });

  if (typeof result.status === 'number' && result.status !== 0) {
    throw new Error(`${scriptName} が失敗しました (exit ${result.status})`);
  }

  if (result.error) {
    throw result.error;
  }
}

function buildMissionPreflightSummary({
  healthLabel,
  healthDetail,
  missionPayload,
  routePayload,
  routeId,
  routeFileName,
  cameraPresetLabel,
  robotCameraLabel,
  streamSceneLabel
}) {
  return [
    `status: ${healthLabel || 'unknown'}`,
    `detail: ${healthDetail || 'none'}`,
    `missionId: ${missionPayload.id || 'none'}`,
    `missionLabel: ${missionPayload.label || 'none'}`,
    `missionDescription: ${missionPayload.description || 'none'}`,
    `fragmentId: ${missionPayload.fragmentId || 'none'}`,
    `fragmentLabel: ${missionPayload.fragmentLabel || 'none'}`,
    `routeId: ${routeId || 'none'}`,
    `routeFile: ${routeFileName || 'none'}`,
    `routeLabel: ${routePayload.label || 'none'}`,
    `routeDescription: ${routePayload.description || 'none'}`,
    `routeAccent: ${routePayload.accent || 'none'}`,
    `worldAsset: ${missionPayload.world?.assetLabel || 'none'}`,
    `worldFrame: ${missionPayload.world?.frameId || 'none'}`,
    `zoneMapUrl: ${missionPayload.zoneMapUrl || 'none'}`,
    `startupMode: ${missionPayload.startupMode || 'none'}`,
    `cameraPresetId: ${missionPayload.cameraPresetId || 'none'}`,
    `cameraPresetLabel: ${cameraPresetLabel || 'none'}`,
    `robotCameraId: ${missionPayload.robotCameraId || 'none'}`,
    `robotCameraLabel: ${robotCameraLabel || 'none'}`,
    `streamSceneId: ${missionPayload.streamSceneId || 'none'}`,
    `streamSceneLabel: ${streamSceneLabel || 'none'}`,
    `launchUrl: ${missionPayload.launchUrl || 'none'}`
  ].join('\n');
}

function buildMissionPublishReport({
  dryRun,
  fragmentId,
  publicRoot,
  routeSource,
  zoneSource,
  routePath,
  routeUrl,
  routeCatalogPath,
  missionPath,
  missionUrl,
  missionCatalogPath,
  missionCatalogUrl,
  zonePath,
  zoneMapUrl,
  missionPayload,
  routePayload,
  routeId,
  routeFileName,
  cameraPresetLabel,
  robotCameraLabel,
  streamSceneLabel,
  preflightHealthLabel,
  preflightHealthDetail,
  preflightSummary,
  preflightOutputPath,
  reportOutputPath,
  validateRequested
}) {
  return {
    version: 1,
    protocol: 'dreamwalker-robot-mission-publish-report/v1',
    dryRun: Boolean(dryRun),
    fragmentId,
    publicRoot,
    mission: {
      id: missionPayload.id,
      label: missionPayload.label,
      description: missionPayload.description,
      accent: missionPayload.accent,
      url: missionUrl,
      path: missionPath,
      launchUrl: missionPayload.launchUrl,
      catalogPath: missionCatalogPath,
      catalogUrl: missionCatalogUrl
    },
    route: {
      id: routeId,
      fileName: routeFileName,
      label: routePayload.label || '',
      description: routePayload.description || '',
      accent: routePayload.accent || '',
      url: routeUrl,
      path: routePath,
      catalogPath: routeCatalogPath,
      source: routeSource
    },
    zones: {
      url: zoneMapUrl || '',
      path: zonePath || '',
      source: zoneSource || ''
    },
    world: {
      assetLabel: missionPayload.world?.assetLabel || '',
      frameId: missionPayload.world?.frameId || ''
    },
    startup: {
      mode: missionPayload.startupMode || '',
      cameraPresetId: missionPayload.cameraPresetId || '',
      cameraPresetLabel,
      robotCameraId: missionPayload.robotCameraId || '',
      robotCameraLabel,
      streamSceneId: missionPayload.streamSceneId || '',
      streamSceneLabel
    },
    preflight: {
      label: preflightHealthLabel,
      detail: preflightHealthDetail,
      summary: preflightSummary,
      outputPath: preflightOutputPath || ''
    },
    outputs: {
      reportOutputPath: reportOutputPath || ''
    },
    validation: {
      requested: Boolean(validateRequested)
    }
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  if (!hasNonEmptyString(args.route) && !hasNonEmptyString(args.bundle)) {
    throw new Error('--route か --bundle のどちらかは必須です。');
  }

  const publicRoot = toAbsolutePath(args['public-root'] ?? path.join(__dirname, '..', 'public'));
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'dreamwalker-publish-mission-'));

  try {
    const parsedBundleInput = hasNonEmptyString(args.bundle)
      ? normalizeBundleInput(await readJsonInput(args.bundle))
      : { bundle: null, artifactPack: null };
    const sourceBundle = parsedBundleInput.bundle;
    const bundleMission = isRecord(sourceBundle?.mission) ? sourceBundle.mission : null;
    const bundleRoute = isRecord(sourceBundle?.route) ? sourceBundle.route : null;
    const bundleZones = isRecord(sourceBundle?.zones) ? sourceBundle.zones : null;

    if (hasNonEmptyString(args.bundle) && !bundleRoute) {
      throw new Error('--bundle には route object が必要です。');
    }

    const resolvedFragmentId = String(
      args.fragment ??
        bundleMission?.fragmentId ??
        sourceBundle?.fragmentId ??
        bundleRoute?.fragmentId ??
        ''
    ).trim();

    if (!hasNonEmptyString(resolvedFragmentId)) {
      throw new Error('--fragment は必須です。bundle に fragmentId があれば省略できます。');
    }

    const routeSource = hasNonEmptyString(args.route)
      ? args.route
      : await materializeJsonInput(
          tempRoot,
          'bundle-route.json',
          bundleRoute
        );
    const zoneSource = hasNonEmptyString(args.zones)
      ? args.zones
      : bundleZones
        ? await materializeJsonInput(
            tempRoot,
            `bundle-zones-${resolvedFragmentId}.json`,
            bundleZones
          )
        : '';
    const sourceRoute = await readJsonInput(routeSource);
    const fragmentConfig = resolveDreamwalkerConfig(resolvedFragmentId);
    const sourceRouteWorld = isRecord(sourceRoute?.world) ? sourceRoute.world : {};
    const sourceRouteLabel = hasNonEmptyString(sourceRoute?.label) ? sourceRoute.label.trim() : '';
    const sourceMissionId = hasNonEmptyString(bundleMission?.id) ? bundleMission.id.trim() : '';
    const sourceMissionLabel = hasNonEmptyString(bundleMission?.label)
      ? bundleMission.label.trim()
      : '';

    if (parsedBundleInput.artifactPack) {
      console.log(
        `Artifact Pack: ${parsedBundleInput.artifactPack.label || 'unnamed'} / ${parsedBundleInput.artifactPack.fileCount} file(s)`
      );
    }
    const resolvedRouteId = sanitizeId(
      args['route-id'] ??
        extractRobotRouteIdFromUrl(bundleMission?.routeUrl) ??
        sourceMissionId ??
        sourceRouteLabel ??
        sourceMissionLabel ??
        inferStemFromSource(hasNonEmptyString(args.route) ? args.route : args.bundle),
      `${resolvedFragmentId}-route`
    );
    const resolvedRouteLabel =
      hasNonEmptyString(args['route-label'])
        ? args['route-label'].trim()
        : sourceRouteLabel || sourceMissionLabel || `${resolvedFragmentId} Route Preset`;
    const resolvedRouteDescription = hasNonEmptyString(args.description)
      ? args.description.trim()
      : hasNonEmptyString(sourceRoute?.description)
        ? sourceRoute.description.trim()
      : hasNonEmptyString(bundleMission?.description)
        ? bundleMission.description.trim()
        : '';
    const resolvedRouteAccent = hasNonEmptyString(args.accent)
      ? args.accent.trim()
      : hasNonEmptyString(sourceRoute?.accent)
        ? sourceRoute.accent.trim()
      : hasNonEmptyString(bundleMission?.accent)
        ? bundleMission.accent.trim()
        : '';
    const resolvedMissionId = sanitizeId(
      args['mission-id'] ?? sourceMissionId ?? resolvedRouteId,
      `${resolvedRouteId}-mission`
    );
    const resolvedMissionLabel = hasNonEmptyString(args['mission-label'])
      ? args['mission-label'].trim()
      : sourceMissionLabel
        ? sourceMissionLabel
      : resolvedRouteLabel.includes('Mission')
        ? resolvedRouteLabel
        : `${resolvedRouteLabel} Mission`;
    const resolvedMissionDescription = hasNonEmptyString(args['mission-description'])
      ? args['mission-description'].trim()
      : hasNonEmptyString(bundleMission?.description)
        ? bundleMission.description.trim()
      : hasNonEmptyString(args.description)
        ? args.description.trim()
        : `${resolvedRouteLabel} mission bundle`;
    const resolvedMissionAccent = hasNonEmptyString(args['mission-accent'])
      ? args['mission-accent'].trim()
      : hasNonEmptyString(bundleMission?.accent)
        ? bundleMission.accent.trim()
      : hasNonEmptyString(args.accent)
        ? args.accent.trim()
        : '#85e3e1';
    const resolvedWorldAssetLabel = hasNonEmptyString(bundleMission?.world?.assetLabel)
      ? bundleMission.world.assetLabel.trim()
      : sourceRouteWorld.assetLabel ?? '';
    const resolvedWorldFrameId = hasNonEmptyString(bundleMission?.world?.frameId)
      ? bundleMission.world.frameId.trim()
      : sourceRouteWorld.frameId ?? sourceRoute.frameId ?? '';
    const resolvedStreamSceneId = hasNonEmptyString(args['stream-scene'])
      ? args['stream-scene'].trim()
      : hasNonEmptyString(bundleMission?.streamSceneId)
        ? bundleMission.streamSceneId.trim()
      : fragmentConfig.streamScenes?.[0]?.id ?? '';
    const startupStreamScene = resolvedStreamSceneId
      ? fragmentConfig.streamScenes?.find((scene) => scene.id === resolvedStreamSceneId) ?? null
      : null;
    const resolvedCameraPresetId = hasNonEmptyString(args['camera-preset'])
      ? args['camera-preset'].trim()
      : hasNonEmptyString(bundleMission?.cameraPresetId)
        ? bundleMission.cameraPresetId.trim()
      : startupStreamScene?.presetId ?? fragmentConfig.homePresetId ?? '';
    const resolvedRobotCameraId = hasNonEmptyString(args['robot-camera'])
      ? args['robot-camera'].trim()
      : hasNonEmptyString(bundleMission?.robotCameraId)
        ? bundleMission.robotCameraId.trim()
      : fragmentConfig.robotics?.defaultCameraId ??
        fragmentConfig.robotics?.cameras?.[0]?.id ??
        '';
    const resolvedStartupMode = normalizeStartupMode(
      args['startup-mode'] ?? bundleMission?.startupMode ?? 'robot'
    );

    if (
      resolvedCameraPresetId &&
      !fragmentConfig.cameraPresets.some((preset) => preset.id === resolvedCameraPresetId)
    ) {
      throw new Error(`camera preset が不正です: ${resolvedCameraPresetId}`);
    }

    if (
      resolvedRobotCameraId &&
      !fragmentConfig.robotics.cameras.some((camera) => camera.id === resolvedRobotCameraId)
    ) {
      throw new Error(`robot camera が不正です: ${resolvedRobotCameraId}`);
    }

    if (
      resolvedStreamSceneId &&
      !fragmentConfig.streamScenes.some((scene) => scene.id === resolvedStreamSceneId)
    ) {
      throw new Error(`stream scene が不正です: ${resolvedStreamSceneId}`);
    }

    if (!resolvedStartupMode) {
      throw new Error(`startup mode が不正です: ${args['startup-mode'] ?? 'robot'}`);
    }

    const routePath = hasNonEmptyString(args['route-path'])
      ? toAbsolutePath(args['route-path'])
      : path.join(publicRoot, 'robot-routes', `${resolvedRouteId}.json`);
    const routeUrl = buildPublicUrl(publicRoot, routePath);
    const routeCatalogPath = hasNonEmptyString(args['catalog-path'])
      ? toAbsolutePath(args['catalog-path'])
      : path.join(publicRoot, 'robot-routes', 'index.json');
    const missionPath = hasNonEmptyString(args['mission-path'])
      ? toAbsolutePath(args['mission-path'])
      : path.join(publicRoot, 'robot-missions', `${resolvedMissionId}.mission.json`);
    const missionUrl = buildPublicUrl(publicRoot, missionPath);
    const missionCatalogPath = hasNonEmptyString(args['mission-catalog-path'])
      ? toAbsolutePath(args['mission-catalog-path'])
      : path.join(publicRoot, 'robot-missions', 'index.json');
    const missionCatalogUrl = buildPublicUrl(publicRoot, missionCatalogPath);
    let zoneMapUrl = hasNonEmptyString(bundleMission?.zoneMapUrl)
      ? bundleMission.zoneMapUrl.trim()
      : '';

    if (hasNonEmptyString(zoneSource)) {
      const zonePath =
        toAbsolutePath(
          args['zone-path'] ??
            path.join(publicRoot, 'manifests', `robotics-${resolvedFragmentId}.zones.json`)
        );
      zoneMapUrl = buildPublicUrl(publicRoot, zonePath);

      if (args['tune-zones']) {
        const tuneArgs = [];
        pushOption(tuneArgs, 'route', routeSource);
        pushOption(tuneArgs, 'zones', zoneSource);
        pushOption(tuneArgs, 'fragment', resolvedFragmentId);
        pushOption(tuneArgs, 'zone-path', zonePath);
        pushOption(tuneArgs, 'public-root', publicRoot);
        pushOption(tuneArgs, 'corridor-padding', args['corridor-padding']);
        pushOption(tuneArgs, 'bounds-padding', args['bounds-padding']);
        pushOption(tuneArgs, 'cost', args.cost);
        pushOption(tuneArgs, 'label-prefix', args['label-prefix']);
        pushFlag(tuneArgs, 'include-hazard-review', Boolean(args['include-hazard-review']));
        pushOption(tuneArgs, 'hazard-cost', args['hazard-cost']);
        pushOption(tuneArgs, 'hazard-label-prefix', args['hazard-label-prefix']);
        pushFlag(tuneArgs, 'include-bounds-review', Boolean(args['include-bounds-review']));
        pushOption(tuneArgs, 'bounds-cost', args['bounds-cost']);
        pushOption(tuneArgs, 'bounds-label-prefix', args['bounds-label-prefix']);
        pushFlag(tuneArgs, 'merge-bounds', Boolean(args['merge-bounds']));
        pushOption(tuneArgs, 'frame-id', args['frame-id']);
        pushOption(tuneArgs, 'resolution', args.resolution);
        pushOption(tuneArgs, 'default-cost', args['default-cost']);
        pushFlag(tuneArgs, 'dry-run', Boolean(args['dry-run']));
        pushFlag(tuneArgs, 'force', Boolean(args.force));
        pushFlag(tuneArgs, 'keep-temp', Boolean(args['keep-temp']));

        console.log('Step 1/2: tune-robot-zones');
        runNodeScript('tune-robot-zones.mjs', tuneArgs);
      } else {
        const stageZoneArgs = [];
        pushOption(stageZoneArgs, 'source', zoneSource);
        pushOption(stageZoneArgs, 'fragment', resolvedFragmentId);
        pushOption(stageZoneArgs, 'zone-path', zonePath);
        pushOption(stageZoneArgs, 'public-root', publicRoot);
        pushOption(stageZoneArgs, 'frame-id', args['frame-id']);
        pushOption(stageZoneArgs, 'resolution', args.resolution);
        pushOption(stageZoneArgs, 'default-cost', args['default-cost']);
        pushFlag(stageZoneArgs, 'dry-run', Boolean(args['dry-run']));
        pushFlag(stageZoneArgs, 'force', Boolean(args.force));

        console.log('Step 1/2: stage-robot-zones');
        runNodeScript('stage-robot-zones.mjs', stageZoneArgs);
      }
    }

    const resolvedRouteCatalogPath = routeCatalogPath;
    const stageRouteArgs = [];
    pushOption(stageRouteArgs, 'source', routeSource);
    pushOption(stageRouteArgs, 'fragment', resolvedFragmentId);
    pushOption(stageRouteArgs, 'route-id', resolvedRouteId);
    pushOption(stageRouteArgs, 'route-label', resolvedRouteLabel);
    pushOption(stageRouteArgs, 'description', resolvedRouteDescription);
    pushOption(stageRouteArgs, 'accent', resolvedRouteAccent);
    pushOption(stageRouteArgs, 'asset-manifest', args['asset-manifest']);
    pushOption(stageRouteArgs, 'catalog-path', resolvedRouteCatalogPath);
    pushOption(stageRouteArgs, 'route-path', routePath);
    pushOption(stageRouteArgs, 'public-root', publicRoot);
    pushOption(stageRouteArgs, 'zone-map-url', zoneMapUrl);
    pushFlag(stageRouteArgs, 'dry-run', Boolean(args['dry-run']));
    pushFlag(stageRouteArgs, 'force', Boolean(args.force));

    console.log(hasNonEmptyString(zoneSource) ? 'Step 2/2: stage-robot-route' : 'Step 1/1: stage-robot-route');
    runNodeScript('stage-robot-route.mjs', stageRouteArgs);

    const stagedRoute = !args['dry-run']
      ? JSON.parse(await readFile(routePath, 'utf8'))
      : sourceRoute;
    const missionPayload = {
      version: 1,
      protocol: 'dreamwalker-robot-mission/v1',
      id: resolvedMissionId,
      label: resolvedMissionLabel,
      description: resolvedMissionDescription,
      fragmentId: resolvedFragmentId,
      fragmentLabel: stagedRoute.fragmentLabel || resolvedFragmentId,
      accent: resolvedMissionAccent,
      routeUrl,
      zoneMapUrl: zoneMapUrl || stagedRoute.world?.zoneMapUrl || '',
      launchUrl: `/?robotMission=${encodeURIComponent(missionUrl)}`,
      cameraPresetId: resolvedCameraPresetId,
      robotCameraId: resolvedRobotCameraId,
      streamSceneId: resolvedStreamSceneId,
      startupMode: resolvedStartupMode,
      world: {
        assetLabel: resolvedWorldAssetLabel,
        frameId: resolvedWorldFrameId
      }
    };
    const routeFileName = path.basename(routePath);
    const cameraPresetLabel =
      fragmentConfig.cameraPresets.find((preset) => preset.id === resolvedCameraPresetId)?.label ??
      resolvedCameraPresetId;
    const robotCameraLabel =
      fragmentConfig.robotics.cameras.find((camera) => camera.id === resolvedRobotCameraId)?.label ??
      resolvedRobotCameraId;
    const streamSceneLabel =
      fragmentConfig.streamScenes.find((scene) => scene.id === resolvedStreamSceneId)?.title ??
      fragmentConfig.streamScenes.find((scene) => scene.id === resolvedStreamSceneId)?.label ??
      resolvedStreamSceneId;
    const preflightWarnings = [];

    if (
      hasNonEmptyString(missionPayload.fragmentId) &&
      hasNonEmptyString(stagedRoute.fragmentId) &&
      missionPayload.fragmentId !== stagedRoute.fragmentId
    ) {
      preflightWarnings.push(
        `mission fragment=${missionPayload.fragmentId} / route fragment=${stagedRoute.fragmentId}`
      );
    }

    if (
      hasNonEmptyString(missionPayload.zoneMapUrl) &&
      hasNonEmptyString(stagedRoute.world?.zoneMapUrl) &&
      missionPayload.zoneMapUrl !== stagedRoute.world.zoneMapUrl
    ) {
      preflightWarnings.push(
        `mission zone=${missionPayload.zoneMapUrl} / route zone=${stagedRoute.world.zoneMapUrl}`
      );
    }

    if (
      hasNonEmptyString(missionPayload.world?.frameId) &&
      hasNonEmptyString(stagedRoute.frameId) &&
      missionPayload.world.frameId !== stagedRoute.frameId
    ) {
      preflightWarnings.push(
        `mission frame=${missionPayload.world.frameId} / route frame=${stagedRoute.frameId}`
      );
    }

    if (
      hasNonEmptyString(missionPayload.world?.assetLabel) &&
      hasNonEmptyString(stagedRoute.world?.assetLabel) &&
      missionPayload.world.assetLabel !== stagedRoute.world.assetLabel
    ) {
      preflightWarnings.push(
        `mission world=${missionPayload.world.assetLabel} / route world=${stagedRoute.world.assetLabel}`
      );
    }

    const preflightHealthLabel = preflightWarnings.length ? 'Mission Warning' : 'Mission Ready';
    const preflightHealthDetail = preflightWarnings.length
      ? preflightWarnings.join(' ; ')
      : `${missionPayload.world.assetLabel || stagedRoute.world?.assetLabel || missionPayload.fragmentLabel} / frame ${missionPayload.world.frameId || stagedRoute.frameId}`;
    const preflightSummary = buildMissionPreflightSummary({
      healthLabel: preflightHealthLabel,
      healthDetail: preflightHealthDetail,
      missionPayload,
      routePayload: stagedRoute,
      routeId: resolvedRouteId,
      routeFileName,
      cameraPresetLabel,
      robotCameraLabel,
      streamSceneLabel
    });

    console.log('Mission Preflight');
    console.log(preflightSummary);

    const preflightOutputPath = hasNonEmptyString(args['preflight-output'])
      ? toAbsolutePath(args['preflight-output'])
      : '';
    if (preflightOutputPath) {
      if (!args['dry-run']) {
        await writeText(preflightOutputPath, `${preflightSummary}\n`);
        console.log(`- preflight: ${preflightOutputPath}`);
      } else {
        console.log(`- preflight output (dry-run): ${preflightOutputPath}`);
      }
    }

    const reportOutputPath = hasNonEmptyString(args['report-output'])
      ? toAbsolutePath(args['report-output'])
      : '';
    const zonePath = hasNonEmptyString(zoneSource)
      ? toAbsolutePath(
          args['zone-path'] ??
            path.join(publicRoot, 'manifests', `robotics-${resolvedFragmentId}.zones.json`)
        )
      : '';
    const publishReport = buildMissionPublishReport({
      dryRun: args['dry-run'],
      fragmentId: resolvedFragmentId,
      publicRoot,
      routeSource,
      zoneSource,
      routePath,
      routeUrl,
      routeCatalogPath: resolvedRouteCatalogPath,
      missionPath,
      missionUrl,
      missionCatalogPath,
      missionCatalogUrl,
      zonePath,
      zoneMapUrl: missionPayload.zoneMapUrl,
      missionPayload,
      routePayload: stagedRoute,
      routeId: resolvedRouteId,
      routeFileName,
      cameraPresetLabel,
      robotCameraLabel,
      streamSceneLabel,
      preflightHealthLabel,
      preflightHealthDetail,
      preflightSummary,
      preflightOutputPath,
      reportOutputPath,
      validateRequested: args.validate
    });

    if (reportOutputPath) {
      if (!args['dry-run']) {
        await writeJson(reportOutputPath, publishReport);
        console.log(`- report: ${reportOutputPath}`);
      } else {
        console.log(`- report output (dry-run): ${reportOutputPath}`);
      }
    }

    if (!args['dry-run']) {
      const missionCatalog = JSON.parse(
        await readFile(missionCatalogPath, 'utf8').catch(() =>
          JSON.stringify({
            version: 1,
            label: 'Dynamic Map Viewer Public Robot Mission Catalog',
            note: 'publish-robot-mission.mjs により生成される public robot mission catalog。',
            missions: []
          })
        )
      );
      const nextMissionCatalog = upsertCatalogEntry('missions', missionCatalog, {
        id: resolvedMissionId,
        label: resolvedMissionLabel,
        url: missionUrl,
        description: resolvedMissionDescription,
        fragmentId: resolvedFragmentId,
        accent: resolvedMissionAccent
      }, {
        label: 'Dynamic Map Viewer Public Robot Mission Catalog',
        note: 'publish-robot-mission.mjs により生成される public robot mission catalog。'
      });

      await writeJson(missionPath, missionPayload);
      await writeJson(missionCatalogPath, nextMissionCatalog);
      console.log(`- mission: ${missionUrl}`);
      console.log(`- mission catalog: ${missionCatalogUrl}`);
    }

    if (args.validate && !args['dry-run']) {
      const validateArgs = [];
      pushOption(validateArgs, 'public-root', publicRoot);
      pushOption(validateArgs, 'robot-route-catalog', resolvedRouteCatalogPath);
      pushOption(validateArgs, 'robot-mission-catalog', missionCatalogPath);

      if (hasNonEmptyString(args['asset-manifest'])) {
        pushOption(validateArgs, 'asset-manifest', args['asset-manifest']);
      }

      if (hasNonEmptyString(args['bundle-catalog'])) {
        pushOption(validateArgs, 'bundle-catalog', args['bundle-catalog']);
      }

      if (hasNonEmptyString(args['public-root']) || hasNonEmptyString(args['catalog-path']) || hasNonEmptyString(args['asset-manifest']) || hasNonEmptyString(args.zones) || hasNonEmptyString(args['mission-catalog-path'])) {
        pushFlag(validateArgs, 'mission-only', true);
      }

      console.log(hasNonEmptyString(zoneSource) ? 'Step 3/3: validate-studio' : 'Step 2/2: validate-studio');
      runNodeScript('validate-studio-assets.mjs', validateArgs);
    }
  } finally {
    if (!args['keep-temp']) {
      await rm(tempRoot, { recursive: true, force: true });
    }
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`ERR: ${message}`);
  process.exitCode = 1;
});
