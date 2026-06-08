import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  dreamwalkerConfig,
  resolveDreamwalkerConfig,
  resolveWorldAssetBundle
} from '../src/app-config.js';
import { summarizeRouteZoneCoverage } from '../src/robot-route-analysis.js';
import { buildSemanticZoneMap } from '../src/semantic-zones.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const defaultPublicRoot = path.join(appRoot, 'public');
const defaultAssetManifestPath = path.join(defaultPublicRoot, 'manifests', 'dreamwalker-live.assets.json');
const defaultCatalogPath = path.join(defaultPublicRoot, 'robot-routes', 'index.json');
const routeProtocolId = 'dreamwalker-robot-route/v1';

function printUsage() {
  console.log(`Dynamic Map Viewer robot route staging

Usage:
  node ./tools/stage-robot-route.mjs --source ./public/robot-routes/residency-window-loop.json
  node ./tools/stage-robot-route.mjs --source https://example.com/route.json --fragment residency --route-id residency-patrol

Options:
  --source <file|url>      source route JSON. required.
  --fragment <id>          target fragment id. defaults to source route fragmentId.
  --route-id <id>          output route id. defaults to source label / file stem.
  --route-label <text>     output route label.
  --description <text>     catalog description.
  --accent <hex>           catalog accent color.
  --frame-id <id>          override route/world frame id.
  --asset-label <text>     override world.assetLabel.
  --splat-url <url>        override world.splatUrl.
  --collider-url <url>     override world.colliderMeshUrl.
  --zone-map-url <url>     override world.zoneMapUrl.
  --asset-manifest <file>  asset manifest path used to enrich world metadata.
  --catalog-path <file>    output robot route catalog path.
  --route-path <file>      custom output route file path.
  --public-root <dir>      custom public root. default: apps/dreamwalker-web/public
  --dry-run                print planned outputs without writing files.
  --force                  overwrite existing staged route file.
  --help                   show this message.
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
    if (key === 'dry-run' || key === 'force' || key === 'help') {
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

function toAbsolutePath(inputPath, baseDir = process.cwd()) {
  if (!inputPath) {
    return '';
  }

  return path.isAbsolute(inputPath)
    ? inputPath
    : path.resolve(baseDir, inputPath);
}

function sanitizeId(value, fallback) {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return normalized || fallback;
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

async function pathExists(filePath) {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonOrFallback(filePath, fallbackValue) {
  if (!(await pathExists(filePath))) {
    return JSON.parse(JSON.stringify(fallbackValue));
  }

  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function firstNonEmptyString(...values) {
  return values.find((value) => hasNonEmptyString(value))?.trim() ?? '';
}

function normalizePosition(positionLike) {
  if (!Array.isArray(positionLike) || positionLike.length < 3) {
    return null;
  }

  const position = positionLike
    .slice(0, 3)
    .map((value) => (typeof value === 'number' ? value : Number(value)));
  return position.every((value) => Number.isFinite(value)) ? position : null;
}

function normalizePose(poseLike) {
  if (!poseLike || typeof poseLike !== 'object') {
    return null;
  }

  const position = normalizePosition(poseLike.position);
  const yawDegrees = Number(poseLike.yawDegrees);

  if (!position || !Number.isFinite(yawDegrees)) {
    return null;
  }

  return {
    position,
    yawDegrees
  };
}

function normalizeWaypoint(waypointLike) {
  if (!waypointLike) {
    return null;
  }

  const position = normalizePosition(waypointLike.position ?? waypointLike);
  return position ? { position } : null;
}

function normalizeWorldContext(worldLike, fallbackLike = {}) {
  const world = worldLike && typeof worldLike === 'object' ? worldLike : {};
  const fallback =
    fallbackLike && typeof fallbackLike === 'object' ? fallbackLike : {};

  return {
    fragmentId: firstNonEmptyString(world.fragmentId, fallback.fragmentId),
    fragmentLabel: firstNonEmptyString(world.fragmentLabel, fallback.fragmentLabel),
    assetLabel: firstNonEmptyString(world.assetLabel, fallback.assetLabel),
    manifestLabel: firstNonEmptyString(world.manifestLabel, fallback.manifestLabel),
    splatUrl: firstNonEmptyString(world.splatUrl, fallback.splatUrl),
    colliderMeshUrl: firstNonEmptyString(world.colliderMeshUrl, fallback.colliderMeshUrl),
    frameId: firstNonEmptyString(world.frameId, fallback.frameId, 'dreamwalker_map'),
    zoneMapUrl: firstNonEmptyString(world.zoneMapUrl, fallback.zoneMapUrl),
    usesDemoFallback: Boolean(
      Object.prototype.hasOwnProperty.call(world, 'usesDemoFallback')
        ? world.usesDemoFallback
        : fallback.usesDemoFallback
    )
  };
}

function normalizeRoutePayload(routeLike) {
  const route = routeLike && typeof routeLike === 'object' ? routeLike : {};
  const pose = normalizePose(route.pose);
  const points = Array.isArray(route.route)
    ? route.route.map((position) => normalizePosition(position)).filter(Boolean)
    : [];
  const waypoint = normalizeWaypoint(route.waypoint);

  if (!pose && points.length === 0) {
    throw new Error('route JSON に pose か route が必要です。');
  }

  const normalizedPose = pose ?? {
    position: [...points[points.length - 1]],
    yawDegrees: 0
  };
  const normalizedRoute =
    points.length > 0 ? points.map((position) => [...position]) : [[...normalizedPose.position]];
  const world = normalizeWorldContext(route.world, {
    fragmentId: route.fragmentId,
    fragmentLabel: route.fragmentLabel,
    frameId: route.frameId
  });

  return {
    version: Number(route.version ?? 1) || 1,
    protocol: firstNonEmptyString(route.protocol, routeProtocolId),
    label: firstNonEmptyString(route.label),
    fragmentId: world.fragmentId,
    fragmentLabel: world.fragmentLabel,
    frameId: world.frameId,
    world,
    pose: {
      position: [...normalizedPose.position],
      yawDegrees: normalizedPose.yawDegrees
    },
    waypoint,
    route: normalizedRoute
  };
}

async function resolveRemoteSource(sourceUrl) {
  const response = await fetch(sourceUrl);

  if (!response.ok) {
    throw new Error(`source route の download に失敗しました: ${response.status} ${response.statusText}`);
  }

  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'dreamwalker-route-'));
  const tempPath = path.join(tempRoot, 'route.json');
  const body = Buffer.from(await response.arrayBuffer());
  await writeFile(tempPath, body);

  return {
    localPath: tempPath,
    displayPath: sourceUrl,
    cleanupPath: tempRoot
  };
}

async function resolveSourceInput(source) {
  if (isRemoteUrl(source)) {
    return resolveRemoteSource(source);
  }

  const localPath = toAbsolutePath(source);
  return {
    localPath,
    displayPath: localPath,
    cleanupPath: ''
  };
}

function upsertRouteCatalogEntry(catalog, nextEntry) {
  const nextRoutes = Array.isArray(catalog.routes) ? [...catalog.routes] : [];
  const existingIndex = nextRoutes.findIndex((entry) =>
    entry.id === nextEntry.id || entry.url === nextEntry.url
  );

  if (existingIndex >= 0) {
    nextRoutes[existingIndex] = {
      ...nextRoutes[existingIndex],
      ...nextEntry
    };
  } else {
    nextRoutes.push(nextEntry);
  }

  return {
    ...catalog,
    version: catalog.version ?? 1,
    label: catalog.label ?? 'Dynamic Map Viewer Public Robot Route Catalog',
    note: catalog.note ?? 'repo 同梱の robot route preset 一覧。robot sandbox の導線や demo route をここへ足す。',
    routes: nextRoutes
  };
}

async function writeRouteFile(targetPath, value, force) {
  await mkdir(path.dirname(targetPath), { recursive: true });

  if (await pathExists(targetPath)) {
    if (!force) {
      throw new Error(`出力先が既に存在します: ${targetPath}\n--force で上書きできます。`);
    }
    await writeJson(targetPath, value);
    return 'overwritten';
  }

  await writeJson(targetPath, value);
  return 'created';
}

async function main() {
  const cleanupPaths = [];
  const args = parseArgs(process.argv.slice(2));

  try {
    if (args.help) {
      printUsage();
      return;
    }

    if (!hasNonEmptyString(args.source)) {
      throw new Error('--source は必須です。');
    }

    const sourceInput = await resolveSourceInput(args.source);
    if (sourceInput.cleanupPath) {
      cleanupPaths.push(sourceInput.cleanupPath);
    }

    const rawRoute = JSON.parse(await readFile(sourceInput.localPath, 'utf8'));
    const sourceRoute = normalizeRoutePayload(rawRoute);
    const fragmentId = firstNonEmptyString(args.fragment, sourceRoute.fragmentId);

    if (!fragmentId) {
      throw new Error('fragment を解決できませんでした。--fragment か source route.fragmentId を指定してください。');
    }

    if (!Object.hasOwn(dreamwalkerConfig.fragments, fragmentId)) {
      throw new Error(`未知の fragment です: ${fragmentId}`);
    }

    const activeConfig = resolveDreamwalkerConfig(fragmentId);
    const publicRoot = toAbsolutePath(args['public-root'] ?? defaultPublicRoot);
    const assetManifestPath = toAbsolutePath(args['asset-manifest'] ?? defaultAssetManifestPath);
    const catalogPath = toAbsolutePath(args['catalog-path'] ?? defaultCatalogPath);
    const assetManifest = await readJsonOrFallback(assetManifestPath, {
      version: 1,
      label: 'Local Dynamic Map Asset Manifest',
      fragments: {}
    });
    const assetBundle = resolveWorldAssetBundle(activeConfig, assetManifest);
    const routeId = sanitizeId(
      args['route-id'] ??
        sourceRoute.label ??
        path.basename(sourceInput.displayPath, path.extname(sourceInput.displayPath)),
      `${fragmentId}-route`
    );
    const routePath = toAbsolutePath(
      args['route-path'] ?? path.join(publicRoot, 'robot-routes', `${routeId}.json`)
    );
    const routeUrl = buildPublicUrl(publicRoot, routePath);
    const routeLabel = firstNonEmptyString(
      args['route-label'],
      sourceRoute.label,
      `${activeConfig.fragmentLabel} Route Preset`
    );
    const world = normalizeWorldContext(
      {
        ...sourceRoute.world,
        fragmentId,
        fragmentLabel: activeConfig.fragmentLabel,
        assetLabel: firstNonEmptyString(args['asset-label'], sourceRoute.world.assetLabel, assetBundle.assetLabel),
        manifestLabel: firstNonEmptyString(sourceRoute.world.manifestLabel, assetBundle.manifestLabel),
        splatUrl: firstNonEmptyString(args['splat-url'], sourceRoute.world.splatUrl, assetBundle.splatUrl),
        colliderMeshUrl: firstNonEmptyString(
          args['collider-url'],
          sourceRoute.world.colliderMeshUrl,
          assetBundle.colliderMeshUrl
        ),
        frameId: firstNonEmptyString(args['frame-id'], sourceRoute.world.frameId, sourceRoute.frameId),
        zoneMapUrl: firstNonEmptyString(
          args['zone-map-url'],
          sourceRoute.world.zoneMapUrl,
          activeConfig.robotics.semanticZoneMapUrl
        ),
        usesDemoFallback:
          Object.prototype.hasOwnProperty.call(sourceRoute.world, 'usesDemoFallback')
            ? sourceRoute.world.usesDemoFallback
            : assetBundle.usesDemoFallback
      },
      {
        fragmentId,
        fragmentLabel: activeConfig.fragmentLabel,
        assetLabel: assetBundle.assetLabel,
        manifestLabel: assetBundle.manifestLabel,
        splatUrl: assetBundle.splatUrl,
        colliderMeshUrl: assetBundle.colliderMeshUrl,
        frameId: sourceRoute.frameId,
        zoneMapUrl: activeConfig.robotics.semanticZoneMapUrl,
        usesDemoFallback: assetBundle.usesDemoFallback
      }
    );

    const nextRoute = {
      version: 1,
      protocol: routeProtocolId,
      label: routeLabel,
      description: firstNonEmptyString(args.description, sourceRoute.description),
      accent: firstNonEmptyString(
        args.accent,
        sourceRoute.accent,
        activeConfig.overlayBranding?.highlight,
        activeConfig.overlayBranding?.accent,
        '#85e3e1'
      ),
      fragmentId,
      fragmentLabel: activeConfig.fragmentLabel,
      frameId: world.frameId,
      world,
      pose: sourceRoute.pose,
      waypoint: sourceRoute.waypoint,
      route: sourceRoute.route
    };
    let zoneCoverageSummary = null;

    if (world.zoneMapUrl && !isRemoteUrl(world.zoneMapUrl)) {
      const zoneMapPath = toAbsolutePath(world.zoneMapUrl.replace(/^\/+/, ''), publicRoot);

      if (await pathExists(zoneMapPath)) {
        const zonePayload = JSON.parse(await readFile(zoneMapPath, 'utf8'));
        const zoneMap = buildSemanticZoneMap(zonePayload);
        zoneCoverageSummary = summarizeRouteZoneCoverage(nextRoute, zoneMap);
      }
    }

    const catalog = await readJsonOrFallback(catalogPath, {
      version: 1,
      label: 'Dynamic Map Viewer Public Robot Route Catalog',
      note: 'stage-robot-route.mjs により生成される public robot route catalog。',
      routes: []
    });
    const nextCatalog = upsertRouteCatalogEntry(catalog, {
      id: routeId,
      label: routeLabel,
      url: routeUrl,
      description: firstNonEmptyString(
        args.description,
        sourceRoute.description,
        `${activeConfig.fragmentLabel} robot route preset / published by stage-robot-route`
      ),
      fragmentId,
      accent: firstNonEmptyString(
        args.accent,
        sourceRoute.accent,
        activeConfig.overlayBranding?.highlight,
        activeConfig.overlayBranding?.accent,
        '#85e3e1'
      )
    });

    const summary = [
      `source: ${sourceInput.displayPath}`,
      `fragment: ${fragmentId}`,
      `route id: ${routeId}`,
      `route output: ${routeUrl}`,
      `catalog: ${catalogPath}`,
      `world asset: ${world.assetLabel || activeConfig.fragmentLabel}`,
      `world splat: ${world.splatUrl || '(none)'}`,
      `world collider: ${world.colliderMeshUrl || '(none)'}`,
      `launch: /?robotRoute=${routeUrl}`
    ];

    if (args['dry-run']) {
      console.log('Dry run only. No files were written.');
      summary.forEach((line) => console.log(`- ${line}`));
      return;
    }

    const routeWriteStatus = await writeRouteFile(routePath, nextRoute, Boolean(args.force));
    await writeJson(catalogPath, nextCatalog);

    console.log(`Staged robot route preset.`);
    console.log(`- route ${routeWriteStatus}: ${routeUrl}`);
    console.log(`- catalog updated: ${catalogPath}`);
    console.log(`- launch: /?robotRoute=${routeUrl}`);
    console.log(`- world: ${world.assetLabel || activeConfig.fragmentLabel} / frame ${world.frameId}`);
    if (zoneCoverageSummary) {
      console.log(
        `- zone coverage: ${zoneCoverageSummary.hitNodeCount}/${nextRoute.route.length} nodes / maxCost ${zoneCoverageSummary.maxCost}`
      );
      if (zoneCoverageSummary.nodeCount > zoneCoverageSummary.hitNodeCount) {
        console.log(
          `- uncovered nodes: ${zoneCoverageSummary.nodeCount - zoneCoverageSummary.hitNodeCount - zoneCoverageSummary.outsideBoundsCount}`
        );
      }
      if (zoneCoverageSummary.hazardNodeCount > 0) {
        console.log(`WARN: hazard nodes ${zoneCoverageSummary.hazardNodeCount}/${nextRoute.route.length}`);
      }
      if (zoneCoverageSummary.outsideBoundsCount > 0) {
        console.log(`WARN: route leaves zone bounds at ${zoneCoverageSummary.outsideBoundsCount} node(s)`);
      }
      if (zoneCoverageSummary.labels.length > 0) {
        console.log(`- zones: ${zoneCoverageSummary.labels.slice(0, 5).join(', ')}`);
      }
    }
    console.log('Next: npm run validate:studio');
  } finally {
    await Promise.all(
      cleanupPaths.map((cleanupPath) => rm(cleanupPath, { recursive: true, force: true }))
    );
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`ERR: ${message}`);
  process.exitCode = 1;
});
