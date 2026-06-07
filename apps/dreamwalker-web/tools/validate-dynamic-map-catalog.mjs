import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { normalizeDynamicMapPreloadMode, normalizeDynamicMapTileCatalog } from '../src/dynamic-map-loading.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const defaultPublicRoot = path.join(appRoot, 'public');
const defaultSiteUrl = 'http://localhost:5173/';

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function isRemoteUrl(value) {
  return typeof value === 'string' && /^https?:\/\//i.test(value.trim());
}

function isLocalPublicUrl(value) {
  const normalized = hasNonEmptyString(value) ? value.trim() : '';
  return normalized.startsWith('/') && !normalized.startsWith('//');
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
  const args = {
    positional: []
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token.startsWith('--')) {
      args.positional.push(token);
      continue;
    }

    const key = token.slice(2);
    if (key === 'help') {
      args[key] = true;
      continue;
    }

    const nextToken = argv[index + 1];
    if (!nextToken || nextToken.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }

    args[key] = nextToken;
    index += 1;
  }

  return args;
}

function printUsage() {
  console.log(`DreamWalker dynamic map catalog validation

Usage:
  node ./tools/validate-dynamic-map-catalog.mjs public/manifests/scene.json
  node ./tools/validate-dynamic-map-catalog.mjs --catalog /manifests/scene.json --site-url http://localhost:5173/

Options:
  --catalog <file|url>       large-scale 3DGS tile catalog JSON. Positional input is also accepted.
  --route <file|url>         optional robot route JSON to validate against ready tiles.
  --public-root <dir>        local public root for checking /splats/... files.
  --site-url <url>           DreamWalker site URL for the printed launch URL.
  --preload-mode <mode>      tile preload mode for the printed launch URL: metadata, cache, or off.
  --tile-id <id>             optional tile override for the printed launch URL.
  --route-playback <0|1>     include robotRoutePlayback=1 in the printed launch URL.
  --route-playback-ms <ms>   route playback interval for the printed launch URL.
  --route-playback-loop <0|1> include robotRoutePlaybackLoop=1 in the printed launch URL.
  --help                     show this message.
`);
}

async function fileExists(filePath) {
  if (!filePath) {
    return false;
  }

  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function toPublicFilePath(publicRoot, assetUrl) {
  if (!isLocalPublicUrl(assetUrl)) {
    return '';
  }

  return path.join(publicRoot, assetUrl.trim().replace(/^\/+/, ''));
}

function toPublicUrlFromFilePath(filePath, publicRoot) {
  const relativePath = path.relative(publicRoot, filePath);

  if (!relativePath || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return '';
  }

  return `/${relativePath.split(path.sep).map(encodeURIComponent).join('/')}`;
}

function resolveCatalogBrowserUrl(catalogInput, publicRoot) {
  if (!hasNonEmptyString(catalogInput)) {
    return '';
  }

  const normalizedInput = catalogInput.trim();
  if (isRemoteUrl(normalizedInput)) {
    return normalizedInput;
  }

  const absoluteInput = toAbsolutePath(normalizedInput);
  const publicUrl = toPublicUrlFromFilePath(absoluteInput, publicRoot);
  if (publicUrl) {
    return publicUrl;
  }

  return isLocalPublicUrl(normalizedInput) ? normalizedInput : '';
}

async function readJsonInput(input, publicRoot) {
  if (isRemoteUrl(input)) {
    const response = await fetch(input);

    if (!response.ok) {
      throw new Error(`Failed to download ${input}: ${response.status} ${response.statusText}`);
    }

    return JSON.parse(await response.text());
  }

  const normalizedInput = hasNonEmptyString(input) ? input.trim() : '';
  const absoluteInput = toAbsolutePath(normalizedInput);
  const filePath = path.isAbsolute(normalizedInput) && await fileExists(absoluteInput)
    ? absoluteInput
    : isLocalPublicUrl(normalizedInput)
      ? toPublicFilePath(publicRoot, normalizedInput)
      : absoluteInput;
  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

function createFinding(level, scope, detail) {
  return {
    level,
    scope,
    detail
  };
}

function countFindings(findings, level) {
  return findings.filter((finding) => finding.level === level).length;
}

function readBooleanOption(value) {
  if (typeof value === 'boolean') {
    return value;
  }

  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  return ['1', 'true', 'yes', 'on', 'play', 'auto'].includes(normalized);
}

function formatNumber(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)).toString() : String(value);
}

function formatAxisRange(axis, range) {
  return `${axis}[${formatNumber(range.min)}, ${formatNumber(range.max)}]`;
}

function normalizeAxes(value) {
  return hasNonEmptyString(value)
    ? value.trim().toLowerCase().slice(0, 2).split('').filter(Boolean)
    : [];
}

function inferTileAxes(tile, fallbackAxes = ['x', 'z']) {
  const tileAxes = normalizeAxes(tile?.axes);
  if (tileAxes.length >= 2) {
    return tileAxes;
  }

  const boundKeys = Object.keys(tile?.coreBounds ?? {});
  const discoveredAxes = boundKeys
    .map((key) => key.match(/^min([XYZ])$/)?.[1]?.toLowerCase())
    .filter(Boolean);

  return discoveredAxes.length >= 2 ? discoveredAxes.slice(0, 2) : fallbackAxes;
}

function catalogAxes(catalog) {
  const tilingAxes = normalizeAxes(catalog?.tiling?.axes);
  if (tilingAxes.length >= 2) {
    return tilingAxes;
  }

  const tileWithAxes = catalog?.tiles?.find((tile) => normalizeAxes(tile?.axes).length >= 2);
  return inferTileAxes(tileWithAxes);
}

function readAxisRange(boundsLike, axis) {
  const bounds = boundsLike && typeof boundsLike === 'object' ? boundsLike : {};
  const upperAxis = axis.toUpperCase();
  const min = Number(bounds[`min${upperAxis}`]);
  const max = Number(bounds[`max${upperAxis}`]);

  return {
    axis,
    min,
    max,
    valid: Number.isFinite(min) && Number.isFinite(max) && min < max
  };
}

function readTileRanges(tile, boundsName, axes) {
  return Object.fromEntries(
    axes.map((axis) => [axis, readAxisRange(tile?.[boundsName], axis)])
  );
}

function areRangesValid(ranges, axes) {
  return axes.every((axis) => ranges[axis]?.valid);
}

function normalizeRoutePosition(positionLike) {
  if (Array.isArray(positionLike)) {
    const [x, y, z] = positionLike.map((value) => Number(value));
    return [x, y, z].every(Number.isFinite) ? { x, y, z } : null;
  }

  if (positionLike && typeof positionLike === 'object') {
    const x = Number(positionLike.x ?? positionLike.position?.[0]);
    const y = Number(positionLike.y ?? positionLike.position?.[1] ?? 0);
    const z = Number(positionLike.z ?? positionLike.position?.[2]);
    return [x, y, z].every(Number.isFinite) ? { x, y, z } : null;
  }

  return null;
}

function formatRoutePosition(position) {
  return `x=${formatNumber(position.x)} y=${formatNumber(position.y)} z=${formatNumber(position.z)}`;
}

function normalizeRobotRoute(routeLike) {
  const route = routeLike && typeof routeLike === 'object' ? routeLike : {};
  const positions = Array.isArray(route.route)
    ? route.route.map((position) => normalizeRoutePosition(position)).filter(Boolean)
    : [];
  const posePosition = normalizeRoutePosition(route.pose?.position ? route.pose : route.pose);

  if (posePosition && positions.length === 0) {
    positions.push(posePosition);
  }

  if (
    posePosition &&
    positions.length > 0 &&
    !positions.some((position) =>
      Math.abs(position.x - posePosition.x) < 1e-6 &&
      Math.abs(position.y - posePosition.y) < 1e-6 &&
      Math.abs(position.z - posePosition.z) < 1e-6
    )
  ) {
    positions.push(posePosition);
  }

  return {
    label: hasNonEmptyString(route.label) ? route.label.trim() : '',
    frameId: hasNonEmptyString(route.frameId) ? route.frameId.trim() : '',
    positions
  };
}

function isPositionInsideRanges(position, ranges, axes) {
  if (!position || !areRangesValid(ranges, axes)) {
    return false;
  }

  return axes.every((axis) => {
    const range = ranges[axis];
    return position[axis] >= range.min && position[axis] <= range.max;
  });
}

function selectReadyTileForRoutePosition(readyTiles, position, axes) {
  const coreMatches = readyTiles.filter((tile) =>
    isPositionInsideRanges(position, readTileRanges(tile, 'coreBounds', axes), axes)
  );
  if (coreMatches.length > 0) {
    return coreMatches[0];
  }

  const expandedMatches = readyTiles.filter((tile) =>
    isPositionInsideRanges(position, readTileRanges(tile, 'expandedBounds', axes), axes)
  );
  return expandedMatches[0] ?? null;
}

async function validateRouteAgainstCatalog(routeInput, catalog, axes, findings, options = {}) {
  if (!hasNonEmptyString(routeInput)) {
    return {
      route: null,
      routeUrl: '',
      tileSequence: []
    };
  }

  const publicRoot = toAbsolutePath(options.publicRoot ?? defaultPublicRoot);
  const routeSource = await readJsonInput(routeInput, publicRoot);
  const route = normalizeRobotRoute(routeSource);
  const routeUrl = resolveCatalogBrowserUrl(routeInput, publicRoot);
  const readyTiles = catalog.tiles.filter((tile) => tile.splatUrl && tile.status !== 'missing-splat');
  const tileSequence = [];
  let uncoveredCount = 0;

  findings.push(createFinding(
    'OK',
    'route:source',
    route.label || routeUrl || routeInput
  ));

  if (route.positions.length === 0) {
    findings.push(createFinding('ERROR', 'route:points', 'robot route has no valid pose or route points'));
    return {
      route,
      routeUrl,
      tileSequence
    };
  }

  findings.push(createFinding('OK', 'route:points', `${route.positions.length} point(s)`));

  route.positions.forEach((position, index) => {
    const tile = selectReadyTileForRoutePosition(readyTiles, position, axes);
    if (!tile) {
      uncoveredCount += 1;
      findings.push(createFinding(
        'ERROR',
        `route:point:${index + 1}`,
        `${formatRoutePosition(position)} is outside ready tile coverage`
      ));
      return;
    }

    if (tileSequence[tileSequence.length - 1] !== tile.id) {
      tileSequence.push(tile.id);
    }
  });

  if (tileSequence.length > 0) {
    findings.push(createFinding('OK', 'route:tile-sequence', tileSequence.join(' -> ')));
  }

  if (uncoveredCount === 0) {
    findings.push(createFinding('OK', 'route:coverage', 'all route points map to ready tiles'));
  }

  if (routeUrl) {
    findings.push(createFinding('OK', 'route:url', routeUrl));
  } else {
    findings.push(createFinding('WARN', 'route:url', 'route input is not under public root; launch URL will not include robotRoute'));
  }

  return {
    route,
    routeUrl,
    tileSequence
  };
}

function validateReadyTileBounds(tile, axes, findings) {
  const scope = `tile:${tile.id}`;
  const coreRanges = readTileRanges(tile, 'coreBounds', axes);
  const expandedRanges = readTileRanges(tile, 'expandedBounds', axes);
  const invalidCoreAxes = axes.filter((axis) => !coreRanges[axis]?.valid);
  const invalidExpandedAxes = axes.filter((axis) => !expandedRanges[axis]?.valid);

  if (invalidCoreAxes.length > 0) {
    findings.push(createFinding('ERROR', scope, `invalid coreBounds for axes: ${invalidCoreAxes.join(', ')}`));
  }

  if (invalidExpandedAxes.length > 0) {
    findings.push(createFinding('ERROR', scope, `invalid expandedBounds for axes: ${invalidExpandedAxes.join(', ')}`));
  }

  if (invalidCoreAxes.length > 0 || invalidExpandedAxes.length > 0) {
    return {
      valid: false,
      coreRanges,
      expandedRanges
    };
  }

  const outsideAxes = axes.filter((axis) => {
    const core = coreRanges[axis];
    const expanded = expandedRanges[axis];
    return expanded.min > core.min + 1e-6 || expanded.max < core.max - 1e-6;
  });

  if (outsideAxes.length > 0) {
    findings.push(createFinding('ERROR', scope, `expandedBounds does not contain coreBounds for axes: ${outsideAxes.join(', ')}`));
    return {
      valid: false,
      coreRanges,
      expandedRanges
    };
  }

  return {
    valid: true,
    coreRanges,
    expandedRanges
  };
}

function validateReadyTileIndex(tile, axes, findings) {
  const scope = `tile:${tile.id}`;
  const tileIndex = tile?.tileIndex && typeof tile.tileIndex === 'object' ? tile.tileIndex : {};
  const values = axes.map((axis) => Number(tileIndex[axis]));
  const missingAxes = axes.filter((axis, index) => !Number.isFinite(values[index]));

  if (missingAxes.length > 0) {
    findings.push(createFinding('WARN', scope, `missing tileIndex for axes: ${missingAxes.join(', ')}`));
    return null;
  }

  const nonIntegerAxes = axes.filter((axis, index) => !Number.isInteger(values[index]));
  if (nonIntegerAxes.length > 0) {
    findings.push(createFinding('ERROR', scope, `tileIndex must be integer for axes: ${nonIntegerAxes.join(', ')}`));
    return null;
  }

  return {
    tile,
    values,
    key: values.join(':')
  };
}

function unionCoreRanges(boundsResults, axes) {
  const validResults = boundsResults.filter((result) => result.valid);

  if (validResults.length === 0) {
    return null;
  }

  return Object.fromEntries(
    axes.map((axis) => [
      axis,
      {
        axis,
        min: Math.min(...validResults.map((result) => result.coreRanges[axis].min)),
        max: Math.max(...validResults.map((result) => result.coreRanges[axis].max))
      }
    ])
  );
}

function validateWorldBoundsCoverage(catalog, boundsResults, axes, findings) {
  const worldBounds = catalog?.tiling?.worldBounds;
  if (!worldBounds || typeof worldBounds !== 'object') {
    return;
  }

  const worldRanges = Object.fromEntries(
    axes.map((axis) => [axis, readAxisRange(worldBounds, axis)])
  );
  const invalidAxes = axes.filter((axis) => !worldRanges[axis]?.valid);
  if (invalidAxes.length > 0) {
    findings.push(createFinding('WARN', 'catalog:world-bounds', `invalid tiling.worldBounds for axes: ${invalidAxes.join(', ')}`));
    return;
  }

  const unionRanges = unionCoreRanges(boundsResults, axes);
  if (!unionRanges) {
    return;
  }

  const uncoveredAxes = axes.filter((axis) => {
    const unionRange = unionRanges[axis];
    const worldRange = worldRanges[axis];
    return unionRange.min > worldRange.min + 1e-6 || unionRange.max < worldRange.max - 1e-6;
  });

  if (uncoveredAxes.length > 0) {
    findings.push(createFinding(
      'WARN',
      'catalog:world-bounds',
      `ready core bounds do not cover tiling.worldBounds for axes: ${uncoveredAxes.join(', ')}`
    ));
    return;
  }

  findings.push(createFinding(
    'OK',
    'catalog:world-bounds',
    axes.map((axis) => formatAxisRange(axis, unionRanges[axis])).join(' ')
  ));
}

function tileKeyFromValues(values) {
  return values.join(':');
}

function summarizeMissingGridKeys(missingKeys) {
  const preview = missingKeys.slice(0, 8).join(', ');
  const suffix = missingKeys.length > 8 ? `, +${missingKeys.length - 8} more` : '';
  return `${preview}${suffix}`;
}

function enumerateRectangularGridKeys(indexedTiles, axes) {
  if (axes.length !== 2 || indexedTiles.length === 0) {
    return {
      expectedCount: indexedTiles.length,
      missingKeys: [],
      ranges: []
    };
  }

  const ranges = axes.map((axis, axisIndex) => {
    const values = indexedTiles.map((entry) => entry.values[axisIndex]);
    return {
      axis,
      min: Math.min(...values),
      max: Math.max(...values)
    };
  });
  const expectedCount = ranges.reduce((count, range) => count * (range.max - range.min + 1), 1);

  if (expectedCount > 20000) {
    return {
      expectedCount,
      missingKeys: [],
      ranges,
      skipped: true
    };
  }

  const seenKeys = new Set(indexedTiles.map((entry) => entry.key));
  const missingKeys = [];
  for (let first = ranges[0].min; first <= ranges[0].max; first += 1) {
    for (let second = ranges[1].min; second <= ranges[1].max; second += 1) {
      const key = tileKeyFromValues([first, second]);
      if (!seenKeys.has(key)) {
        missingKeys.push(`${axes[0]}${first}/${axes[1]}${second}`);
      }
    }
  }

  return {
    expectedCount,
    missingKeys,
    ranges
  };
}

function analyzeIndexedTileGrid(indexedTiles, axes, boundsByTileId, findings) {
  if (indexedTiles.length === 0) {
    return;
  }

  const byKey = new Map();
  for (const indexedTile of indexedTiles) {
    const existing = byKey.get(indexedTile.key);
    if (existing) {
      findings.push(createFinding('ERROR', `tile:${indexedTile.tile.id}`, `duplicate tileIndex with ${existing.tile.id}: ${indexedTile.key}`));
      continue;
    }

    byKey.set(indexedTile.key, indexedTile);
  }

  const uniqueIndexedTiles = [...byKey.values()];
  const grid = enumerateRectangularGridKeys(uniqueIndexedTiles, axes);
  const rangeSummary = grid.ranges
    .map((range) => `${range.axis}${range.min}..${range.max}`)
    .join(' ');

  if (grid.skipped) {
    findings.push(createFinding(
      'OK',
      'catalog:grid',
      `${uniqueIndexedTiles.length} indexed tile(s), rectangular coverage scan skipped for ${grid.expectedCount} slot(s)`
    ));
  } else if (grid.missingKeys.length > 0) {
    findings.push(createFinding(
      'WARN',
      'catalog:grid',
      `${uniqueIndexedTiles.length}/${grid.expectedCount} rectangular slot(s) occupied (${rangeSummary}); missing ${summarizeMissingGridKeys(grid.missingKeys)}`
    ));
  } else {
    findings.push(createFinding(
      'OK',
      'catalog:grid',
      `${uniqueIndexedTiles.length}/${grid.expectedCount} rectangular slot(s) occupied (${rangeSummary})`
    ));
  }

  let cardinalLinks = 0;
  let coreGapCount = 0;
  const coreGapExamples = [];

  for (const indexedTile of uniqueIndexedTiles) {
    for (let axisIndex = 0; axisIndex < axes.length; axisIndex += 1) {
      const neighborValues = [...indexedTile.values];
      neighborValues[axisIndex] += 1;
      const neighbor = byKey.get(tileKeyFromValues(neighborValues));
      if (!neighbor) {
        continue;
      }

      cardinalLinks += 1;
      const axis = axes[axisIndex];
      const leftBounds = boundsByTileId.get(indexedTile.tile.id);
      const rightBounds = boundsByTileId.get(neighbor.tile.id);
      if (!leftBounds?.valid || !rightBounds?.valid) {
        continue;
      }

      const gap = rightBounds.coreRanges[axis].min - leftBounds.coreRanges[axis].max;
      if (gap > 1e-6) {
        coreGapCount += 1;
        if (coreGapExamples.length < 6) {
          coreGapExamples.push(`${indexedTile.tile.id}->${neighbor.tile.id} ${axis} gap ${formatNumber(gap)}`);
        }
      }
    }
  }

  findings.push(createFinding('OK', 'catalog:adjacency', `${cardinalLinks} cardinal neighbor link(s)`));

  if (coreGapCount > 0) {
    findings.push(createFinding(
      'WARN',
      'catalog:core-coverage',
      `${coreGapCount} cardinal core gap(s): ${coreGapExamples.join(', ')}`
    ));
  }
}

function buildLaunchUrlFromCatalogUrl(catalogUrl, options = {}) {
  if (!hasNonEmptyString(catalogUrl)) {
    return '';
  }

  const siteUrl = hasNonEmptyString(options.siteUrl) ? options.siteUrl.trim() : defaultSiteUrl;
  const preloadMode = normalizeDynamicMapPreloadMode(options.preloadMode ?? 'metadata');
  const absoluteSiteUrl = isRemoteUrl(siteUrl);
  const launchUrl = new URL(siteUrl, 'http://dreamwalker.local');

  launchUrl.searchParams.set('tileCatalog', catalogUrl.trim());
  launchUrl.searchParams.set('tilePreload', preloadMode);

  if (hasNonEmptyString(options.tileId)) {
    launchUrl.searchParams.set('tileId', options.tileId.trim());
  }

  if (hasNonEmptyString(options.routeUrl)) {
    launchUrl.searchParams.set('robotRoute', options.routeUrl.trim());
  }

  if (readBooleanOption(options.routePlayback) || readBooleanOption(options.routePlaybackLoop)) {
    launchUrl.searchParams.set('robotRoutePlayback', '1');
  }

  const routePlaybackMs = Number(options.routePlaybackMs);
  if (Number.isFinite(routePlaybackMs) && routePlaybackMs > 0) {
    launchUrl.searchParams.set('robotRoutePlaybackMs', String(Math.floor(routePlaybackMs)));
  }

  if (readBooleanOption(options.routePlaybackLoop)) {
    launchUrl.searchParams.set('robotRoutePlaybackLoop', '1');
  }

  return absoluteSiteUrl
    ? launchUrl.toString()
    : `${launchUrl.pathname}${launchUrl.search}${launchUrl.hash}`;
}

export function buildDynamicMapCatalogLaunchUrl(catalogInput, options = {}) {
  const publicRoot = toAbsolutePath(options.publicRoot ?? defaultPublicRoot);
  const catalogUrl = resolveCatalogBrowserUrl(catalogInput, publicRoot);
  return buildLaunchUrlFromCatalogUrl(catalogUrl, options);
}

export async function validateDynamicMapCatalog(catalogInput, options = {}) {
  const publicRoot = toAbsolutePath(options.publicRoot ?? defaultPublicRoot);
  const source = await readJsonInput(catalogInput, publicRoot);
  const catalog = normalizeDynamicMapTileCatalog(source);
  const findings = [];

  if (catalog.type === 'large-scale-3dgs-tile-catalog') {
    findings.push(createFinding('OK', 'catalog:type', catalog.type));
  } else {
    findings.push(createFinding('ERROR', 'catalog:type', `expected large-scale-3dgs-tile-catalog, got ${catalog.type || 'none'}`));
  }

  if (catalog.tiles.length > 0) {
    findings.push(createFinding('OK', 'catalog:tiles', `${catalog.tiles.length} tile(s)`));
  } else {
    findings.push(createFinding('ERROR', 'catalog:tiles', 'catalog has no tiles'));
  }

  if (catalog.summary.readyTileCount > 0) {
    findings.push(createFinding('OK', 'catalog:ready', `${catalog.summary.readyTileCount} ready tile(s)`));
  } else {
    findings.push(createFinding('ERROR', 'catalog:ready', 'catalog has no ready tiles'));
  }

  const axes = catalogAxes(catalog);
  const seenTileIds = new Set();
  const seenSplatUrls = new Map();
  const readyTiles = [];
  const indexedTiles = [];
  const boundsResults = [];
  const boundsByTileId = new Map();

  for (const tile of catalog.tiles) {
    const scope = `tile:${tile.id}`;
    const isReadyTile = Boolean(tile.splatUrl) && tile.status !== 'missing-splat';

    if (seenTileIds.has(tile.id)) {
      findings.push(createFinding('ERROR', scope, 'duplicate tile id'));
    } else {
      seenTileIds.add(tile.id);
    }

    if (!isReadyTile) {
      findings.push(createFinding('WARN', scope, `not ready: ${tile.status || 'unknown'}`));
      continue;
    }

    readyTiles.push(tile);

    if (!tile.splatUrl) {
      findings.push(createFinding('ERROR', scope, 'ready tile has no splatUrl'));
      continue;
    }

    const existingSplatTileId = seenSplatUrls.get(tile.splatUrl);
    if (existingSplatTileId) {
      findings.push(createFinding('ERROR', scope, `duplicate splatUrl with ${existingSplatTileId}: ${tile.splatUrl}`));
    } else {
      seenSplatUrls.set(tile.splatUrl, tile.id);
    }

    const declaredTileAxes = normalizeAxes(tile.axes);
    if (declaredTileAxes.length >= 2 && declaredTileAxes.join('') !== axes.join('')) {
      findings.push(createFinding('WARN', scope, `tile axes ${declaredTileAxes.join('')} differ from catalog axes ${axes.join('')}`));
    }

    const boundsResult = validateReadyTileBounds(tile, axes, findings);
    boundsResults.push(boundsResult);
    boundsByTileId.set(tile.id, boundsResult);

    const indexedTile = validateReadyTileIndex(tile, axes, findings);
    if (indexedTile) {
      indexedTiles.push(indexedTile);
    }

    if (isRemoteUrl(tile.splatUrl)) {
      findings.push(createFinding('WARN', scope, `remote splat not checked: ${tile.splatUrl}`));
      continue;
    }

    if (!isLocalPublicUrl(tile.splatUrl)) {
      findings.push(createFinding('WARN', scope, `splatUrl is not a public URL: ${tile.splatUrl}`));
      continue;
    }

    if (!tile.splatUrl.startsWith('/splats/')) {
      findings.push(createFinding('WARN', scope, `local splatUrl is outside /splats: ${tile.splatUrl}`));
    }

    const splatPath = toPublicFilePath(publicRoot, tile.splatUrl);
    if (await fileExists(splatPath)) {
      findings.push(createFinding('OK', scope, `${tile.splatUrl} -> ${splatPath}`));
    } else {
      findings.push(createFinding('ERROR', scope, `missing local splat: ${tile.splatUrl} -> ${splatPath}`));
    }
  }

  if (readyTiles.length > 0 && boundsResults.every((result) => result.valid)) {
    findings.push(createFinding('OK', 'catalog:bounds', `${readyTiles.length} ready tile bound set(s) valid`));
  }

  if (readyTiles.length > 1 && indexedTiles.length === 0) {
    findings.push(createFinding('WARN', 'catalog:grid', 'ready tiles have no usable tileIndex; preload adjacency falls back to distance'));
  } else {
    analyzeIndexedTileGrid(indexedTiles, axes, boundsByTileId, findings);
  }

  validateWorldBoundsCoverage(catalog, boundsResults, axes, findings);

  const routeValidation = await validateRouteAgainstCatalog(
    options.routeInput ?? options.route,
    catalog,
    axes,
    findings,
    { publicRoot }
  );
  const catalogUrl = resolveCatalogBrowserUrl(catalogInput, publicRoot);
  const launchUrl = buildLaunchUrlFromCatalogUrl(catalogUrl, {
    ...options,
    routeUrl: routeValidation.routeUrl
  });

  if (catalogUrl) {
    findings.push(createFinding('OK', 'launch:catalog', catalogUrl));
    findings.push(createFinding('OK', 'launch:url', launchUrl));
  } else {
    findings.push(createFinding('WARN', 'launch:catalog', 'catalog input is not under public root; launch URL unavailable'));
  }

  const errorCount = countFindings(findings, 'ERROR');
  const warningCount = countFindings(findings, 'WARN');

  return {
    catalog,
    catalogUrl,
    launchUrl,
    route: routeValidation.route,
    routeTileSequence: routeValidation.tileSequence,
    routeUrl: routeValidation.routeUrl,
    findings,
    errorCount,
    warningCount,
    ok: errorCount === 0
  };
}

export function formatDynamicMapCatalogValidation(result) {
  const lines = ['Dynamic map catalog validation', ''];

  for (const finding of result.findings) {
    lines.push(`${finding.level.padEnd(5, ' ')} ${finding.scope.padEnd(24, ' ')} ${finding.detail}`);
  }

  lines.push('');
  lines.push(`Summary: ${result.errorCount} error(s), ${result.warningCount} warning(s)`);

  if (result.launchUrl) {
    lines.push(`Launch URL: ${result.launchUrl}`);
  }

  return lines.join('\n');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  const catalogInput = args.catalog ?? args.positional[0];
  if (!hasNonEmptyString(catalogInput)) {
    throw new Error('--catalog or positional catalog input is required');
  }

  const result = await validateDynamicMapCatalog(catalogInput, {
    publicRoot: args['public-root'],
    siteUrl: args['site-url'],
    preloadMode: args['preload-mode'],
    routeInput: args.route,
    routePlayback: args['route-playback'],
    routePlaybackLoop: args['route-playback-loop'],
    routePlaybackMs: args['route-playback-ms'],
    tileId: args['tile-id']
  });

  console.log(formatDynamicMapCatalogValidation(result));

  if (!result.ok) {
    process.exitCode = 1;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((error) => {
    console.error(`ERROR validate-dynamic-map-catalog ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  });
}
