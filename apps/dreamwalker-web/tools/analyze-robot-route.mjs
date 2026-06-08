import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {
  analyzeRouteAgainstZoneMap,
  buildRouteTuningDiagnostics
} from '../src/robot-route-analysis.js';
import { buildSemanticZoneMap } from '../src/semantic-zones.js';

const analysisProtocolId = 'dreamwalker-robot-route-analysis/v1';

function printUsage() {
  console.log(`Dynamic Map Viewer robot route analysis

Usage:
  node ./tools/analyze-robot-route.mjs --route ./public/robot-routes/residency-window-loop.json --zones ./public/manifests/robotics-residency.zones.json

Options:
  --route <file|url>       route JSON. required.
  --zones <file|url>       semantic zone JSON. required.
  --corridor-padding <n>   suggested safe corridor padding in meters. default: 0.75
  --bounds-padding <n>     suggested zone bounds padding in meters. default: corridor padding
  --json                   print JSON report to stdout.
  --output <file>          also write JSON report to file.
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
    if (key === 'json' || key === 'help') {
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

function firstNonEmptyString(...values) {
  return values.find((value) => hasNonEmptyString(value))?.trim() ?? '';
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

async function readJsonInput(source) {
  if (isRemoteUrl(source)) {
    const response = await fetch(source);

    if (!response.ok) {
      throw new Error(`download に失敗しました: ${response.status} ${response.statusText}`);
    }

    return JSON.parse(await response.text());
  }

  const filePath = toAbsolutePath(source);
  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
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
    protocol: firstNonEmptyString(route.protocol, 'dreamwalker-robot-route/v1'),
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

function buildReport(route, zoneMap, analysis) {
  const diagnostics = buildRouteTuningDiagnostics(route, zoneMap, analysis, {
    corridorPadding: Number.isFinite(Number(route?.analysisOptions?.corridorPadding))
      ? Number(route.analysisOptions.corridorPadding)
      : undefined,
    boundsPadding: Number.isFinite(Number(route?.analysisOptions?.boundsPadding))
      ? Number(route.analysisOptions.boundsPadding)
      : undefined
  });

  return {
    version: analysisProtocolId,
    generatedAt: new Date().toISOString(),
    route: {
      label: route.label || route.world.assetLabel || route.fragmentLabel || 'Robot Route',
      protocol: route.protocol,
      fragmentId: route.fragmentId,
      fragmentLabel: route.fragmentLabel,
      frameId: route.frameId,
      nodeCount: analysis.nodeCount,
      pose: route.pose,
      waypoint: route.waypoint,
      world: route.world
    },
    zoneMap: {
      frameId: zoneMap.frameId,
      zoneCount: zoneMap.zones.length,
      resolution: zoneMap.resolution,
      defaultCost: zoneMap.defaultCost,
      bounds: {
        minX: zoneMap.minX,
        maxX: zoneMap.maxX,
        minZ: zoneMap.minZ,
        maxZ: zoneMap.maxZ
      }
    },
    compatibility: {
      frameMatch: route.frameId === zoneMap.frameId
    },
    summary: {
      hitNodeCount: analysis.hitNodeCount,
      outsideBoundsCount: analysis.outsideBoundsCount,
      hazardNodeCount: analysis.hazardNodeCount,
      maxCost: analysis.maxCost,
      labels: analysis.labels
    },
    diagnostics,
    recommendations: diagnostics.recommendations,
    nodes: analysis.nodes
  };
}

function printTextReport(report, outputPath = '') {
  console.log('Dynamic Map Viewer Robot Route Analysis');
  console.log(`- route: ${report.route.label}`);
  console.log(
    `- fragment: ${report.route.fragmentId || '(none)'} / ${report.route.fragmentLabel || '(none)'}`
  );
  console.log(
    `- frame: route=${report.route.frameId} / zone=${report.zoneMap.frameId}${report.compatibility.frameMatch ? '' : ' (drift)'}`
  );
  console.log(
    `- nodes: ${report.summary.hitNodeCount}/${report.route.nodeCount} covered / ${report.summary.outsideBoundsCount} outside bounds / ${report.summary.hazardNodeCount} hazard`
  );
  console.log(`- uncovered: ${report.diagnostics.uncoveredNodeCount}`);
  console.log(
    `- zone map: ${report.zoneMap.zoneCount} zones / resolution ${report.zoneMap.resolution} / defaultCost ${report.zoneMap.defaultCost}`
  );
  console.log(`- max cost: ${report.summary.maxCost}`);

  if (report.summary.labels.length > 0) {
    console.log(`- zones: ${report.summary.labels.join(', ')}`);
  }

  if (report.recommendations.length > 0) {
    console.log('- recommendations:');
    report.recommendations.forEach((recommendation) => {
      const range = Array.isArray(recommendation.nodeRange)
        ? ` node ${recommendation.nodeRange[0]}-${recommendation.nodeRange[1]}`
        : '';
      console.log(`  - [${recommendation.kind}]${range} ${recommendation.message}`);
    });
  }

  if (outputPath) {
    console.log(`- wrote: ${outputPath}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  if (!hasNonEmptyString(args.route)) {
    throw new Error('--route は必須です。');
  }

  if (!hasNonEmptyString(args.zones)) {
    throw new Error('--zones は必須です。');
  }

  const analysisOptions = {
    corridorPadding: hasNonEmptyString(args['corridor-padding'])
      ? Number(args['corridor-padding'])
      : undefined,
    boundsPadding: hasNonEmptyString(args['bounds-padding'])
      ? Number(args['bounds-padding'])
      : undefined
  };

  if (
    analysisOptions.corridorPadding !== undefined &&
    (!Number.isFinite(analysisOptions.corridorPadding) || analysisOptions.corridorPadding < 0)
  ) {
    throw new Error('--corridor-padding は 0 以上の数値で指定してください。');
  }

  if (
    analysisOptions.boundsPadding !== undefined &&
    (!Number.isFinite(analysisOptions.boundsPadding) || analysisOptions.boundsPadding < 0)
  ) {
    throw new Error('--bounds-padding は 0 以上の数値で指定してください。');
  }

  const rawRoute = await readJsonInput(args.route);
  const rawZones = await readJsonInput(args.zones);
  const route = normalizeRoutePayload(rawRoute);
  route.analysisOptions = analysisOptions;
  const zoneMap = buildSemanticZoneMap(rawZones);
  const analysis = analyzeRouteAgainstZoneMap(route, zoneMap);
  const report = buildReport(route, zoneMap, analysis);
  const outputPath = hasNonEmptyString(args.output) ? toAbsolutePath(args.output) : '';

  if (outputPath) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  }

  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  printTextReport(report, outputPath);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`ERR: ${message}`);
  process.exitCode = 1;
});
