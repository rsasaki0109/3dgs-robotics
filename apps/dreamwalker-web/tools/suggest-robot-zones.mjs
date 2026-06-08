import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {
  analyzeRouteAgainstZoneMap,
  buildRouteTuningDiagnostics
} from '../src/robot-route-analysis.js';
import { buildSemanticZoneMap, serializeSemanticZoneMap } from '../src/semantic-zones.js';

function printUsage() {
  console.log(`Dynamic Map Viewer robot zone suggestion

Usage:
  node ./tools/suggest-robot-zones.mjs --route ./public/robot-routes/residency-window-loop.json --zones ./public/manifests/robotics-residency.zones.json

Options:
  --route <file|url>         route JSON. required.
  --zones <file|url>         semantic zone JSON. required.
  --output <file>            write suggested zone JSON.
  --corridor-padding <n>     suggested safe corridor padding. default: 0.75
  --bounds-padding <n>       suggested bounds padding. default: corridor padding
  --cost <n>                 cost for generated safe corridor zones. default: 15
  --label-prefix <text>      generated zone label prefix. default: Suggested Corridor
  --include-hazard-review    generate review rects for hazard overlap segments.
  --hazard-cost <n>          cost for generated hazard review zones. default: 65
  --hazard-label-prefix <t>  hazard review label prefix. default: Hazard Review
  --include-bounds-review    generate review rects for outside-bounds segments.
  --bounds-cost <n>          cost for generated bounds review zones. default: 30
  --bounds-label-prefix <t>  bounds review label prefix. default: Bounds Review
  --merge-bounds             expand zone bounds to include suggested outside-bounds coverage.
  --json                     print suggested zone JSON to stdout.
  --help                     show this message.
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
      key === 'json' ||
      key === 'help' ||
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

  const raw = await readFile(toAbsolutePath(source), 'utf8');
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

  return {
    fragmentId: hasNonEmptyString(route.fragmentId) ? route.fragmentId.trim() : '',
    fragmentLabel: hasNonEmptyString(route.fragmentLabel) ? route.fragmentLabel.trim() : '',
    frameId: hasNonEmptyString(route.frameId) ? route.frameId.trim() : 'dreamwalker_map',
    label: hasNonEmptyString(route.label) ? route.label.trim() : 'Robot Route',
    pose: normalizedPose,
    waypoint,
    route: points.length > 0 ? points : [[...normalizedPose.position]]
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

function buildSuggestedZone(segment, index, options) {
  const rect = segment?.suggestedRect ?? segment?.suggestedBounds;

  if (!rect) {
    return null;
  }

  const kind = hasNonEmptyString(options.kind) ? options.kind.trim() : 'suggested';
  const labelPrefix = hasNonEmptyString(options.labelPrefix)
    ? options.labelPrefix.trim()
    : 'Suggested Zone';
  const zoneId = sanitizeId(
    `${labelPrefix}-${segment.nodeRange?.[0] ?? index}-${segment.nodeRange?.[1] ?? index}`,
    `${kind}-${index + 1}`
  );
  const zoneLabel = `${labelPrefix} ${segment.nodeRange?.[0] ?? index}-${segment.nodeRange?.[1] ?? index}`;
  const tags = Array.isArray(options.tags) ? options.tags.filter(hasNonEmptyString) : [];

  return {
    id: zoneId,
    label: zoneLabel,
    shape: 'rect',
    center: rect.center,
    size: rect.size,
    cost: Math.round(options.cost),
    tags
  };
}

function mergeBounds(basePayload, diagnostics, shouldMergeBounds) {
  const currentBounds = basePayload?.bounds ?? {};
  const suggestedBounds = diagnostics.routeBoundsWithPadding;

  if (!shouldMergeBounds || !suggestedBounds) {
    return currentBounds;
  }

  return {
    minX: Math.min(Number(currentBounds.minX), suggestedBounds.minX),
    maxX: Math.max(Number(currentBounds.maxX), suggestedBounds.maxX),
    minZ: Math.min(Number(currentBounds.minZ), suggestedBounds.minZ),
    maxZ: Math.max(Number(currentBounds.maxZ), suggestedBounds.maxZ)
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  if (!hasNonEmptyString(args.route) || !hasNonEmptyString(args.zones)) {
    throw new Error('--route と --zones は必須です。');
  }

  const corridorPadding = hasNonEmptyString(args['corridor-padding'])
    ? Number(args['corridor-padding'])
    : 0.75;
  const boundsPadding = hasNonEmptyString(args['bounds-padding'])
    ? Number(args['bounds-padding'])
    : corridorPadding;
  const generatedCost = hasNonEmptyString(args.cost) ? Number(args.cost) : 15;
  const labelPrefix = hasNonEmptyString(args['label-prefix'])
    ? args['label-prefix'].trim()
    : 'Suggested Corridor';
  const hazardCost = hasNonEmptyString(args['hazard-cost']) ? Number(args['hazard-cost']) : 65;
  const hazardLabelPrefix = hasNonEmptyString(args['hazard-label-prefix'])
    ? args['hazard-label-prefix'].trim()
    : 'Hazard Review';
  const boundsCost = hasNonEmptyString(args['bounds-cost']) ? Number(args['bounds-cost']) : 30;
  const boundsLabelPrefix = hasNonEmptyString(args['bounds-label-prefix'])
    ? args['bounds-label-prefix'].trim()
    : 'Bounds Review';

  if (!Number.isFinite(corridorPadding) || corridorPadding < 0) {
    throw new Error('--corridor-padding は 0 以上の数値で指定してください。');
  }

  if (!Number.isFinite(boundsPadding) || boundsPadding < 0) {
    throw new Error('--bounds-padding は 0 以上の数値で指定してください。');
  }

  if (!Number.isFinite(generatedCost) || generatedCost < 0 || generatedCost > 100) {
    throw new Error('--cost は 0 から 100 の数値で指定してください。');
  }

  if (!Number.isFinite(hazardCost) || hazardCost < 0 || hazardCost > 100) {
    throw new Error('--hazard-cost は 0 から 100 の数値で指定してください。');
  }

  if (!Number.isFinite(boundsCost) || boundsCost < 0 || boundsCost > 100) {
    throw new Error('--bounds-cost は 0 から 100 の数値で指定してください。');
  }

  const rawRoute = await readJsonInput(args.route);
  const rawZones = await readJsonInput(args.zones);
  const route = normalizeRoutePayload(rawRoute);
  const zoneMap = buildSemanticZoneMap(rawZones);
  const analysis = analyzeRouteAgainstZoneMap(route, zoneMap);
  const diagnostics = buildRouteTuningDiagnostics(route, zoneMap, analysis, {
    corridorPadding,
    boundsPadding
  });
  const serializedZoneMap = serializeSemanticZoneMap(zoneMap);
  const corridorZones = diagnostics.recommendations
    .filter((recommendation) => recommendation.kind === 'uncovered-corridor')
    .map((recommendation, index) =>
      buildSuggestedZone(recommendation, index, {
        kind: 'suggested-corridor',
        labelPrefix,
        cost: generatedCost,
        tags: ['safe', 'corridor', 'suggested']
      })
    )
    .filter(Boolean);
  const hazardReviewZones = Boolean(args['include-hazard-review'])
    ? diagnostics.recommendations
        .filter((recommendation) => recommendation.kind === 'hazard-overlap')
        .map((recommendation, index) =>
          buildSuggestedZone(recommendation, index, {
            kind: 'hazard-review',
            labelPrefix: hazardLabelPrefix,
            cost: hazardCost,
            tags: ['review', 'hazard-overlap', 'suggested']
          })
        )
        .filter(Boolean)
    : [];
  const boundsReviewZones = Boolean(args['include-bounds-review'])
    ? diagnostics.recommendations
        .filter((recommendation) => recommendation.kind === 'outside-bounds')
        .map((recommendation, index) =>
          buildSuggestedZone(recommendation, index, {
            kind: 'bounds-review',
            labelPrefix: boundsLabelPrefix,
            cost: boundsCost,
            tags: ['review', 'bounds', 'suggested']
          })
        )
        .filter(Boolean)
    : [];
  const suggestedZones = [
    ...corridorZones,
    ...hazardReviewZones,
    ...boundsReviewZones
  ];

  const nextPayload = {
    ...serializedZoneMap,
    bounds: mergeBounds(serializedZoneMap, diagnostics, Boolean(args['merge-bounds'])),
    zones: [
      ...serializedZoneMap.zones,
      ...suggestedZones
    ],
    suggestionSummary: {
      routeLabel: route.label,
      generatedCorridorCount: corridorZones.length,
      generatedHazardReviewCount: hazardReviewZones.length,
      generatedBoundsReviewCount: boundsReviewZones.length,
      uncoveredSegmentCount: diagnostics.uncoveredSegments.length,
      hazardSegmentCount: diagnostics.hazardSegments.length,
      outsideBoundsSegmentCount: diagnostics.outsideBoundsSegments.length
    }
  };

  const outputPath = hasNonEmptyString(args.output) ? toAbsolutePath(args.output) : '';

  if (outputPath) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(nextPayload, null, 2)}\n`, 'utf8');
  }

  if (args.json) {
    console.log(JSON.stringify(nextPayload, null, 2));
    return;
  }

  console.log('Dynamic Map Viewer Robot Zone Suggestion');
  console.log(`- route: ${route.label}`);
  console.log(`- frame: route=${route.frameId} / zone=${zoneMap.frameId}`);
  console.log(`- generated corridors: ${corridorZones.length}`);
  if (args['include-hazard-review']) {
    console.log(`- generated hazard reviews: ${hazardReviewZones.length}`);
  }
  if (args['include-bounds-review']) {
    console.log(`- generated bounds reviews: ${boundsReviewZones.length}`);
  }
  console.log(`- uncovered segments: ${diagnostics.uncoveredSegments.length}`);
  console.log(`- hazard segments: ${diagnostics.hazardSegments.length}`);
  console.log(`- outside-bounds segments: ${diagnostics.outsideBoundsSegments.length}`);
  if (args['merge-bounds']) {
    console.log('- bounds: merged with routeBoundsWithPadding');
  }

  suggestedZones.forEach((zone) => {
    console.log(`- zone: ${zone.label} / size ${zone.size[0]} x ${zone.size[1]} / cost ${zone.cost}`);
  });

  if (outputPath) {
    console.log(`- wrote: ${outputPath}`);
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`ERR: ${message}`);
  process.exitCode = 1;
});
