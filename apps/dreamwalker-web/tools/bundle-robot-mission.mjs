import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { resolveDreamwalkerConfig } from '../src/app-config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const defaultPublicRoot = path.join(__dirname, '..', 'public');
const robotMissionArtifactPackProtocolId = 'dreamwalker-robot-mission-artifact-pack/v1';
const robotMissionPublishReportProtocolId = 'dreamwalker-robot-mission-publish-report/v1';

function printUsage() {
  console.log(`Dynamic Map Viewer robot mission bundle

Usage:
  node ./tools/bundle-robot-mission.mjs --mission ./public/robot-missions/residency-window-loop.mission.json
  node ./tools/bundle-robot-mission.mjs --mission ./public/robot-missions/residency-window-loop.mission.json --output ./dist/residency-window-loop.artifact-pack.json

Options:
  --mission <file|url>    required. source mission manifest.
  --route <file|url>      optional. default: mission.routeUrl resolved from public root.
  --zones <file|url>      optional. default: mission.zoneMapUrl resolved from public root.
  --public-root <dir>     local public root for resolving route/zone local URLs.
  --output <file>         output artifact pack path. default: ./<mission-id>.artifact-pack.json
  --label <text>          artifact pack label override.
  --help                  show this message.
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
    if (key === 'help') {
      args.help = true;
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

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function toAbsolutePath(inputPath, baseDir = process.cwd()) {
  if (!inputPath) {
    return '';
  }

  return path.isAbsolute(inputPath)
    ? inputPath
    : path.resolve(baseDir, inputPath);
}

function quoteShellValue(value) {
  const normalized = String(value ?? '');
  return `'${normalized.replace(/'/g, `'\\''`)}'`;
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

function buildPublicFilePath(publicRoot, assetUrl) {
  if (!hasNonEmptyString(assetUrl) || !assetUrl.trim().startsWith('/')) {
    return '';
  }

  return path.join(publicRoot, assetUrl.trim().replace(/^\/+/, ''));
}

async function resolveJsonSource({ explicitSource, fallbackUrl, publicRoot, label }) {
  if (hasNonEmptyString(explicitSource)) {
    return {
      source: explicitSource.trim(),
      json: await readJsonInput(explicitSource.trim())
    };
  }

  if (!hasNonEmptyString(fallbackUrl)) {
    return { source: '', json: null };
  }

  if (isRemoteUrl(fallbackUrl)) {
    return {
      source: fallbackUrl.trim(),
      json: await readJsonInput(fallbackUrl.trim())
    };
  }

  if (fallbackUrl.trim().startsWith('/')) {
    const filePath = buildPublicFilePath(publicRoot, fallbackUrl);
    if (!filePath) {
      throw new Error(`${label} の local public path を解決できません: ${fallbackUrl}`);
    }

    return {
      source: filePath,
      json: JSON.parse(await readFile(filePath, 'utf8'))
    };
  }

  return {
    source: fallbackUrl.trim(),
    json: await readJsonInput(fallbackUrl.trim())
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
  return routeMatch?.[1] ? sanitizeId(routeMatch[1], '') : '';
}

function buildMissionSlug(missionLike) {
  return sanitizeId(
    missionLike?.id ?? missionLike?.label ?? 'dreamwalker-robot-mission',
    'dreamwalker-robot-mission'
  );
}

function buildArtifactFileName(missionLike) {
  return `${buildMissionSlug(missionLike)}.artifact-pack.json`;
}

function buildPublishedMissionFileName(missionLike) {
  return `${buildMissionSlug(missionLike)}.mission.json`;
}

function buildPreflightSummary({
  healthLabel,
  healthDetail,
  mission,
  route,
  routeId,
  routeFileName,
  cameraPresetLabel,
  robotCameraLabel,
  streamSceneLabel
}) {
  return [
    `status: ${healthLabel || 'unknown'}`,
    `detail: ${healthDetail || 'none'}`,
    `missionId: ${mission.id || 'none'}`,
    `missionLabel: ${mission.label || 'none'}`,
    `missionDescription: ${mission.description || 'none'}`,
    `fragmentId: ${mission.fragmentId || 'none'}`,
    `fragmentLabel: ${mission.fragmentLabel || 'none'}`,
    `routeId: ${routeId || 'none'}`,
    `routeFile: ${routeFileName || 'none'}`,
    `routeLabel: ${route.label || 'none'}`,
    `routeDescription: ${route.description || 'none'}`,
    `routeAccent: ${route.accent || 'none'}`,
    `worldAsset: ${mission.world?.assetLabel || route.world?.assetLabel || 'none'}`,
    `worldFrame: ${mission.world?.frameId || route.frameId || 'none'}`,
    `zoneMapUrl: ${mission.zoneMapUrl || route.world?.zoneMapUrl || 'none'}`,
    `startupMode: ${mission.startupMode || 'none'}`,
    `cameraPresetId: ${mission.cameraPresetId || 'none'}`,
    `cameraPresetLabel: ${cameraPresetLabel || 'none'}`,
    `robotCameraId: ${mission.robotCameraId || 'none'}`,
    `robotCameraLabel: ${robotCameraLabel || 'none'}`,
    `streamSceneId: ${mission.streamSceneId || 'none'}`,
    `streamSceneLabel: ${streamSceneLabel || 'none'}`,
    `launchUrl: ${mission.launchUrl || 'none'}`
  ].join('\n');
}

function buildPublishReport({
  publicRoot,
  missionSource,
  routeSource,
  zoneSource,
  mission,
  route,
  routeId,
  routeFileName,
  cameraPresetLabel,
  robotCameraLabel,
  streamSceneLabel,
  preflightLabel,
  preflightDetail,
  preflightSummary
}) {
  return {
    version: 1,
    protocol: robotMissionPublishReportProtocolId,
    dryRun: false,
    fragmentId: mission.fragmentId || route.fragmentId || '',
    publicRoot,
    mission: {
      id: mission.id || '',
      label: mission.label || '',
      description: mission.description || '',
      accent: mission.accent || '',
      url: missionSource.startsWith('/') ? missionSource : mission.routeUrl || '',
      path: missionSource,
      launchUrl: mission.launchUrl || '',
      catalogPath: buildPublicFilePath(publicRoot, '/robot-missions/index.json'),
      catalogUrl: '/robot-missions/index.json'
    },
    route: {
      id: routeId,
      fileName: routeFileName,
      label: route.label || '',
      description: route.description || '',
      accent: route.accent || '',
      url: mission.routeUrl || '',
      path: routeSource,
      catalogPath: buildPublicFilePath(publicRoot, '/robot-routes/index.json'),
      source: routeSource
    },
    zones: {
      url: mission.zoneMapUrl || '',
      path: zoneSource,
      source: zoneSource
    },
    world: {
      assetLabel: mission.world?.assetLabel || route.world?.assetLabel || '',
      frameId: mission.world?.frameId || route.frameId || ''
    },
    startup: {
      mode: mission.startupMode || '',
      cameraPresetId: mission.cameraPresetId || '',
      cameraPresetLabel,
      robotCameraId: mission.robotCameraId || '',
      robotCameraLabel,
      streamSceneId: mission.streamSceneId || '',
      streamSceneLabel
    },
    preflight: {
      label: preflightLabel,
      detail: preflightDetail,
      summary: preflightSummary,
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

function buildValidateCommand(artifactFileName, mission, route) {
  const routeId = extractRobotRouteIdFromUrl(mission.routeUrl);
  const routeFileName = routeId ? `${routeId}.json` : 'none';
  return [
    `# validate input: /absolute/path/to/${artifactFileName}`,
    `# preflight: Mission Ready`,
    `# detail: ${mission.world?.assetLabel || route.world?.assetLabel || mission.fragmentLabel || mission.fragmentId || 'none'} / frame ${mission.world?.frameId || route.frameId || 'none'}`,
    `# target: ${mission.fragmentId || 'none'} / ${mission.world?.assetLabel || route.world?.assetLabel || 'none'} / ${mission.world?.frameId || route.frameId || 'none'}`,
    `# zone: ${mission.zoneMapUrl || route.world?.zoneMapUrl || 'none'}`,
    `# launch: ${mission.launchUrl || 'none'}`,
    `# route file: ${routeFileName}`,
    'npm run validate:robot-bundle -- \\',
    `  --bundle ${quoteShellValue(`/absolute/path/to/${artifactFileName}`)}`
  ].join('\n');
}

function buildReleaseCommand(artifactFileName, mission, route) {
  return [
    `# release input: /absolute/path/to/${artifactFileName}`,
    `# preflight: Mission Ready`,
    `# detail: ${mission.world?.assetLabel || route.world?.assetLabel || mission.fragmentLabel || mission.fragmentId || 'none'} / frame ${mission.world?.frameId || route.frameId || 'none'}`,
    `# target: ${mission.fragmentId || 'none'} / ${mission.world?.assetLabel || route.world?.assetLabel || 'none'} / ${mission.world?.frameId || route.frameId || 'none'}`,
    `# zone: ${mission.zoneMapUrl || route.world?.zoneMapUrl || 'none'}`,
    `# launch: ${mission.launchUrl || 'none'}`,
    `# auto outputs: /absolute/path/to/${artifactFileName.replace(/\.json$/i, '.preflight.txt')} + /absolute/path/to/${artifactFileName.replace(/\.json$/i, '.publish-report.json')}`,
    'npm run release:robot-mission -- \\',
    `  --bundle ${quoteShellValue(`/absolute/path/to/${artifactFileName}`)} \\`,
    '  --force \\',
    '  --validate'
  ].join('\n');
}

function buildPublishCommand(artifactFileName, mission, route) {
  const routeId = extractRobotRouteIdFromUrl(mission.routeUrl);
  return [
    `# publish input: /absolute/path/to/${artifactFileName}`,
    `# preflight: Mission Ready`,
    `# detail: ${mission.world?.assetLabel || route.world?.assetLabel || mission.fragmentLabel || mission.fragmentId || 'none'} / frame ${mission.world?.frameId || route.frameId || 'none'}`,
    `# target: ${mission.fragmentId || 'none'} / ${mission.world?.assetLabel || route.world?.assetLabel || 'none'} / ${mission.world?.frameId || route.frameId || 'none'}`,
    `# zone: ${mission.zoneMapUrl || route.world?.zoneMapUrl || 'none'}`,
    `# launch: ${mission.launchUrl || 'none'}`,
    `# route file: ${routeId ? `${routeId}.json` : 'none'}`,
    'npm run publish:robot-mission -- \\',
    `  --bundle ${quoteShellValue(`/absolute/path/to/${artifactFileName}`)} \\`,
    '  --force \\',
    '  --validate'
  ].join('\n');
}

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  if (!hasNonEmptyString(args.mission)) {
    throw new Error('--mission は必須です。');
  }

  const publicRoot = toAbsolutePath(args['public-root'] ?? defaultPublicRoot);
  const missionSource = args.mission.trim();
  const mission = await readJsonInput(missionSource);

  if (!isRecord(mission)) {
    throw new Error('mission JSON が不正です。');
  }

  const routeResolved = await resolveJsonSource({
    explicitSource: args.route,
    fallbackUrl: mission.routeUrl,
    publicRoot,
    label: 'route'
  });
  if (!isRecord(routeResolved.json)) {
    throw new Error('route JSON を解決できませんでした。');
  }

  const zonesResolved = await resolveJsonSource({
    explicitSource: args.zones,
    fallbackUrl: mission.zoneMapUrl,
    publicRoot,
    label: 'zones'
  });

  const fragmentId = hasNonEmptyString(mission.fragmentId)
    ? mission.fragmentId.trim()
    : hasNonEmptyString(routeResolved.json.fragmentId)
      ? routeResolved.json.fragmentId.trim()
      : '';
  if (!fragmentId) {
    throw new Error('fragmentId を解決できませんでした。');
  }

  const fragmentConfig = resolveDreamwalkerConfig(fragmentId);
  const route = routeResolved.json;
  const routeId = extractRobotRouteIdFromUrl(mission.routeUrl);
  const routeFileName = routeId ? `${routeId}.json` : 'none';
  const cameraPresetLabel =
    fragmentConfig.cameraPresets.find((preset) => preset.id === mission.cameraPresetId)?.label ??
    mission.cameraPresetId ??
    '';
  const robotCameraLabel =
    fragmentConfig.robotics.cameras.find((camera) => camera.id === mission.robotCameraId)?.label ??
    mission.robotCameraId ??
    '';
  const streamSceneLabel =
    fragmentConfig.streamScenes.find((scene) => scene.id === mission.streamSceneId)?.title ??
    fragmentConfig.streamScenes.find((scene) => scene.id === mission.streamSceneId)?.label ??
    mission.streamSceneId ??
    '';

  const preflightWarnings = [];
  if (
    hasNonEmptyString(mission.fragmentId) &&
    hasNonEmptyString(route.fragmentId) &&
    mission.fragmentId !== route.fragmentId
  ) {
    preflightWarnings.push(
      `mission fragment=${mission.fragmentId} / route fragment=${route.fragmentId}`
    );
  }
  if (
    hasNonEmptyString(mission.zoneMapUrl) &&
    hasNonEmptyString(route.world?.zoneMapUrl) &&
    mission.zoneMapUrl !== route.world.zoneMapUrl
  ) {
    preflightWarnings.push(
      `mission zone=${mission.zoneMapUrl} / route zone=${route.world.zoneMapUrl}`
    );
  }
  if (
    hasNonEmptyString(mission.world?.frameId) &&
    hasNonEmptyString(route.frameId) &&
    mission.world.frameId !== route.frameId
  ) {
    preflightWarnings.push(
      `mission frame=${mission.world.frameId} / route frame=${route.frameId}`
    );
  }
  if (
    hasNonEmptyString(mission.world?.assetLabel) &&
    hasNonEmptyString(route.world?.assetLabel) &&
    mission.world.assetLabel !== route.world.assetLabel
  ) {
    preflightWarnings.push(
      `mission world=${mission.world.assetLabel} / route world=${route.world.assetLabel}`
    );
  }

  const preflightLabel = preflightWarnings.length ? 'Mission Warning' : 'Mission Ready';
  const preflightDetail = preflightWarnings.length
    ? preflightWarnings.join(' ; ')
    : `${mission.world?.assetLabel || route.world?.assetLabel || mission.fragmentLabel || fragmentId} / frame ${mission.world?.frameId || route.frameId || 'none'}`;
  const preflightSummary = buildPreflightSummary({
    healthLabel: preflightLabel,
    healthDetail: preflightDetail,
    mission,
    route,
    routeId,
    routeFileName,
    cameraPresetLabel,
    robotCameraLabel,
    streamSceneLabel
  });

  const outputPath = hasNonEmptyString(args.output)
    ? toAbsolutePath(args.output)
    : path.resolve(process.cwd(), buildArtifactFileName(mission));
  const artifactFileName = path.basename(outputPath);
  const baseName = artifactFileName.replace(/\.json$/i, '');
  const packLabel = hasNonEmptyString(args.label)
    ? args.label.trim()
    : mission.label || route.label || baseName;
  const publishReport = buildPublishReport({
    publicRoot,
    missionSource,
    routeSource: routeResolved.source,
    zoneSource: zonesResolved.source,
    mission,
    route,
    routeId,
    routeFileName,
    cameraPresetLabel,
    robotCameraLabel,
    streamSceneLabel,
    preflightLabel,
    preflightDetail,
    preflightSummary
  });
  const draftBundle = {
    fragmentId,
    mission,
    route,
    ...(isRecord(zonesResolved.json) ? { zones: zonesResolved.json } : {})
  };
  const publishedMissionFileName = buildPublishedMissionFileName(mission);
  const artifactPack = {
    version: 1,
    protocol: robotMissionArtifactPackProtocolId,
    label: packLabel,
    files: [
      {
        kind: 'draft-bundle',
        fileName: `${baseName}.draft-bundle.json`,
        content: JSON.stringify(draftBundle, null, 2)
      },
      {
        kind: 'mission',
        fileName: publishedMissionFileName,
        content: JSON.stringify(mission, null, 2)
      },
      {
        kind: 'published-preview',
        fileName: publishedMissionFileName,
        content: JSON.stringify(mission, null, 2)
      },
      {
        kind: 'launch-url',
        fileName: `${baseName}.launch-url.txt`,
        content: mission.launchUrl || ''
      },
      {
        kind: 'preflight-summary',
        fileName: `${baseName}.preflight.txt`,
        content: preflightSummary
      },
      {
        kind: 'publish-report',
        fileName: `${baseName}.publish-report.json`,
        content: JSON.stringify(publishReport, null, 2)
      },
      {
        kind: 'validate-command',
        fileName: `${baseName}.validate-command.txt`,
        content: buildValidateCommand(artifactFileName, mission, route)
      },
      {
        kind: 'release-command',
        fileName: `${baseName}.release-command.txt`,
        content: buildReleaseCommand(artifactFileName, mission, route)
      },
      {
        kind: 'publish-command',
        fileName: `${baseName}.publish-command.txt`,
        content: buildPublishCommand(artifactFileName, mission, route)
      }
    ]
  };

  await writeJson(outputPath, artifactPack);
  console.log(`Artifact pack created: ${outputPath}`);
  console.log(`- label: ${packLabel}`);
  console.log(`- fragment: ${fragmentId}`);
  console.log(`- mission: ${mission.id || mission.label || 'none'}`);
  console.log(`- route: ${route.label || 'none'} / ${route.route?.length ?? 0} node(s)`);
  console.log(`- zones: ${isRecord(zonesResolved.json) ? (zonesResolved.json.zones?.length ?? 0) : 0}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`ERR: ${message}`);
  process.exitCode = 1;
});
