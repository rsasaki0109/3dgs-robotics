import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const defaultPublicRoot = path.join(appRoot, 'public');

const robotMissionProtocolId = 'dreamwalker-robot-mission/v1';
const robotRouteProtocolId = 'dreamwalker-robot-route/v1';
const robotMissionArtifactPackProtocolId = 'dreamwalker-robot-mission-artifact-pack/v1';
const robotMissionPublishReportProtocolId = 'dreamwalker-robot-mission-publish-report/v1';

function printUsage() {
  console.log(`Dynamic Map Viewer robot bundle validation

Usage:
  node ./tools/validate-robot-bundle.mjs --bundle ./downloads/dreamwalker-live-residency.artifact-pack.json
  node ./tools/validate-robot-bundle.mjs --bundle ./downloads/dreamwalker-live-residency-draft-bundle.json

Options:
  --bundle <file|url>         robot mission draft bundle JSON or artifact pack JSON.
  --public-root <dir>         local public root for checking local route/zone/mission URLs.
  --help                      show this message.
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

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
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

async function fileExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

function toPublicFilePath(publicRoot, assetUrl) {
  if (!hasNonEmptyString(assetUrl) || !assetUrl.trim().startsWith('/')) {
    return '';
  }

  return path.join(publicRoot, assetUrl.trim().replace(/^\/+/, ''));
}

function sanitizeId(value, fallback = '') {
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

function normalizeDraftBundle(bundleLike) {
  const bundle = isRecord(bundleLike) ? bundleLike : {};
  const mission = isRecord(bundle.mission) ? bundle.mission : bundle;
  const route = isRecord(bundle.route)
    ? bundle.route
    : isRecord(bundle.robotRoute)
      ? bundle.robotRoute
      : null;

  if (!route) {
    throw new Error('draft bundle には route object が必要です。');
  }

  const fragmentId =
    (hasNonEmptyString(bundle.fragmentId) && bundle.fragmentId.trim()) ||
    (hasNonEmptyString(mission.fragmentId) && mission.fragmentId.trim()) ||
    (hasNonEmptyString(route.fragmentId) && route.fragmentId.trim()) ||
    '';

  if (!fragmentId) {
    throw new Error('draft bundle には fragmentId が必要です。');
  }

  return {
    bundle,
    mission,
    route,
    fragmentId
  };
}

function createReporter() {
  const issues = [];

  return {
    ok(scope, detail) {
      issues.push({ level: 'OK', scope, detail });
    },
    warn(scope, detail) {
      issues.push({ level: 'WARN', scope, detail });
    },
    error(scope, detail) {
      issues.push({ level: 'ERROR', scope, detail });
    },
    flush() {
      for (const issue of issues) {
        console.log(`${issue.level.padEnd(5, ' ')} ${issue.scope.padEnd(28, ' ')} ${issue.detail}`);
      }

      const errorCount = issues.filter((issue) => issue.level === 'ERROR').length;
      const warningCount = issues.filter((issue) => issue.level === 'WARN').length;
      console.log(`\nSummary: ${errorCount} error(s), ${warningCount} warning(s)`);
      return { errorCount, warningCount };
    }
  };
}

function readArtifactEntry(files, kind) {
  return files.find(
    (entry) =>
      isRecord(entry) &&
      hasNonEmptyString(entry.kind) &&
      entry.kind.trim() === kind
  ) ?? null;
}

function parseArtifactEntryJson(entry, scope, reporter) {
  if (!entry) {
    return null;
  }

  try {
    return typeof entry.content === 'string'
      ? JSON.parse(entry.content)
      : entry.content;
  } catch (error) {
    reporter.error(scope, `${entry.kind} の JSON parse に失敗しました: ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}

async function validateLocalPublicUrl(assetUrl, publicRoot, scope, reporter) {
  if (!hasNonEmptyString(assetUrl) || !assetUrl.trim().startsWith('/')) {
    reporter.warn(scope, `local public path ではありません: ${assetUrl || 'none'}`);
    return;
  }

  const filePath = toPublicFilePath(publicRoot, assetUrl);
  if (!filePath) {
    reporter.warn(scope, `local public path に解決できません: ${assetUrl}`);
    return;
  }

  if (await fileExists(filePath)) {
    reporter.ok(scope, `${assetUrl} -> ${filePath}`);
  } else {
    reporter.warn(scope, `public root に見つかりません: ${assetUrl}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    printUsage();
    return;
  }

  if (!hasNonEmptyString(args.bundle)) {
    throw new Error('--bundle は必須です。');
  }

  const reporter = createReporter();
  const publicRoot = toAbsolutePath(args['public-root'] ?? defaultPublicRoot);
  const source = await readJsonInput(args.bundle);

  let draftBundleSource = source;
  let artifactPack = null;

  if (
    isRecord(source) &&
    hasNonEmptyString(source.protocol) &&
    source.protocol.trim() === robotMissionArtifactPackProtocolId
  ) {
    artifactPack = source;
    const files = Array.isArray(source.files) ? source.files : [];
    reporter.ok(
      'artifact-pack',
      `${source.label || 'unnamed'} / ${files.length} file(s)`
    );

    const draftBundleEntry = readArtifactEntry(files, 'draft-bundle');
    if (!draftBundleEntry) {
      reporter.error('artifact-pack', 'draft-bundle entry がありません。');
      const { errorCount } = reporter.flush();
      process.exit(errorCount > 0 ? 1 : 0);
    }

    reporter.ok(
      'artifact-pack:draft',
      hasNonEmptyString(draftBundleEntry.fileName) ? draftBundleEntry.fileName.trim() : 'draft-bundle entry found'
    );
    draftBundleSource = parseArtifactEntryJson(draftBundleEntry, 'artifact-pack:draft', reporter);

    const missionEntry = readArtifactEntry(files, 'mission');
    const previewEntry = readArtifactEntry(files, 'published-preview');
    const launchEntry = readArtifactEntry(files, 'launch-url');
    const preflightEntry = readArtifactEntry(files, 'preflight-summary');
    const reportEntry = readArtifactEntry(files, 'publish-report');
    const validateCommandEntry = readArtifactEntry(files, 'validate-command');
    const releaseCommandEntry = readArtifactEntry(files, 'release-command');
    const commandEntry = readArtifactEntry(files, 'publish-command');

    for (const [kind, entry] of [
      ['mission', missionEntry],
      ['published-preview', previewEntry],
      ['launch-url', launchEntry],
      ['preflight-summary', preflightEntry],
      ['publish-report', reportEntry],
      ['validate-command', validateCommandEntry],
      ['release-command', releaseCommandEntry],
      ['publish-command', commandEntry]
    ]) {
      if (entry) {
        reporter.ok(
          `artifact-pack:${kind}`,
          hasNonEmptyString(entry.fileName) ? entry.fileName.trim() : `${kind} entry found`
        );
      } else {
        reporter.warn(`artifact-pack:${kind}`, `${kind} entry がありません。`);
      }
    }

    if (preflightEntry && typeof preflightEntry.content === 'string') {
      if (
        preflightEntry.content.includes('status:') &&
        preflightEntry.content.includes('launchUrl:')
      ) {
        reporter.ok('artifact-pack:preflight', 'embedded preflight summary を確認しました。');
      } else {
        reporter.warn('artifact-pack:preflight', 'embedded preflight summary の key が不足しています。');
      }
    }

    if (reportEntry) {
      const reportJson = parseArtifactEntryJson(reportEntry, 'artifact-pack:report', reporter);
      if (isRecord(reportJson)) {
        if (reportJson.protocol === robotMissionPublishReportProtocolId) {
          reporter.ok('artifact-pack:report', reportJson.protocol);
        } else {
          reporter.error('artifact-pack:report', `protocol が不正です: ${reportJson.protocol ?? 'none'}`);
        }
      }
    }

    if (validateCommandEntry && typeof validateCommandEntry.content === 'string') {
      if (
        validateCommandEntry.content.includes('validate:robot-bundle') &&
        validateCommandEntry.content.includes('.artifact-pack.json')
      ) {
        reporter.ok('artifact-pack:validate', 'artifact-pack native validate command を確認しました。');
      } else {
        reporter.warn('artifact-pack:validate', 'validate command が artifact-pack 前提になっていません。');
      }
    }

    if (releaseCommandEntry && typeof releaseCommandEntry.content === 'string') {
      if (
        (
          releaseCommandEntry.content.includes('release:robot-mission') ||
          (
            releaseCommandEntry.content.includes('validate:robot-bundle') &&
            releaseCommandEntry.content.includes('publish:robot-mission')
          )
        ) &&
        releaseCommandEntry.content.includes('.artifact-pack.json')
      ) {
        reporter.ok('artifact-pack:release', 'artifact-pack native release command を確認しました。');
      } else {
        reporter.warn('artifact-pack:release', 'release command が validate/publish の連結前提になっていません。');
      }
    }

    if (commandEntry && typeof commandEntry.content === 'string') {
      if (
        commandEntry.content.includes('publish:robot-mission') &&
        commandEntry.content.includes('.artifact-pack.json')
      ) {
        reporter.ok('artifact-pack:command', 'artifact-pack native publish command を確認しました。');
      } else {
        reporter.warn('artifact-pack:command', 'publish command が artifact-pack 前提になっていません。');
      }
    }
  } else {
    reporter.warn('bundle', 'artifact-pack ではなく draft bundle として検査します。');
  }

  if (!isRecord(draftBundleSource)) {
    reporter.error('bundle', 'draft bundle を解決できませんでした。');
    const { errorCount } = reporter.flush();
    process.exit(errorCount > 0 ? 1 : 0);
  }

  const { mission, route, fragmentId } = normalizeDraftBundle(draftBundleSource);
  reporter.ok('bundle', `fragment ${fragmentId}`);

  if (mission.protocol === robotMissionProtocolId) {
    reporter.ok('bundle:mission', `${mission.id || 'no-id'} / ${mission.label || 'no-label'}`);
  } else {
    reporter.warn('bundle:mission', `protocol が未設定か不一致です: ${mission.protocol ?? 'none'}`);
  }

  if (route.protocol === robotRouteProtocolId) {
    reporter.ok('bundle:route', `${route.label || 'no-label'} / ${route.route?.length ?? 0} node(s)`);
  } else {
    reporter.warn('bundle:route', `protocol が未設定か不一致です: ${route.protocol ?? 'none'}`);
  }

  const routeId = extractRobotRouteIdFromUrl(mission.routeUrl);
  if (routeId) {
    reporter.ok('bundle:route-url', `${mission.routeUrl} / routeId ${routeId}`);
  } else {
    reporter.warn('bundle:route-url', `routeUrl が不正か未設定です: ${mission.routeUrl ?? 'none'}`);
  }

  if (hasNonEmptyString(mission.launchUrl) && mission.launchUrl.includes('?robotMission=')) {
    reporter.ok('bundle:launch', mission.launchUrl);
  } else {
    reporter.warn('bundle:launch', `launchUrl が mission-native ではありません: ${mission.launchUrl ?? 'none'}`);
  }

  await validateLocalPublicUrl(mission.routeUrl, publicRoot, 'local:route', reporter);
  await validateLocalPublicUrl(mission.zoneMapUrl, publicRoot, 'local:zone', reporter);
  await validateLocalPublicUrl(mission.launchUrl?.includes('?robotMission=') ? decodeURIComponent(mission.launchUrl.split('?robotMission=')[1] ?? '') : '', publicRoot, 'local:mission', reporter);

  const { errorCount } = reporter.flush();
  process.exit(errorCount > 0 ? 1 : 0);
}

main().catch((error) => {
  console.error(`ERROR validate-robot-bundle         ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
