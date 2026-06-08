import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { dreamwalkerConfig, resolveDreamwalkerConfig } from '../src/app-config.js';
import { buildSemanticZoneMap, serializeSemanticZoneMap } from '../src/semantic-zones.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const defaultPublicRoot = path.join(appRoot, 'public');

function printUsage() {
  console.log(`Dynamic Map Viewer robot zone staging

Usage:
  node ./tools/stage-robot-zones.mjs --source ./public/manifests/robotics-residency.zones.json
  node ./tools/stage-robot-zones.mjs --source https://example.com/zones.json --fragment residency

Options:
  --source <file|url>      source semantic zone JSON. required.
  --fragment <id>          target fragment id. defaults to file name or configured path.
  --zone-path <file>       custom output zone manifest path.
  --public-root <dir>      custom public root. default: apps/dreamwalker-web/public
  --frame-id <id>          override frame id.
  --resolution <value>     override resolution.
  --default-cost <value>   override default cost.
  --dry-run                print planned outputs without writing files.
  --force                  overwrite existing staged zone file.
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

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function sanitizeFragmentId(value) {
  return String(value ?? '').trim().toLowerCase();
}

function inferFragmentIdFromSource(source) {
  const fileName = path.basename(source).toLowerCase();

  for (const fragmentId of Object.keys(dreamwalkerConfig.fragments)) {
    if (fileName.includes(fragmentId)) {
      return fragmentId;
    }
  }

  return '';
}

async function resolveRemoteSource(sourceUrl) {
  const response = await fetch(sourceUrl);

  if (!response.ok) {
    throw new Error(`source zone の download に失敗しました: ${response.status} ${response.statusText}`);
  }

  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'dreamwalker-zones-'));
  const tempPath = path.join(tempRoot, 'zones.json');
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

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function writeZoneFile(targetPath, value, force) {
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

    const publicRoot = toAbsolutePath(args['public-root'] ?? defaultPublicRoot);
    const inferredFragmentId = inferFragmentIdFromSource(sourceInput.displayPath);
    const fragmentId = sanitizeFragmentId(args.fragment || inferredFragmentId);

    if (!fragmentId) {
      throw new Error('fragment を解決できませんでした。--fragment を指定してください。');
    }

    if (!Object.hasOwn(dreamwalkerConfig.fragments, fragmentId)) {
      throw new Error(`未知の fragment です: ${fragmentId}`);
    }

    const activeConfig = resolveDreamwalkerConfig(fragmentId);
    const targetZoneUrl =
      activeConfig.robotics.semanticZoneMapUrl ||
      `/manifests/robotics-${fragmentId}.zones.json`;
    const targetPath = toAbsolutePath(
      args['zone-path'] ??
        path.join(publicRoot, targetZoneUrl.replace(/^\/+/, ''))
    );

    const rawPayload = JSON.parse(await readFile(sourceInput.localPath, 'utf8'));
    const normalizedMap = buildSemanticZoneMap({
      ...rawPayload,
      frameId: hasNonEmptyString(args['frame-id']) ? args['frame-id'].trim() : rawPayload.frameId,
      resolution: hasNonEmptyString(args.resolution) ? Number(args.resolution) : rawPayload.resolution,
      defaultCost: hasNonEmptyString(args['default-cost']) ? Number(args['default-cost']) : rawPayload.defaultCost
    });
    const nextPayload = serializeSemanticZoneMap(normalizedMap);
    const targetUrl = buildPublicUrl(publicRoot, targetPath);

    const summary = [
      `source: ${sourceInput.displayPath}`,
      `fragment: ${fragmentId}`,
      `zone output: ${targetUrl}`,
      `frame: ${nextPayload.frameId}`,
      `resolution: ${nextPayload.resolution}`,
      `zones: ${nextPayload.zones.length}`
    ];

    if (args['dry-run']) {
      console.log('Dry run only. No files were written.');
      summary.forEach((line) => console.log(`- ${line}`));
      return;
    }

    const writeStatus = await writeZoneFile(targetPath, nextPayload, Boolean(args.force));

    console.log(`Staged robot zones.`);
    console.log(`- zone file ${writeStatus}: ${targetUrl}`);
    console.log(`- fragment: ${activeConfig.fragmentLabel}`);
    console.log(`- frame: ${nextPayload.frameId}`);
    console.log(`- zones: ${nextPayload.zones.length}`);
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
