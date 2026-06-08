import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { dreamwalkerConfig, resolveDreamwalkerConfig } from '../src/app-config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const defaultPublicRoot = path.join(appRoot, 'public');
const defaultManifestPath = path.join(defaultPublicRoot, 'manifests', 'dreamwalker-live.assets.json');
const defaultCatalogPath = path.join(defaultPublicRoot, 'studio-bundles', 'index.json');

function printUsage() {
  console.log(`Dynamic Map Viewer Marble fragment staging

Usage:
  node ./tools/stage-marble-fragment.mjs --fragment residency --source-dir /path/to/export
  node ./tools/stage-marble-fragment.mjs --fragment residency --splat https://.../main.sog --collider https://.../collider.glb

Options:
  --fragment <id>          Dynamic Map Viewer fragment id. required.
  --source-dir <dir>       source directory that contains splat and collider files.
  --splat <file|url>       source splat file or http(s) url. if omitted, auto-detect from --source-dir.
  --collider <file|url>    source collider file or http(s) url. if omitted, auto-detect from --source-dir.
  --label <text>           asset label written into the manifest.
  --world-note <text>      world note written into the manifest.
  --bundle-id <id>         studio bundle id. default: <fragment>-stage
  --bundle-label <text>    studio bundle label.
  --bundle-description <text>
                           catalog description for the generated bundle.
  --bundle-path <file>     custom bundle file path. defaults under public/studio-bundles/.
  --manifest-path <file>   custom asset manifest path.
  --catalog-path <file>    custom studio bundle catalog path.
  --public-root <dir>      custom public root. default: apps/dreamwalker-web/public
  --dry-run                print planned outputs without writing files.
  --force                  overwrite existing staged files.
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

function deriveOutputUrl(expectedUrl, fallbackDirectory, fallbackStem, sourcePath) {
  const sourceExtension = path.extname(sourcePath);

  if (expectedUrl) {
    const expectedDirectory = path.posix.dirname(expectedUrl);
    const expectedExtension = path.posix.extname(expectedUrl);
    const expectedStem = path.posix.basename(expectedUrl, expectedExtension);
    return path.posix.join(expectedDirectory, `${expectedStem}${sourceExtension || expectedExtension}`);
  }

  return path.posix.join(fallbackDirectory, `${fallbackStem}${sourceExtension}`);
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
    return structuredClone(fallbackValue);
  }

  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function writeJson(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function listSourceCandidates(sourceDir) {
  const entries = await readdir(sourceDir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(sourceDir, entry.name));
}

function getSplatScore(filePath) {
  const fileName = path.basename(filePath).toLowerCase();
  const extension = path.extname(fileName);
  const extensionWeights = new Map([
    ['.sog', 100],
    ['.ksplat', 85],
    ['.spz', 70],
    ['.splat', 60],
    ['.ply', 50]
  ]);

  let score = extensionWeights.get(extension) ?? -1000;
  if (fileName.includes('main')) score += 20;
  if (fileName.includes('high')) score += 8;
  if (fileName.includes('preview') || fileName.includes('thumb')) score -= 20;
  if (fileName.includes('low') || fileName.includes('mobile')) score -= 12;
  return score;
}

function getColliderScore(filePath) {
  const fileName = path.basename(filePath).toLowerCase();
  const extension = path.extname(fileName);
  const extensionWeights = new Map([
    ['.glb', 100],
    ['.gltf', 80]
  ]);

  let score = extensionWeights.get(extension) ?? -1000;
  if (fileName.includes('collider')) score += 30;
  if (fileName.includes('collision')) score += 25;
  if (fileName.includes('proxy')) score += 12;
  if (fileName.includes('walk')) score += 8;
  if (fileName.includes('low')) score -= 5;
  return score;
}

async function detectBestCandidate(sourceDir, kind) {
  const candidates = await listSourceCandidates(sourceDir);
  const scored = await Promise.all(
    candidates.map(async (candidatePath) => {
      const fileStat = await stat(candidatePath);
      const score = kind === 'splat'
        ? getSplatScore(candidatePath)
        : getColliderScore(candidatePath);

      return {
        candidatePath,
        score,
        size: fileStat.size
      };
    })
  );

  const best = scored
    .filter((entry) => entry.score > -1000)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }

      if (right.size !== left.size) {
        return right.size - left.size;
      }

      return left.candidatePath.localeCompare(right.candidatePath);
    })[0];

  return best?.candidatePath ?? '';
}

async function resolveSourcePath({
  explicitPath,
  sourceDir,
  kind
}) {
  if (explicitPath) {
    return toAbsolutePath(explicitPath, sourceDir || process.cwd());
  }

  if (!sourceDir) {
    return '';
  }

  return detectBestCandidate(sourceDir, kind);
}

async function resolveRemoteSource(sourceUrl, kind) {
  const response = await fetch(sourceUrl);

  if (!response.ok) {
    throw new Error(`${kind} source の download に失敗しました: ${response.status} ${response.statusText}`);
  }

  const parsedUrl = new URL(sourceUrl);
  const urlExtension = path.extname(parsedUrl.pathname);
  const fallbackExtension = kind === 'collider' ? '.glb' : '.sog';
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'dreamwalker-stage-'));
  const tempPath = path.join(tempRoot, `${kind}${urlExtension || fallbackExtension}`);
  const body = Buffer.from(await response.arrayBuffer());

  await writeFile(tempPath, body);

  return {
    localPath: tempPath,
    displayPath: sourceUrl,
    cleanupPath: tempRoot
  };
}

async function resolveSourceInput({
  explicitPath,
  sourceDir,
  kind
}) {
  if (explicitPath && isRemoteUrl(explicitPath)) {
    return resolveRemoteSource(explicitPath, kind);
  }

  const localPath = await resolveSourcePath({
    explicitPath,
    sourceDir,
    kind
  });

  if (!localPath) {
    return null;
  }

  return {
    localPath,
    displayPath: localPath,
    cleanupPath: ''
  };
}

async function copyIfNeeded(sourcePath, targetPath, force) {
  await mkdir(path.dirname(targetPath), { recursive: true });

  if (path.resolve(sourcePath) === path.resolve(targetPath)) {
    return 'unchanged';
  }

  const targetExists = await pathExists(targetPath);
  if (targetExists && !force) {
    throw new Error(`出力先が既に存在します: ${targetPath}\n--force で上書きできます。`);
  }

  await copyFile(sourcePath, targetPath);
  return targetExists ? 'overwritten' : 'created';
}

function buildDefaultWorldNote(fragmentLabel, splatSourceLabel, colliderSourceLabel) {
  const stagedAt = new Date().toISOString().slice(0, 10);
  return [
    `${fragmentLabel} fragment 用に実 Marble asset を staging 済み。`,
    `Splat: ${path.basename(splatSourceLabel)}`,
    `Collider: ${path.basename(colliderSourceLabel)}`,
    `Staged at: ${stagedAt}`
  ].join(' ');
}

function upsertCatalogEntry(catalog, nextEntry) {
  const nextBundles = Array.isArray(catalog.bundles) ? [...catalog.bundles] : [];
  const existingIndex = nextBundles.findIndex((entry) =>
    entry.id === nextEntry.id || entry.url === nextEntry.url
  );

  if (existingIndex >= 0) {
    nextBundles[existingIndex] = {
      ...nextBundles[existingIndex],
      ...nextEntry
    };
  } else {
    nextBundles.push(nextEntry);
  }

  return {
    ...catalog,
    version: catalog.version ?? 1,
    label: catalog.label ?? 'Dynamic Map Viewer Public Bundle Catalog',
    note: catalog.note ?? 'repo 同梱の studio bundle 一覧。',
    bundles: nextBundles
  };
}

async function main() {
  const cleanupPaths = [];
  const args = parseArgs(process.argv.slice(2));

  try {
    if (args.help) {
      printUsage();
      return;
    }

    const fragmentId = String(args.fragment ?? '').trim();
    if (!fragmentId) {
      throw new Error('--fragment は必須です。');
    }

    if (!Object.hasOwn(dreamwalkerConfig.fragments, fragmentId)) {
      throw new Error(`未知の fragment です: ${fragmentId}`);
    }

    const worldConfig = resolveDreamwalkerConfig(fragmentId);
    const sourceDir = args['source-dir']
      ? toAbsolutePath(args['source-dir'])
      : '';
    const publicRoot = toAbsolutePath(args['public-root'] ?? defaultPublicRoot);
    const manifestPath = toAbsolutePath(args['manifest-path'] ?? defaultManifestPath);
    const catalogPath = toAbsolutePath(args['catalog-path'] ?? defaultCatalogPath);
    const bundleId = sanitizeId(args['bundle-id'] ?? `${fragmentId}-stage`, `${fragmentId}-stage`);
    const bundlePath = toAbsolutePath(
      args['bundle-path'] ?? path.join(publicRoot, 'studio-bundles', `${bundleId}.json`)
    );

    const splatSource = await resolveSourceInput({
      explicitPath: args.splat,
      sourceDir,
      kind: 'splat'
    });
    const colliderSource = await resolveSourceInput({
      explicitPath: args.collider,
      sourceDir,
      kind: 'collider'
    });

    if (!splatSource) {
      throw new Error('splat source を解決できませんでした。--splat か --source-dir を指定してください。');
    }

    if (!colliderSource) {
      throw new Error('collider source を解決できませんでした。--collider か --source-dir を指定してください。');
    }

    if (splatSource.cleanupPath) {
      cleanupPaths.push(splatSource.cleanupPath);
    }

    if (colliderSource.cleanupPath) {
      cleanupPaths.push(colliderSource.cleanupPath);
    }

    if (!(await pathExists(splatSource.localPath))) {
      throw new Error(`splat source が見つかりません: ${splatSource.displayPath}`);
    }

    if (!(await pathExists(colliderSource.localPath))) {
      throw new Error(`collider source が見つかりません: ${colliderSource.displayPath}`);
    }

    const assetBundleDefaults = worldConfig.assetBundle ?? {};
    const splatOutputUrl = deriveOutputUrl(
      assetBundleDefaults.expectedSplatUrl,
      '/splats',
      `${fragmentId}-main`,
      splatSource.localPath
    );
    const colliderOutputUrl = deriveOutputUrl(
      assetBundleDefaults.expectedColliderMeshUrl,
      '/colliders',
      `${fragmentId}-main-collider`,
      colliderSource.localPath
    );
    const splatOutputPath = path.join(publicRoot, splatOutputUrl.replace(/^\/+/, ''));
    const colliderOutputPath = path.join(publicRoot, colliderOutputUrl.replace(/^\/+/, ''));

    const assetLabel = String(args.label ?? `${worldConfig.fragmentLabel} Marble`).trim();
    const worldNote = String(
      args['world-note'] ?? buildDefaultWorldNote(
        worldConfig.fragmentLabel,
        splatSource.displayPath,
        colliderSource.displayPath
      )
    ).trim();
    const bundleLabel = String(args['bundle-label'] ?? `${worldConfig.fragmentLabel} Stage Bundle`).trim();
    const bundleDescription = String(
      args['bundle-description'] ?? `${worldConfig.fragmentLabel} Marble stage / auto-staged bundle`
    ).trim();
    const bundleUrl = buildPublicUrl(publicRoot, bundlePath);

    const defaultScene = worldConfig.streamScenes?.[0] ?? null;
    const defaultCameraPresetId =
      defaultScene?.presetId ??
      worldConfig.homePresetId ??
      worldConfig.cameraPresets?.[0]?.id ??
      '';
    const defaultOverlayPresetId = dreamwalkerConfig.overlayPresets?.[1]?.id ??
      dreamwalkerConfig.overlayPresets?.[0]?.id ??
      '';
    const defaultFilterId = dreamwalkerConfig.dreamFilters?.[0]?.id ?? '';
    const defaultRatioId = dreamwalkerConfig.photoRatios?.[0]?.id ?? '';

    const manifest = await readJsonOrFallback(manifestPath, {
      version: 1,
      label: 'Local Dynamic Map Asset Manifest',
      note: 'stage-marble-fragment.mjs により生成される asset manifest。',
      fragments: {}
    });
    const nextManifest = {
      ...manifest,
      version: manifest.version ?? 1,
      fragments: {
        ...(manifest.fragments ?? {}),
        [fragmentId]: {
          ...(manifest.fragments?.[fragmentId] ?? {}),
          label: assetLabel,
          splatUrl: splatOutputUrl,
          colliderMeshUrl: colliderOutputUrl,
          expectedSplatUrl: assetBundleDefaults.expectedSplatUrl ?? '',
          expectedColliderMeshUrl: assetBundleDefaults.expectedColliderMeshUrl ?? '',
          worldNote
        }
      }
    };

    const nextBundle = {
      version: 1,
      label: bundleLabel,
      note: `Auto-generated by stage-marble-fragment.mjs for ${worldConfig.fragmentLabel}.`,
      assetWorkspace: {
        version: 1,
        label: `${bundleLabel} Assets`,
        fragments: {
          [fragmentId]: {
            label: assetLabel,
            splatUrl: splatOutputUrl,
            colliderMeshUrl: colliderOutputUrl,
            worldNote
          }
        }
      },
      state: {
        fragmentId,
        streamSceneId: defaultScene?.id ?? '',
        overlayPresetId: defaultOverlayPresetId,
        filterId: defaultFilterId,
        ratioId: defaultRatioId,
        cameraPresetId: defaultCameraPresetId
      }
    };

    const catalog = await readJsonOrFallback(catalogPath, {
      version: 1,
      label: 'Dynamic Map Viewer Public Bundle Catalog',
      note: 'stage-marble-fragment.mjs により生成される public studio bundle catalog。',
      bundles: []
    });
    const nextCatalog = upsertCatalogEntry(catalog, {
      id: bundleId,
      label: bundleLabel,
      url: bundleUrl,
      description: bundleDescription,
      fragmentId,
      accent: worldConfig.overlayBranding?.accent ?? '#85e3e1'
    });

    const summary = [
      `fragment: ${fragmentId}`,
      `splat source: ${splatSource.displayPath}`,
      `collider source: ${colliderSource.displayPath}`,
      `splat output: ${splatOutputUrl}`,
      `collider output: ${colliderOutputUrl}`,
      `manifest: ${manifestPath}`,
      `bundle: ${bundlePath}`,
      `catalog: ${catalogPath}`,
      `launch url: /?studioBundle=${bundleUrl}`
    ];

    if (args['dry-run']) {
      console.log('Dry run only. No files were written.');
      summary.forEach((line) => console.log(`- ${line}`));
      return;
    }

    const splatCopyStatus = await copyIfNeeded(
      splatSource.localPath,
      splatOutputPath,
      Boolean(args.force)
    );
    const colliderCopyStatus = await copyIfNeeded(
      colliderSource.localPath,
      colliderOutputPath,
      Boolean(args.force)
    );
    await writeJson(manifestPath, nextManifest);
    await writeJson(bundlePath, nextBundle);
    await writeJson(catalogPath, nextCatalog);

    console.log(`Staged ${worldConfig.fragmentLabel} fragment.`);
    console.log(`- splat ${splatCopyStatus}: ${splatOutputUrl}`);
    console.log(`- collider ${colliderCopyStatus}: ${colliderOutputUrl}`);
    console.log(`- manifest updated: ${manifestPath}`);
    console.log(`- bundle updated: ${bundlePath}`);
    console.log(`- catalog updated: ${catalogPath}`);
    console.log(`- launch: /?studioBundle=${bundleUrl}`);

    const splatExtension = path.extname(splatSource.localPath).toLowerCase();
    if (splatExtension !== '.sog') {
      console.log(`WARN: current web pipeline is tuned for .sog. staged extension: ${splatExtension}`);
    }

    const colliderExtension = path.extname(colliderSource.localPath).toLowerCase();
    if (!['.glb', '.gltf'].includes(colliderExtension)) {
      console.log(`WARN: collider extension is unusual for browser staging: ${colliderExtension}`);
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
