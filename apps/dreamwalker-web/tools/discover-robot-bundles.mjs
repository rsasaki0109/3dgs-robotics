import process from 'node:process';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultDiscoveryRoots, discoverRobotBundles } from './robot-bundle-discovery.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');

function printUsage() {
  console.log(`Dynamic Map Viewer robot bundle discovery

Usage:
  node ./tools/discover-robot-bundles.mjs
  node ./tools/discover-robot-bundles.mjs --root /absolute/path/to/search --validate

Options:
  --root <dir>       add search root. repeatable.
  --limit <n>        max files to list per category. default: 20
  --validate         run validate:robot-bundle for discovered artifact packs.
  --help             show this message.
`);
}

function parseArgs(argv) {
  const args = {
    roots: [],
    validate: false,
    limit: 20
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help') {
      args.help = true;
      continue;
    }

    if (token === '--validate') {
      args.validate = true;
      continue;
    }

    if (token === '--root' || token === '--limit') {
      const nextToken = argv[index + 1];
      if (!nextToken || nextToken.startsWith('--')) {
        throw new Error(`値が必要です: ${token}`);
      }

      if (token === '--root') {
        args.roots.push(path.resolve(nextToken));
      } else {
        const parsedLimit = Number.parseInt(nextToken, 10);
        if (!Number.isFinite(parsedLimit) || parsedLimit <= 0) {
          throw new Error(`--limit は正の整数が必要です: ${nextToken}`);
        }
        args.limit = parsedLimit;
      }

      index += 1;
    }
  }

  return args;
}

function printSection(title, values, limit) {
  console.log(`\n${title}: ${values.length}`);
  for (const value of values.slice(0, limit)) {
    console.log(`- ${typeof value === 'string' ? value : value.path}`);
  }
  if (values.length > limit) {
    console.log(`- ... ${values.length - limit} more`);
  }
}

function runValidator(filePath) {
  const result = spawnSync(
    process.execPath,
    [path.join(__dirname, 'validate-robot-bundle.mjs'), '--bundle', filePath],
    {
      cwd: appRoot,
      encoding: 'utf8'
    }
  );

  return {
    status: result.status ?? 1,
    stdout: result.stdout || '',
    stderr: result.stderr || ''
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }

  const roots = Array.from(new Set(args.roots.length ? args.roots : defaultDiscoveryRoots()));
  const found = await discoverRobotBundles(roots);

  console.log(`Search roots (${roots.length}):`);
  for (const root of roots) {
    console.log(`- ${root}`);
  }

  printSection('Artifact Packs', found.artifactPacks, args.limit);
  printSection('Mission Manifests', found.missionManifests, args.limit);
  printSection('Splats', found.splats, args.limit);
  printSection('Colliders', found.colliders, args.limit);

  if (args.validate && found.artifactPacks.length) {
    console.log('\nArtifact Pack Validation:');
    for (const artifactPack of found.artifactPacks.slice(0, args.limit)) {
      const result = runValidator(artifactPack.path);
      const summaryLine =
        result.stdout
          .split('\n')
          .find((line) => line.startsWith('Summary:')) || 'Summary: unavailable';
      console.log(`- ${artifactPack.path}`);
      console.log(`  ${summaryLine}`);
      if (result.status !== 0 && result.stderr) {
        console.log(`  stderr: ${result.stderr.trim()}`);
      }
    }
  }

  if (
    found.artifactPacks.length === 0 &&
    found.splats.length > 0 &&
    found.colliders.length > 0
  ) {
    console.log('\nHint: splat と collider は見つかっています。artifact-pack が無いので Mission Export か publish 導線へまだ乗っていません。');
  }
}

try {
  await main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
