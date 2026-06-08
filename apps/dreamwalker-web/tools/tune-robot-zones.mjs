import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function printUsage() {
  console.log(`Dynamic Map Viewer robot zone autotune

Usage:
  node ./tools/tune-robot-zones.mjs --route ./public/robot-routes/residency-window-loop.json --zones ./public/manifests/robotics-residency.zones.json --fragment residency --force

Options:
  --route <file|url>            route JSON. required.
  --zones <file|url>            semantic zone JSON. required.
  --fragment <id>               target fragment id. required.
  --zone-path <file>            final staged zone file path.
  --public-root <dir>           custom public root.
  --corridor-padding <n>        pass through to suggest:robot-zones.
  --bounds-padding <n>          pass through to suggest:robot-zones.
  --cost <n>                    pass through to suggest:robot-zones.
  --label-prefix <text>         pass through to suggest:robot-zones.
  --include-hazard-review       generate hazard review zones.
  --hazard-cost <n>             pass through to suggest:robot-zones.
  --hazard-label-prefix <text>  pass through to suggest:robot-zones.
  --include-bounds-review       generate bounds review zones.
  --bounds-cost <n>             pass through to suggest:robot-zones.
  --bounds-label-prefix <text>  pass through to suggest:robot-zones.
  --merge-bounds                expand zone bounds with route padding.
  --frame-id <id>               pass through to stage:robot-zones.
  --resolution <value>          pass through to stage:robot-zones.
  --default-cost <value>        pass through to stage:robot-zones.
  --dry-run                     run suggest + stage in dry-run mode.
  --force                       overwrite final staged zone file.
  --validate                    run validate:studio after staging when using default public root.
  --keep-temp                   keep intermediate suggested zone JSON.
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

function pushFlag(args, key, value = true) {
  if (value) {
    args.push(`--${key}`);
  }
}

function pushOption(args, key, value) {
  if (hasNonEmptyString(value)) {
    args.push(`--${key}`, String(value).trim());
  }
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

  if (!hasNonEmptyString(args.fragment)) {
    throw new Error('--fragment は必須です。');
  }

  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'dreamwalker-zone-tune-'));
  const suggestedZonePath = path.join(tempRoot, `${String(args.fragment).trim()}-suggested.zones.json`);

  try {
    const suggestArgs = [];
    pushOption(suggestArgs, 'route', args.route);
    pushOption(suggestArgs, 'zones', args.zones);
    pushOption(suggestArgs, 'output', suggestedZonePath);
    pushOption(suggestArgs, 'corridor-padding', args['corridor-padding']);
    pushOption(suggestArgs, 'bounds-padding', args['bounds-padding']);
    pushOption(suggestArgs, 'cost', args.cost);
    pushOption(suggestArgs, 'label-prefix', args['label-prefix']);
    pushFlag(suggestArgs, 'include-hazard-review', Boolean(args['include-hazard-review']));
    pushOption(suggestArgs, 'hazard-cost', args['hazard-cost']);
    pushOption(suggestArgs, 'hazard-label-prefix', args['hazard-label-prefix']);
    pushFlag(suggestArgs, 'include-bounds-review', Boolean(args['include-bounds-review']));
    pushOption(suggestArgs, 'bounds-cost', args['bounds-cost']);
    pushOption(suggestArgs, 'bounds-label-prefix', args['bounds-label-prefix']);
    pushFlag(suggestArgs, 'merge-bounds', Boolean(args['merge-bounds']));

    console.log('Step 1/2: suggest-robot-zones');
    runNodeScript('suggest-robot-zones.mjs', suggestArgs);

    const stageArgs = [];
    pushOption(stageArgs, 'source', suggestedZonePath);
    pushOption(stageArgs, 'fragment', args.fragment);
    pushOption(stageArgs, 'zone-path', args['zone-path']);
    pushOption(stageArgs, 'public-root', args['public-root']);
    pushOption(stageArgs, 'frame-id', args['frame-id']);
    pushOption(stageArgs, 'resolution', args.resolution);
    pushOption(stageArgs, 'default-cost', args['default-cost']);
    pushFlag(stageArgs, 'dry-run', Boolean(args['dry-run']));
    pushFlag(stageArgs, 'force', Boolean(args.force));

    console.log('Step 2/2: stage-robot-zones');
    runNodeScript('stage-robot-zones.mjs', stageArgs);

    if (args.validate && !args['dry-run']) {
      if (hasNonEmptyString(args['public-root'])) {
        console.log('WARN: custom public-root 指定時は validate:studio を自動実行しません。');
      } else {
        console.log('Step 3/3: validate-studio');
        runNodeScript('validate-studio-assets.mjs', []);
      }
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
