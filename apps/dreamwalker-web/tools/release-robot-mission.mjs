import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { defaultDiscoveryRoots, discoverRobotBundles } from './robot-bundle-discovery.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function printUsage() {
  console.log(`Dynamic Map Viewer robot mission release

Usage:
  node ./tools/release-robot-mission.mjs --bundle ./downloads/dreamwalker-live-residency.artifact-pack.json --force --validate
  node ./tools/release-robot-mission.mjs --discover --force --validate

Options:
  --bundle <file|url>           required. artifact pack または draft bundle JSON.
  --discover                    search roots から最新の artifact pack を自動選択します。
  --root <dir>                  discover search root. repeatable.
  --output-dir <dir>            auto preflight/report output 先。未指定時は local bundle の隣を使います。
  --public-root <dir>           optional. validator にも引き渡します。
  --validate-public-root <dir>  optional. validator だけ別 public root を使います。
  --skip-validate               skip validate:robot-bundle and run publish only.
  --help                        show this message.

他の option はそのまま publish:robot-mission へ渡します。
--preflight-output / --report-output を未指定で local bundle を使う場合は、
<bundle stem>.preflight.txt と <bundle stem>.publish-report.json を自動で出力します。`);
}

function parseArgs(argv) {
  const parsed = {
    publishArgs: [],
    bundle: '',
    discover: false,
    roots: [],
    outputDir: '',
    publicRoot: '',
    validatePublicRoot: '',
    skipValidate: false,
    help: false
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (token === '--help') {
      parsed.help = true;
      continue;
    }

    if (token === '--skip-validate') {
      parsed.skipValidate = true;
      continue;
    }

    if (token === '--discover') {
      parsed.discover = true;
      continue;
    }

    if (
      token === '--bundle' ||
      token === '--root' ||
      token === '--output-dir' ||
      token === '--public-root' ||
      token === '--validate-public-root'
    ) {
      const nextToken = argv[index + 1];
      if (!nextToken || nextToken.startsWith('--')) {
        throw new Error(`値が必要です: ${token}`);
      }

      if (token === '--bundle') {
        parsed.bundle = nextToken;
      } else if (token === '--root') {
        parsed.roots.push(path.resolve(nextToken));
      } else if (token === '--output-dir') {
        parsed.outputDir = path.resolve(nextToken);
      } else if (token === '--public-root') {
        parsed.publicRoot = nextToken;
      } else if (token === '--validate-public-root') {
        parsed.validatePublicRoot = nextToken;
      }

      if (
        token !== '--validate-public-root' &&
        token !== '--root' &&
        token !== '--output-dir'
      ) {
        parsed.publishArgs.push(token, nextToken);
      }

      index += 1;
      continue;
    }

    parsed.publishArgs.push(token);
  }

  return parsed;
}

function hasNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function isRemoteUrl(value) {
  return typeof value === 'string' && /^https?:\/\//i.test(value.trim());
}

function publishArgsContainOption(args, optionName) {
  return args.some((entry) => entry === optionName);
}

function inferBundleStem(bundleInput) {
  const rawName = isRemoteUrl(bundleInput)
    ? path.basename(new URL(bundleInput).pathname)
    : path.basename(bundleInput);
  const withoutJson = rawName.replace(/\.json$/i, '');
  const normalized = withoutJson.replace(/\.(artifact-pack|draft-bundle)$/i, '');
  return normalized || 'dreamwalker-robot-mission';
}

function resolveAutoOutputPaths(bundleInput, outputDir) {
  if (!hasNonEmptyString(bundleInput)) {
    return null;
  }

  const outputRoot = hasNonEmptyString(outputDir)
    ? path.resolve(outputDir)
    : isRemoteUrl(bundleInput)
      ? ''
      : path.dirname(path.resolve(bundleInput));

  if (!outputRoot) {
    return null;
  }

  const stem = inferBundleStem(bundleInput);
  return {
    outputRoot,
    preflightOutput: path.join(outputRoot, `${stem}.preflight.txt`),
    reportOutput: path.join(outputRoot, `${stem}.publish-report.json`)
  };
}

function runStep(scriptFileName, args) {
  const result = spawnSync(process.execPath, [path.join(__dirname, scriptFileName), ...args], {
    cwd: path.resolve(__dirname, '..'),
    stdio: 'inherit'
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

async function resolveBundleInput(parsed) {
  if (parsed.bundle) {
    return parsed.bundle;
  }

  if (!parsed.discover) {
    return '';
  }

  const roots = parsed.roots.length ? parsed.roots : defaultDiscoveryRoots();
  const discovery = await discoverRobotBundles(roots);
  if (!discovery.artifactPacks.length) {
    throw new Error(
      `artifact-pack が見つかりません。search roots: ${roots.join(', ')}`
    );
  }

  const selected = discovery.artifactPacks[0];
  console.log(`[release] discovered artifact-pack: ${selected.path}`);
  if (discovery.artifactPacks.length > 1) {
    console.log(
      `[release] using latest of ${discovery.artifactPacks.length} candidates`
    );
  }
  return selected.path;
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));

  if (parsed.help) {
    printUsage();
    return;
  }

  const bundleInput = await resolveBundleInput(parsed);

  if (!bundleInput) {
    throw new Error('--bundle か --discover が必要です。release は bundle / artifact pack 前提です。');
  }

  const validateArgs = ['--bundle', bundleInput];
  const validatePublicRoot = parsed.validatePublicRoot || parsed.publicRoot;
  if (validatePublicRoot) {
    validateArgs.push('--public-root', validatePublicRoot);
  }

  if (!parsed.bundle) {
    const bundleIndex = parsed.publishArgs.findIndex((entry) => entry === '--bundle');
    if (bundleIndex === -1) {
      parsed.publishArgs.unshift(bundleInput);
      parsed.publishArgs.unshift('--bundle');
    }
  }

  const autoOutputs = resolveAutoOutputPaths(bundleInput, parsed.outputDir);
  const hasPreflightOutput = publishArgsContainOption(parsed.publishArgs, '--preflight-output');
  const hasReportOutput = publishArgsContainOption(parsed.publishArgs, '--report-output');
  if (autoOutputs) {
    if (!hasPreflightOutput) {
      parsed.publishArgs.push('--preflight-output', autoOutputs.preflightOutput);
    }
    if (!hasReportOutput) {
      parsed.publishArgs.push('--report-output', autoOutputs.reportOutput);
    }
    console.log('[release] auto outputs');
    console.log(`- preflight: ${hasPreflightOutput ? 'user-specified' : autoOutputs.preflightOutput}`);
    console.log(`- report: ${hasReportOutput ? 'user-specified' : autoOutputs.reportOutput}`);
  } else if (!hasPreflightOutput || !hasReportOutput) {
    console.log('[release] auto outputs unavailable (remote bundle without --output-dir)');
  }

  if (!parsed.skipValidate) {
    console.log('[release] validate:robot-bundle');
    runStep('validate-robot-bundle.mjs', validateArgs);
  } else {
    console.log('[release] skip validate:robot-bundle');
  }

  console.log('[release] publish:robot-mission');
  runStep('publish-robot-mission.mjs', parsed.publishArgs);
}

try {
  await main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
