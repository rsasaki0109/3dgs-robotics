import { execFile } from 'node:child_process';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { expect, test } from '@playwright/test';
import WebSocket, { WebSocketServer } from 'ws';
import {
  buildCameraFrameMessage,
  buildDepthFrameMessage,
  robotBridgeProtocolId
} from '../src/robotics-bridge.js';
import { sim2realQueryProtocolId } from '../src/sim2real-query.js';

const shardStorageKeyPrefix = 'dreamwalker-live-collected-shards:';
const assetWorkspaceStorageKey = 'dreamwalker-live-asset-workspace';
const sceneWorkspaceStorageKey = 'dreamwalker-live-scene-workspace';
const semanticZoneWorkspaceStorageKey = 'dreamwalker-live-semantic-zone-workspace';
const studioBundleShelfStorageKey = 'dreamwalker-live-studio-bundle-shelf';
const robotRouteShelfStorageKey = 'dreamwalker-live-robot-route-shelf';
const robotMissionDraftBundleShelfStorageKey =
  'dreamwalker-live-robot-mission-draft-bundle-shelf';
const execFileAsync = promisify(execFile);
const dreamwalkerWebRoot = path.resolve(fileURLToPath(new URL('..', import.meta.url)));

function ensureBridgeMessageBuffer(socket) {
  if (socket.__dreamwalkerBridgeBuffer) {
    return socket.__dreamwalkerBridgeBuffer;
  }

  const state = {
    messages: [],
    errors: []
  };

  socket.__dreamwalkerBridgeBuffer = state;

  socket.on('message', (buffer) => {
    try {
      state.messages.push(JSON.parse(buffer.toString()));
      if (state.messages.length > 100) {
        state.messages.shift();
      }
    } catch (error) {
      state.errors.push(error);
    }
  });

  socket.on('error', (error) => {
    state.errors.push(error);
  });

  return state;
}

function waitForBridgeOpen(socket) {
  return new Promise((resolve, reject) => {
    ensureBridgeMessageBuffer(socket);

    const timeoutId = setTimeout(() => {
      reject(new Error('robot bridge socket open timed out'));
    }, 10_000);

    socket.once('open', () => {
      clearTimeout(timeoutId);
      resolve();
    });

    socket.once('error', (error) => {
      clearTimeout(timeoutId);
      reject(error);
    });
  });
}

function waitForBridgeMessage(socket, predicate) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const state = ensureBridgeMessageBuffer(socket);

    function poll() {
      if (state.errors.length > 0) {
        reject(state.errors.shift());
        return;
      }

      const matchIndex = state.messages.findIndex((message) => predicate(message));
      if (matchIndex >= 0) {
        const [message] = state.messages.splice(matchIndex, 1);
        resolve(message);
        return;
      }

      if (Date.now() - startedAt >= 10_000) {
        reject(new Error('robot bridge message timed out'));
        return;
      }

      setTimeout(poll, 20);
    }

    poll();
  });
}

function waitForCondition(predicate, label, timeoutMs = 10_000, intervalMs = 50) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();

    function poll() {
      try {
        const value = predicate();
        if (value) {
          resolve(value);
          return;
        }
      } catch (error) {
        reject(error);
        return;
      }

      if (Date.now() - startedAt >= timeoutMs) {
        reject(new Error(`${label} timed out`));
        return;
      }

      setTimeout(poll, intervalMs);
    }

    poll();
  });
}

function findLastMessage(messages, predicate) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (predicate(messages[index])) {
      return messages[index];
    }
  }

  return null;
}

function decodeFloat32LeSamples(buffer, maxSamples = 128) {
  const sampleCount = Math.min(Math.floor(buffer.length / 4), maxSamples);
  const samples = [];

  for (let index = 0; index < sampleCount; index += 1) {
    samples.push(buffer.readFloatLE(index * 4));
  }

  return samples;
}

async function createRosbridgeMockServer() {
  const frames = [];
  let activeSocket = null;
  const server = new WebSocketServer({
    host: '127.0.0.1',
    port: 0
  });

  await new Promise((resolve, reject) => {
    server.once('listening', resolve);
    server.once('error', reject);
  });

  server.on('connection', (socket) => {
    activeSocket = socket;

    socket.on('message', (buffer) => {
      frames.push(JSON.parse(buffer.toString()));
    });

    socket.on('close', () => {
      if (activeSocket === socket) {
        activeSocket = null;
      }
    });
  });

  return {
    frames,
    getSocket() {
      return activeSocket;
    },
    url: `ws://127.0.0.1:${server.address().port}`,
    async close() {
      activeSocket?.close();
      await new Promise((resolve) => server.close(resolve));
    }
  };
}

async function createRoboticsMockServer() {
  let activeSocket = null;
  const messages = [];
  const server = new WebSocketServer({
    host: '127.0.0.1',
    port: 0
  });

  await new Promise((resolve, reject) => {
    server.once('listening', resolve);
    server.once('error', reject);
  });

  server.on('connection', (socket) => {
    activeSocket = socket;

    socket.on('message', (payload, isBinary) => {
      messages.push({
        isBinary,
        payload
      });
    });

    socket.on('close', () => {
      if (activeSocket === socket) {
        activeSocket = null;
      }
    });
  });

  return {
    messages,
    getSocket() {
      return activeSocket;
    },
    url: `ws://127.0.0.1:${server.address().port}/robotics`,
    async close() {
      activeSocket?.close();
      await new Promise((resolve) => server.close(resolve));
    }
  };
}

function buildSim2realDepthBase64(width, height) {
  const buffer = Buffer.alloc(width * height * 4);

  for (let index = 0; index < width * height; index += 1) {
    const x = index % width;
    const y = Math.floor(index / width);
    buffer.writeFloatLE(1.0 + x * 0.02 + y * 0.01, index * 4);
  }

  return buffer.toString('base64');
}

function buildMockLocalizationImageBenchmarkReport(message, endpoint = '') {
  const groundTruthBundle =
    message && typeof message.groundTruthBundle === 'object' ? message.groundTruthBundle : {};
  const captures = Array.isArray(groundTruthBundle.captures) ? groundTruthBundle.captures : [];
  const firstCapture = captures[0] ?? null;
  const lastCapture = captures.at(-1) ?? firstCapture;
  const estimate =
    message && typeof message.estimate === 'object' && !Array.isArray(message.estimate)
      ? message.estimate
      : {};
  const estimateLabel =
    typeof estimate.label === 'string' && estimate.label.trim()
      ? estimate.label.trim()
      : 'Localization Estimate';
  const isOffsetRun = /offset|run b/i.test(estimateLabel);
  const matchedCount = Math.max(
    1,
    Math.min(
      captures.length || 1,
      Array.isArray(estimate.poses) && estimate.poses.length > 0 ? estimate.poses.length : captures.length || 1
    )
  );

  return {
    protocol: 'dreamwalker-localization-image-benchmark/v1',
    type: 'localization-image-benchmark-report',
    createdAt: '2026-04-02T00:10:00.000Z',
    endpoint,
    alignment:
      typeof message.alignment === 'string' && message.alignment.trim() ? message.alignment.trim() : 'auto',
    groundTruth: {
      fragmentId:
        typeof groundTruthBundle.fragmentId === 'string' ? groundTruthBundle.fragmentId : 'residency',
      fragmentLabel:
        typeof groundTruthBundle.fragmentLabel === 'string'
          ? groundTruthBundle.fragmentLabel
          : 'Residency Run Shelf GT'
    },
    estimate: {
      label: estimateLabel,
      sourceType:
        typeof estimate.sourceType === 'string' && estimate.sourceType.trim()
          ? estimate.sourceType.trim()
          : 'poses'
    },
    matching: {
      matchedCount
    },
    metrics: {
      summary: {
        lpips: { mean: isOffsetRun ? 0.5 : 0.123 },
        psnr: { mean: isOffsetRun ? 19.8 : 28.5 },
        ssim: { mean: isOffsetRun ? 0.744 : 0.932 }
      },
      highlights: {
        lpips: {
          ordering: 'max',
          frameIndex: isOffsetRun ? 0 : Math.max(0, matchedCount - 1),
          value: isOffsetRun ? 0.661 : 0.245,
          groundTruthLabel:
            typeof lastCapture?.label === 'string'
              ? isOffsetRun
                ? firstCapture?.label || lastCapture.label
                : lastCapture.label
              : 'gt:end',
          estimateLabel,
          groundTruthColorJpegBase64:
            isOffsetRun
              ? firstCapture?.response?.colorJpegBase64 || lastCapture?.response?.colorJpegBase64 || ''
              : lastCapture?.response?.colorJpegBase64 || firstCapture?.response?.colorJpegBase64 || '',
          renderedColorJpegBase64:
            isOffsetRun
              ? firstCapture?.response?.colorJpegBase64 || lastCapture?.response?.colorJpegBase64 || ''
              : lastCapture?.response?.colorJpegBase64 || firstCapture?.response?.colorJpegBase64 || ''
        }
      }
    },
    frames: []
  };
}

async function createSim2RealMockServer() {
  const messages = [];
  let activeSocket = null;
  const server = new WebSocketServer({
    host: '127.0.0.1',
    port: 0
  });

  await new Promise((resolve, reject) => {
    server.once('listening', resolve);
    server.once('error', reject);
  });

  const url = `ws://127.0.0.1:${server.address().port}/sim2real`;

  server.on('connection', (socket) => {
    activeSocket = socket;
    socket.send(
      JSON.stringify({
        protocol: sim2realQueryProtocolId,
        type: 'query-ready',
        transport: 'ws',
        endpoint: url,
        frameId: 'dreamwalker_map',
        renderer: 'gsplat',
        rendererReason: 'mock renderer',
        requestTypes: ['render', 'localization-image-benchmark'],
        defaults: {
          width: 64,
          height: 48,
          fovDegrees: 60,
          nearClip: 0.05,
          farClip: 50,
          pointRadius: 1
        }
      })
    );

    socket.on('message', (buffer) => {
      const message = JSON.parse(buffer.toString());
      messages.push(message);

      if (message.type !== 'render') {
        if (message.type === 'localization-image-benchmark') {
          socket.send(JSON.stringify(buildMockLocalizationImageBenchmarkReport(message, url)));
        }
        return;
      }

      const width = Number(message.width) || 64;
      const height = Number(message.height) || 48;
      socket.send(
        JSON.stringify({
          protocol: sim2realQueryProtocolId,
          type: 'render-result',
          frameId: 'dreamwalker_map',
          width,
          height,
          fovDegrees: Number(message.fovDegrees) || 60,
          nearClip: Number(message.nearClip) || 0.05,
          farClip: Number(message.farClip) || 50,
          pointRadius: Number(message.pointRadius) || 1,
          pose: message.pose,
          cameraInfo: {
            frameId: 'dreamwalker_map',
            width,
            height,
            distortionModel: 'plumb_bob',
            d: [0, 0, 0, 0, 0],
            k: [60, 0, width / 2, 0, 60, height / 2, 0, 0, 1],
            r: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            p: [60, 0, width / 2, 0, 0, 60, height / 2, 0, 0, 0, 1, 0]
          },
          colorEncoding: 'jpeg',
          colorJpegBase64: Buffer.from([0xff, 0xd8, 0xff, 0xd9]).toString('base64'),
          depthEncoding: '32FC1',
          depthBase64: buildSim2realDepthBase64(width, height)
        })
      );
    });

    socket.on('close', () => {
      if (activeSocket === socket) {
        activeSocket = null;
      }
    });
  });

  return {
    messages,
    getSocket() {
      return activeSocket;
    },
    url,
    async close() {
      activeSocket?.close();
      await new Promise((resolve) => server.close(resolve));
    }
  };
}

async function createLiveLocalizationMockServer() {
  let activeSocket = null;
  const server = new WebSocketServer({
    host: '127.0.0.1',
    port: 0
  });

  await new Promise((resolve, reject) => {
    server.once('listening', resolve);
    server.once('error', reject);
  });

  server.on('connection', (socket) => {
    activeSocket = socket;

    socket.on('close', () => {
      if (activeSocket === socket) {
        activeSocket = null;
      }
    });
  });

  return {
    async send(message) {
      const socket = await waitForCondition(
        () => activeSocket,
        'live localization socket connection'
      );
      socket.send(JSON.stringify(message));
    },
    url: `ws://127.0.0.1:${server.address().port}/localization`,
    async close() {
      activeSocket?.close();
      await new Promise((resolve) => server.close(resolve));
    }
  };
}

function spawnRosbridgeRelay(args = []) {
  const child = spawn('node', ['./tools/robotics-rosbridge-relay.mjs', ...args], {
    cwd: dreamwalkerWebRoot,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';

  child.stdout.on('data', (chunk) => {
    output += chunk.toString();
  });
  child.stderr.on('data', (chunk) => {
    output += chunk.toString();
  });

  return {
    child,
    getOutput() {
      return output;
    }
  };
}

async function stopChildProcess(child) {
  if (!child || child.exitCode !== null) {
    return;
  }

  child.kill('SIGTERM');

  await new Promise((resolve) => {
    const timeoutId = setTimeout(() => {
      child.kill('SIGKILL');
      resolve();
    }, 3_000);

    child.once('exit', () => {
      clearTimeout(timeoutId);
      resolve();
    });
  });
}

test('Dynamic Map Viewer smoke flow', async ({ page, browser, baseURL }) => {
  await page.goto('/?relay=1');
  await page.evaluate(({ storageKeyPrefixValue, assetWorkspaceStorageKeyValue }) => {
    const keysToRemove = [];
    for (let index = 0; index < window.localStorage.length; index += 1) {
      const key = window.localStorage.key(index);
      if (key?.startsWith(storageKeyPrefixValue)) {
        keysToRemove.push(key);
      }
    }

    keysToRemove.forEach((key) => window.localStorage.removeItem(key));
    window.localStorage.removeItem(assetWorkspaceStorageKeyValue);
  }, {
    storageKeyPrefixValue: shardStorageKeyPrefix,
    assetWorkspaceStorageKeyValue: assetWorkspaceStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');

  await expect(page.getByRole('heading', { name: 'Dynamic Map Viewer' })).toBeVisible();
  await expect(page.locator('.dreamwalker-stage canvas')).toBeVisible();
  await expect(page.getByText('No Splat Configured')).toHaveCount(0);
  await expect(leftPanel.getByText('Local Dynamic Map Asset Manifest', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Residency', { exact: true }).first()).toBeVisible();
  await expect(leftPanel.getByText('Proxy Floor', { exact: true }).first()).toBeVisible();
  await expect(leftPanel.getByText('Demo Fallback', { exact: true }).first()).toBeVisible();
  await expect(leftPanel.getByText('World Health', { exact: true })).toBeVisible();
  await leftPanel
    .getByLabel('Asset Workspace JSON Import')
    .fill('{"label":"Imported Workspace","fragments":{"residency":{"label":"Residency Long Use"}}}');
  await leftPanel.getByRole('button', { name: 'Apply Pasted Asset Workspace JSON', exact: true }).click();
  await expect(leftPanel.getByLabel('Workspace Manifest Label')).toHaveValue('Imported Workspace');
  await leftPanel.getByRole('button', { name: 'Save Asset Workspace', exact: true }).click();
  await page.reload();
  await expect(leftPanel.getByText('Residency Long Use', { exact: true }).first()).toBeVisible();
  await leftPanel.getByRole('button', { name: 'Reset Asset Workspace', exact: true }).click();
  await expect(leftPanel.getByText('Residency Marble', { exact: true }).first()).toBeVisible();

  await topbar.getByRole('button', { name: 'Photo', exact: true }).click();
  await expect(page.locator('.photo-guides')).toBeVisible();

  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await expect(page.locator('.stream-safe-overlay')).toBeVisible();
  await expect(page.locator('.live-scene-card').getByRole('heading', { name: 'Residency Intro' })).toBeVisible();
  await page.keyboard.press('8');
  await expect(page.locator('.stream-safe-overlay')).toHaveAttribute('data-overlay-preset', 'side-stack');
  await page.keyboard.press('5');
  await expect(page.locator('.live-scene-card').getByRole('heading', { name: 'Window Talk' })).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Window Memo', { exact: true })).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('写真モード告知を 1 回', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"streamSceneTitle": "Window Talk"');
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"overlayBrandingBadge": "Window Talk"');
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"overlayBrandingId": "residency-branding"');
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"overlayPresetId": "side-stack"');
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"overlayMemoTitle": "Window Memo"');
  await expect(rightPanel.getByText('Overlay transport: Relay SSE')).toBeVisible();
  await expect(rightPanel.getByText('Relay URL: http://127.0.0.1:8787')).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await rightPanel.getByRole('button', { name: 'Download Scene JSON', exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toContain('window-talk');

  const overlayContext = await browser.newContext();
  const overlayPage = await overlayContext.newPage();
  await overlayPage.goto(`${baseURL}/overlay.html?relay=1`);
  await expect(overlayPage.locator('.obs-overlay-shell[data-overlay-preset="side-stack"]')).toBeVisible();
  await expect(overlayPage.locator('.obs-overlay-shell[data-overlay-branding="residency-branding"]')).toBeVisible();
  await expect(overlayPage.getByText('Window Talk', { exact: true }).first()).toBeVisible();
  await expect(
    overlayPage.locator('.obs-overlay-card').getByRole('heading', { name: 'Window Talk', exact: true })
  ).toBeVisible();
  await expect(overlayPage.getByText('Window Memo', { exact: true })).toBeVisible();
  await expect(overlayPage.getByText('写真モード告知を 1 回', { exact: true })).toBeVisible();
  await page.keyboard.press('6');
  await expect(page.locator('.live-scene-card').getByRole('heading', { name: 'Gate Recap' })).toBeVisible();
  await expect(
    overlayPage.locator('.obs-overlay-card').getByRole('heading', { name: 'Gate Recap', exact: true })
  ).toBeVisible();

  await topbar.getByRole('button', { name: 'Explore', exact: true }).click();
  await expect(page.locator('.stream-safe-overlay')).toHaveCount(0);

  await rightPanel.getByRole('button', { name: 'Shard 01', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Shard 02', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Shard 03', exact: true }).scrollIntoViewIfNeeded();
  await rightPanel.getByRole('button', { name: 'Shard 03', exact: true }).click();

  await expect(leftPanel.getByText('3 / 3', { exact: true })).toBeVisible();
  await expect(rightPanel.getByRole('button', { name: 'Gate Open', exact: true })).toBeVisible();

  await rightPanel.getByRole('button', { name: 'Gate Open', exact: true }).click();
  await expect(leftPanel.getByText('Echo Chamber', { exact: true }).first()).toBeVisible();
  await expect(page).toHaveURL(/#echo-chamber$/);
  await expect(leftPanel.getByText('0 / 3', { exact: true })).toBeVisible();
  await expect(rightPanel.getByRole('button', { name: 'Echo 01', exact: true })).toBeVisible();
  await expect(overlayPage.locator('.obs-overlay-shell[data-overlay-branding="echo-chamber-branding"]')).toBeVisible();
  await expect(
    overlayPage.locator('.obs-overlay-card').getByRole('heading', { name: 'Echo Threshold', exact: true })
  ).toBeVisible();

  await page.reload();
  await expect(leftPanel.getByText('Echo Chamber', { exact: true }).first()).toBeVisible();
  await expect(leftPanel.getByText('0 / 3', { exact: true })).toBeVisible();

  await page.evaluate(() => {
    window.location.hash = 'residency';
  });
  await expect(leftPanel.getByText('Residency', { exact: true }).first()).toBeVisible();
  await expect(leftPanel.getByText('3 / 3', { exact: true })).toBeVisible();

  await leftPanel.getByRole('button', { name: 'Reset Dream State', exact: true }).click();
  await expect(leftPanel.getByText('0 / 3', { exact: true })).toBeVisible();

  await topbar.getByRole('button', { name: 'Walk', exact: true }).click();
  await expect(page.locator('.walk-hud')).toBeVisible();
  await expect(page.locator('.reticle-label')).toContainText('F:');
  await page.keyboard.press('f');
  await expect(page.getByRole('heading', { name: 'Echo Note: ここに住んでいる設定' })).toBeVisible();
  await page.getByRole('button', { name: 'Close' }).click();
  await overlayContext.close();

  await page.getByRole('button', { name: 'Orbit' }).click();
  await expect(page.locator('.walk-hud')).toHaveCount(0);
});

test('Stream scene workspace persists and resets', async ({ page, browser, baseURL }) => {
  await page.goto('/?relay=1');
  await page.evaluate(({ sceneWorkspaceStorageKeyValue }) => {
    window.localStorage.removeItem(sceneWorkspaceStorageKeyValue);
  }, {
    sceneWorkspaceStorageKeyValue: sceneWorkspaceStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');

  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await page.keyboard.press('8');
  await page.keyboard.press('5');
  await rightPanel.getByLabel('Scene Workspace Title').fill('Window Talk Long Use');
  await rightPanel.getByLabel('Scene Memo Title').fill('Long Use Memo');
  await rightPanel.getByLabel('Scene Memo Items').fill('進捗を 1 本\n深夜の雑談を 1 本\n写真モード告知を 1 回');
  await rightPanel.getByRole('button', { name: 'Save Scene Workspace', exact: true }).click();

  await expect(page.locator('.live-scene-card').getByRole('heading', { name: 'Window Talk Long Use' })).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Long Use Memo', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"streamSceneTitle": "Window Talk Long Use"');
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"overlayMemoTitle": "Long Use Memo"');

  const overlayContext = await browser.newContext();
  const overlayPage = await overlayContext.newPage();
  await overlayPage.goto(`${baseURL}/overlay.html?relay=1`);
  await expect(
    overlayPage.locator('.obs-overlay-card').getByRole('heading', { name: 'Window Talk Long Use', exact: true })
  ).toBeVisible();
  await expect(overlayPage.getByText('Long Use Memo', { exact: true })).toBeVisible();

  await page.reload();
  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await page.keyboard.press('5');
  await expect(page.locator('.live-scene-card').getByRole('heading', { name: 'Window Talk Long Use' })).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Long Use Memo', { exact: true })).toBeVisible();

  await rightPanel.getByRole('button', { name: 'Reset Scene Workspace', exact: true }).click();
  await page.keyboard.press('5');
  await expect(page.locator('.live-scene-card').getByRole('heading', { name: 'Window Talk' })).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Window Memo', { exact: true })).toBeVisible();
  await expect(
    overlayPage.locator('.obs-overlay-card').getByRole('heading', { name: 'Window Talk', exact: true })
  ).toBeVisible();
  await expect(overlayPage.getByText('Window Memo', { exact: true })).toBeVisible();
  await overlayContext.close();
});

test('Studio bundle updates asset and scene drafts together', async ({ page }) => {
  await page.goto('/?relay=1');
  await page.evaluate(({ assetWorkspaceStorageKeyValue, sceneWorkspaceStorageKeyValue, semanticZoneWorkspaceStorageKeyValue }) => {
    window.localStorage.removeItem(assetWorkspaceStorageKeyValue);
    window.localStorage.removeItem(sceneWorkspaceStorageKeyValue);
    window.localStorage.removeItem(semanticZoneWorkspaceStorageKeyValue);
  }, {
    assetWorkspaceStorageKeyValue: assetWorkspaceStorageKey,
    sceneWorkspaceStorageKeyValue: sceneWorkspaceStorageKey,
    semanticZoneWorkspaceStorageKeyValue: semanticZoneWorkspaceStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await leftPanel.getByLabel('Studio Bundle JSON Import').fill(
    '{"label":"Studio Bundle","state":{"fragmentId":"residency","streamSceneId":"window-talk","overlayPresetId":"headline-ribbon"},"assetWorkspace":{"fragments":{"residency":{"label":"Bundle Residency"}}},"sceneWorkspace":{"fragments":{"residency":{"streamScenes":[{"id":"window-talk","title":"Bundle Window Talk","overlayMemo":{"title":"Bundle Memo","items":["beat one","beat two"],"footer":"bundle footer"}}]}}},"semanticZoneWorkspace":{"residency":{"frameId":"dreamwalker_map","resolution":0.5,"defaultCost":0,"bounds":{"minX":-6,"maxX":6,"minZ":0,"maxZ":12},"zones":[{"id":"bundle-stage-zone","label":"Bundle Stage Zone","shape":"rect","center":[0,0,5.8],"size":[5,3.4],"cost":31,"tags":["bundle","stream"]}]}},"robotRoute":{"pose":{"position":[1.2,0,7.0],"yawDegrees":30},"route":[[0,0,5.8],[1.2,0,7.0]],"waypoint":{"position":[1.2,0,9.8]}}}'
  );
  await leftPanel.getByRole('button', { name: 'Apply Pasted Studio Bundle JSON', exact: true }).click();
  await expect(leftPanel.getByText('Bundle Residency', { exact: true }).first()).toBeVisible();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(currentZoneCard).toContainText('Bundle Stage Zone');
  await expect(poseCard).toContainText('x 1.20 / z 7.00');
  await expect(routeCard).toContainText('2 nodes');

  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await expect(page.locator('.stream-safe-overlay')).toHaveAttribute('data-overlay-preset', 'headline-ribbon');
  await expect(
    page.locator('.live-scene-card').getByRole('heading', { name: 'Bundle Window Talk', exact: true })
  ).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Bundle Memo', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"streamSceneTitle": "Bundle Window Talk"');
});

test('Studio bundle url loads startup stage set', async ({ page }) => {
  await page.goto('/?studioBundle=/studio-bundles/dreamwalker-live.sample.json');
  await page.evaluate(({ assetWorkspaceStorageKeyValue, sceneWorkspaceStorageKeyValue, semanticZoneWorkspaceStorageKeyValue, studioBundleShelfStorageKeyValue }) => {
    window.localStorage.removeItem(assetWorkspaceStorageKeyValue);
    window.localStorage.removeItem(sceneWorkspaceStorageKeyValue);
    window.localStorage.removeItem(semanticZoneWorkspaceStorageKeyValue);
    window.localStorage.removeItem(studioBundleShelfStorageKeyValue);
  }, {
    assetWorkspaceStorageKeyValue: assetWorkspaceStorageKey,
    sceneWorkspaceStorageKeyValue: sceneWorkspaceStorageKey,
    semanticZoneWorkspaceStorageKeyValue: semanticZoneWorkspaceStorageKey,
    studioBundleShelfStorageKeyValue: studioBundleShelfStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);
  await expect(leftPanel.getByText('Dynamic Map Viewer Sample Studio Bundle', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('/studio-bundles/dreamwalker-live.sample.json', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Sample Residency Bundle', { exact: true }).first()).toBeVisible();

  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await expect(page.locator('.stream-safe-overlay')).toHaveAttribute('data-overlay-preset', 'side-stack');
  await expect(
    page.locator('.live-scene-card').getByRole('heading', { name: 'Sample Window Talk', exact: true })
  ).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Sample Bundle Memo', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Live Scene JSON')).toContainText('"streamSceneTitle": "Sample Window Talk"');

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(currentZoneCard).toContainText('Sample Stage Zone');
  await expect(poseCard).toContainText('x 1.00 / z 6.80');
  await expect(routeCard).toContainText('3 nodes');
});

test('Public studio bundle catalog applies sample bundle', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(({ assetWorkspaceStorageKeyValue, sceneWorkspaceStorageKeyValue, semanticZoneWorkspaceStorageKeyValue, studioBundleShelfStorageKeyValue }) => {
    window.localStorage.removeItem(assetWorkspaceStorageKeyValue);
    window.localStorage.removeItem(sceneWorkspaceStorageKeyValue);
    window.localStorage.removeItem(semanticZoneWorkspaceStorageKeyValue);
    window.localStorage.removeItem(studioBundleShelfStorageKeyValue);
  }, {
    assetWorkspaceStorageKeyValue: assetWorkspaceStorageKey,
    sceneWorkspaceStorageKeyValue: sceneWorkspaceStorageKey,
    semanticZoneWorkspaceStorageKeyValue: semanticZoneWorkspaceStorageKey,
    studioBundleShelfStorageKeyValue: studioBundleShelfStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await expect(leftPanel.getByText('Dynamic Map Viewer Public Bundle Catalog', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Sample Residency Bundle', { exact: true })).toBeVisible();
  await expect(
    leftPanel
      .locator('.state-card')
      .filter({ hasText: 'Sample Residency Bundle' })
      .getByText('Demo Fallback', { exact: true })
  ).toBeVisible();
  await leftPanel
    .locator('.state-card')
    .filter({ hasText: 'Sample Residency Bundle' })
    .getByRole('button', { name: 'Apply', exact: true })
    .click();

  await expect(leftPanel.getByText('Sample Residency Bundle', { exact: true }).first()).toBeVisible();
  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await expect(page.locator('.stream-safe-overlay')).toHaveAttribute('data-overlay-preset', 'side-stack');
  await expect(
    page.locator('.live-scene-card').getByRole('heading', { name: 'Sample Window Talk', exact: true })
  ).toBeVisible();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(currentZoneCard).toContainText('Sample Stage Zone');
  await expect(poseCard).toContainText('x 1.00 / z 6.80');
  await expect(routeCard).toContainText('3 nodes');
});

test('Studio bundle shelf saves and reapplies snapshots', async ({ page }) => {
  await page.goto('/?relay=1');
  await page.evaluate(({ semanticZoneWorkspaceStorageKeyValue, studioBundleShelfStorageKeyValue }) => {
    window.localStorage.removeItem(semanticZoneWorkspaceStorageKeyValue);
    window.localStorage.removeItem(studioBundleShelfStorageKeyValue);
  }, {
    semanticZoneWorkspaceStorageKeyValue: semanticZoneWorkspaceStorageKey,
    studioBundleShelfStorageKeyValue: studioBundleShelfStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await leftPanel.getByLabel('Studio Bundle JSON Import').fill(
    '{"label":"Studio Bundle","state":{"fragmentId":"residency","streamSceneId":"window-talk","overlayPresetId":"headline-ribbon"},"assetWorkspace":{"fragments":{"residency":{"label":"Shelf Residency"}}},"sceneWorkspace":{"fragments":{"residency":{"streamScenes":[{"id":"window-talk","title":"Shelf Window Talk","overlayMemo":{"title":"Shelf Memo","items":["beat one"],"footer":"shelf footer"}}]}}},"semanticZoneWorkspace":{"residency":{"frameId":"dreamwalker_map","resolution":0.5,"defaultCost":0,"bounds":{"minX":-6,"maxX":6,"minZ":0,"maxZ":12},"zones":[{"id":"shelf-stage-zone","label":"Shelf Stage Zone","shape":"rect","center":[0,0,5.8],"size":[5,3.4],"cost":27,"tags":["shelf","stream"]}]}},"robotRoute":{"pose":{"position":[1.4,0,7.2],"yawDegrees":22},"route":[[0,0,5.8],[1.4,0,7.2]],"waypoint":{"position":[1.4,0,10.0]}}}'
  );
  await leftPanel.getByRole('button', { name: 'Apply Pasted Studio Bundle JSON', exact: true }).click();
  await leftPanel.getByLabel('Studio Bundle Label').fill('Night Shelf');
  await leftPanel.getByRole('button', { name: 'Save Studio Bundle Snapshot', exact: true }).click();
  await expect(leftPanel.getByText('Night Shelf', { exact: true })).toBeVisible();

  await page.reload();
  await expect(leftPanel.getByText('Night Shelf', { exact: true })).toBeVisible();
  await leftPanel
    .locator('.state-card')
    .filter({ hasText: 'Night Shelf' })
    .getByRole('button', { name: 'Apply', exact: true })
    .click();
  await topbar.getByRole('button', { name: 'Live', exact: true }).click();
  await expect(
    page.locator('.live-scene-card').getByRole('heading', { name: 'Shelf Window Talk', exact: true })
  ).toBeVisible();
  await expect(page.locator('.stream-safe-overlay').getByText('Shelf Memo', { exact: true })).toBeVisible();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(currentZoneCard).toContainText('Shelf Stage Zone');
  await expect(poseCard).toContainText('x 1.40 / z 7.20');
  await expect(routeCard).toContainText('2 nodes');
});

test('World health detects missing local files from custom manifest', async ({ page }) => {
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());

    if (url.pathname === '/test-manifest.json') {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          label: 'Broken Manifest',
          fragments: {
            residency: {
              label: 'Broken Residency',
              splatUrl: '/splats/missing-residency.sog',
              colliderMeshUrl: '/colliders/missing-residency.glb'
            }
          }
        })
      });
      return;
    }

    await route.continue();
  });

  await page.goto('/?assetManifest=/test-manifest.json');

  const leftPanel = page.locator('.left-panel');
  await expect(leftPanel.getByText('Broken Residency', { exact: true }).first()).toBeVisible();
  await expect(leftPanel.getByText('Missing Splat File', { exact: true })).toBeVisible();
});

test('Broken public bundle blocks apply and launch', async ({ page }) => {
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());

    if (url.pathname === '/test-catalog.json') {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          label: 'Broken Catalog',
          bundles: [
            {
              id: 'broken-bundle',
              label: 'Broken Bundle',
              url: '/studio-bundles/missing-bundle.json',
              fragmentId: 'residency'
            }
          ]
        })
      });
      return;
    }

    await route.continue();
  });

  await page.goto('/?studioBundleCatalog=/test-catalog.json');

  const brokenCard = page
    .locator('.state-card')
    .filter({ hasText: 'Broken Bundle' });

  await expect(brokenCard.getByText('Bundle Missing', { exact: true })).toBeVisible();
  await expect(brokenCard.getByRole('button', { name: 'Apply', exact: true })).toBeDisabled();
  await expect(brokenCard.getByRole('button', { name: 'Launch', exact: true })).toBeDisabled();
});

test('Overlay entry avoids scene runtime requests', async ({ browser, baseURL }) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  const requests = [];

  page.on('request', (request) => {
    requests.push(request.url());
  });

  await page.goto(`${baseURL}/overlay.html?relay=1`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(600);

  const runtimeRequests = requests.filter((url) =>
    /DreamwalkerScene|@playcanvas|playcanvas\.js|sync-ammo|src\/App\.jsx|src\/styles\.css/.test(url)
  );

  expect(runtimeRequests).toEqual([]);
  await context.close();
});

test('Main entry avoids external font requests', async ({ browser, baseURL }) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  const requests = [];

  page.on('request', (request) => {
    requests.push(request.url());
  });

  await page.goto(`${baseURL}/`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);

  const fontRequests = requests.filter((url) =>
    /fonts\.googleapis\.com|fonts\.gstatic\.com/.test(url)
  );

  expect(fontRequests).toEqual([]);
  await context.close();
});

test('Walk runtime loads on demand', async ({ browser, baseURL }) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  const requests = [];

  page.on('request', (request) => {
    requests.push(request.url());
  });

  await page.goto(`${baseURL}/`, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(1200);

  const eagerWalkRequests = requests.filter((url) =>
    /WalkRuntime|first-person-controller/.test(url)
  );

  expect(eagerWalkRequests).toEqual([]);

  await page.locator('.topbar').getByRole('button', { name: 'Walk', exact: true }).click();
  await page.waitForTimeout(1200);

  const walkRequests = requests.filter((url) =>
    /WalkRuntime|first-person-controller/.test(url)
  );

  expect(walkRequests.length).toBeGreaterThan(0);
  await context.close();
});

test('Robot mode teleop updates pose and waypoint overlay', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const missionExportPanel = rightPanel
    .locator('.robot-route-export-panel')
    .filter({ hasText: 'Mission Export' })
    .first();
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(page.locator('.robotics-hud')).toBeVisible();
  await expect(rightPanel.getByRole('heading', { name: 'Robot Mode' })).toBeVisible();
  await expect(rightPanel.getByText('Front Camera Panel', { exact: true })).toBeVisible();
  await expect(routeCard).toContainText('1 node');

  const initialPose = await poseCard.textContent();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();
  await expect(poseCard).not.toHaveText(initialPose ?? '');
  await expect(routeCard).toContainText('2 nodes');

  await rightPanel.getByRole('button', { name: 'Drop Waypoint', exact: true }).click();
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('m ahead');

  await rightPanel.getByRole('button', { name: 'Top View', exact: true }).click();
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Top View');
  await expect(page.locator('.robotics-route-overlay .robotics-route-trail')).toHaveAttribute('points', / /);

  await rightPanel.getByRole('button', { name: 'Clear Route', exact: true }).click();
  await expect(routeCard).toContainText('1 node');

  await rightPanel.getByRole('button', { name: 'Reset Robot Pose', exact: true }).click();
  await expect(poseCard).toContainText('x 0.00 / z 5.80');
});

test('Robot mode route JSON exports and imports pose waypoint and route', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Drop Waypoint', exact: true }).click();

  const exportedRouteJson = await rightPanel.getByLabel('Robot Route JSON', { exact: true }).inputValue();
  const exportedRoute = JSON.parse(exportedRouteJson);
  expect(exportedRoute.protocol).toBe('dreamwalker-robot-route/v1');
  expect(exportedRoute.world.fragmentId).toBe('residency');
  expect(exportedRoute.world.assetLabel).toBe('Residency Marble');
  expect(exportedRoute.world.frameId).toBe('dreamwalker_map');
  expect(exportedRoute.route.length).toBeGreaterThanOrEqual(3);
  expect(exportedRoute.waypoint.position).toHaveLength(3);

  await rightPanel.getByLabel('Robot Route JSON Import', { exact: true }).fill(
    '{"pose":{"position":[1.5,0,8.2],"yawDegrees":90},"route":[[0,0,5.8],[1.5,0,8.2]],"waypoint":{"position":[2.5,0,9.2]}}'
  );
  await rightPanel.getByRole('button', { name: 'Apply Pasted Route JSON', exact: true }).click();

  await expect(poseCard).toContainText('x 1.50 / z 8.20');
  await expect(routeCard).toContainText('2 nodes');
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('1.41 m ahead');
  await expect(rightPanel.getByLabel('Robot Route JSON', { exact: true })).toContainText('"yawDegrees": 90');
});

test('Robot mode route shelf saves persists and reapplies snapshots', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(({ robotRouteShelfStorageKeyValue }) => {
    window.localStorage.removeItem(robotRouteShelfStorageKeyValue);
  }, {
    robotRouteShelfStorageKeyValue: robotRouteShelfStorageKey
  });
  await page.reload();

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Drop Waypoint', exact: true }).click();
  await rightPanel.getByLabel('Route Snapshot Label', { exact: true }).fill('Residency Replay 1');
  await rightPanel.getByRole('button', { name: 'Save Route Snapshot', exact: true }).click();
  await expect(rightPanel.getByText('Residency Replay 1', { exact: true })).toBeVisible();

  await rightPanel.getByRole('button', { name: 'Reset Robot Pose', exact: true }).click();
  await expect(poseCard).toContainText('x 0.00 / z 5.80');

  await page.reload();
  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(rightPanel.getByText('Residency Replay 1', { exact: true })).toBeVisible();

  await rightPanel
    .locator('.state-card')
    .filter({ hasText: 'Residency Replay 1' })
    .getByRole('button', { name: 'Apply', exact: true })
    .click();

  await expect(poseCard).not.toContainText('x 0.00 / z 5.80');
  await expect(routeCard).toContainText('2 nodes');
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('2.80 m ahead');
});

test('Robot route url loads startup route set', async ({ page }) => {
  await page.goto('/?robotRoute=/robot-routes/residency-window-loop.json');

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const missionExportPanel = rightPanel
    .locator('.robot-route-export-panel')
    .filter({ hasText: 'Mission Export' })
    .first();
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await expect(leftPanel.getByText('Route Loaded', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('/robot-routes/residency-window-loop.json', { exact: true })).toBeVisible();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(poseCard).toContainText('x 2.40 / z 8.20');
  await expect(routeCard).toContainText('4 nodes');
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('2.80 m ahead');
});

test('Public robot route catalog applies sample route', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const publicRouteCard = leftPanel
    .locator('.state-card')
    .filter({ hasText: 'Residency Window Loop' })
    .filter({ hasText: 'preset' });
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await expect(leftPanel.getByText('Dynamic Map Viewer Public Robot Route Catalog', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Residency Window Loop', { exact: true })).toBeVisible();
  await expect(publicRouteCard).toContainText('World Match');

  await publicRouteCard.getByRole('button', { name: 'Apply', exact: true }).click();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(poseCard).toContainText('x 2.40 / z 8.20');
  await expect(routeCard).toContainText('4 nodes');
});

test('Robot mission url loads startup mission set', async ({ page }) => {
  await page.goto('/?robotMission=/robot-missions/residency-window-loop.mission.json');

  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const missionExportPanel = rightPanel
    .locator('.robot-route-export-panel')
    .filter({ hasText: 'Mission Export' })
    .first();
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);
  const missionJson = rightPanel.getByLabel('Robot Mission JSON');
  const routeJson = rightPanel.locator('#robot-route-json');
  const publishedMissionPreviewJson = rightPanel.getByLabel(
    'Published Mission Preview JSON'
  );
  const missionDraftBundleJson = rightPanel.getByLabel('Mission Draft Bundle JSON');
  const missionArtifactPackJson = rightPanel.getByLabel('Mission Artifact Pack JSON');
  const missionValidateCommand = rightPanel.getByLabel('Mission Validate Command');
  const missionReleaseCommand = rightPanel.getByLabel('Mission Release Command');
  const publishReportJson = rightPanel.getByLabel('Publish Report JSON');
  const missionPublishCommand = rightPanel.getByLabel('Mission Publish Command');
  const missionIdInput = rightPanel.getByLabel('Mission ID');
  const missionLabelInput = rightPanel.getByLabel('Mission Label');
  const missionDescriptionInput = rightPanel.getByLabel('Mission Description');
  const routeLabelInput = rightPanel.getByLabel('Route Label');
  const routeDescriptionInput = rightPanel.getByLabel('Route Description');
  const routeAccentInput = rightPanel.getByLabel('Route Accent');
  const missionFragmentIdInput = rightPanel.getByLabel('Mission Fragment ID');
  const missionFragmentLabelInput = rightPanel.getByLabel('Mission Fragment Label');
  const missionRouteIdInput = rightPanel.getByLabel('Mission Route ID');
  const missionAccentInput = rightPanel.getByLabel('Mission Accent');
  const missionRouteUrlInput = rightPanel.getByLabel('Mission Route URL');
  const missionZoneMapUrlInput = rightPanel.getByLabel('Mission Zone Map URL');
  const missionWorldAssetLabelInput = rightPanel.getByLabel(
    'Mission World Asset Label'
  );
  const missionWorldFrameIdInput = rightPanel.getByLabel(
    'Mission World Frame ID'
  );

  await expect(leftPanel.getByText('Mission Loaded', { exact: true })).toBeVisible();
  await expect(
    leftPanel.getByText('/robot-missions/residency-window-loop.mission.json', { exact: true })
  ).toBeVisible();
  await expect(
    leftPanel.getByText('Mission route: /robot-routes/residency-window-loop.json', { exact: true })
  ).toBeVisible();
  await expect(leftPanel.getByText('Mission preset: window', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Mission robot camera: chase', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Mission startup mode: robot', { exact: true })).toBeVisible();
  await expect(rightPanel.getByRole('heading', { name: 'Robot Mode', exact: true })).toBeVisible();
  await expect(missionExportPanel.getByText('Mission Ready', { exact: true })).toBeVisible();
  await expect(poseCard).toContainText('x 2.40 / z 8.20');
  await expect(routeCard).toContainText('4 nodes');
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Chase Camera');
  await expect(
    rightPanel.getByRole('button', { name: 'Chase Camera', exact: true })
  ).toHaveClass(/active/);
  await expect(
    leftPanel.getByRole('button', { name: '2. Window', exact: true })
  ).toHaveClass(/active/);
  await expect(missionJson).toContainText('"cameraPresetId": "window"');
  await expect(missionJson).toContainText('"robotCameraId": "chase"');
  await expect(missionJson).toContainText('"streamSceneId": "window-talk"');
  await expect(missionJson).toContainText('"startupMode": "robot"');
  await expect(publishedMissionPreviewJson).toContainText(
    '"launchUrl": "/?robotMission=%2Frobot-missions%2Fresidency-window-loop.mission.json"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"routeUrl": "/robot-routes/residency-window-loop.json"'
  );
  await expect(missionDraftBundleJson).toContainText('"protocol": "dreamwalker-robot-mission/v1"');
  await expect(missionDraftBundleJson).toContainText('"protocol": "dreamwalker-robot-route/v1"');
  await expect(missionDraftBundleJson).toContainText('"frameId": "dreamwalker_map"');
  await expect(missionArtifactPackJson).toContainText(
    '"protocol": "dreamwalker-robot-mission-artifact-pack/v1"'
  );
  await expect(missionArtifactPackJson).toContainText(
    '"kind": "draft-bundle"'
  );
  await expect(publishReportJson).toContainText(
    '"protocol": "dreamwalker-robot-mission-publish-report/v1"'
  );
  await expect(publishReportJson).toContainText(
    '"url": "/robot-missions/residency-window-loop.mission.json"'
  );
  await expect(missionValidateCommand).toContainText(
    'npm run validate:robot-bundle --'
  );
  await expect(missionValidateCommand).toContainText(
    '# validate input: /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.json'
  );
  await expect(missionReleaseCommand).toContainText(
    'npm run release:robot-mission --'
  );
  await expect(missionReleaseCommand).toContainText(
    '--bundle /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.json'
  );
  await expect(missionReleaseCommand).toContainText(
    '# auto outputs: /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.preflight.txt + /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.publish-report.json'
  );
  await expect(missionPublishCommand).toContainText('npm run publish:robot-mission --');
  await expect(missionPublishCommand).toContainText(
    '# publish input: /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.json'
  );
  await expect(missionPublishCommand).toContainText('# preflight: Mission Ready');
  await expect(missionPublishCommand).toContainText(
    '--bundle /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.json'
  );
  await expect(missionPublishCommand).toContainText('--validate');
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Published Preview', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Published Preview', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Launch', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Launch', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Preflight', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Preflight', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Publish Report', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Publish Report', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Validate', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Validate', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Release', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Release', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Publish Command', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Copy Artifact Pack', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByRole('button', { name: 'Download Artifact Pack', exact: true })
  ).toBeVisible();
  await expect(
    rightPanel.getByText('published file residency-window-loop.mission.json', {
      exact: true
    })
  ).toBeVisible();

  await missionIdInput.fill('Residency Night Patrol');
  await missionLabelInput.fill('Residency Night Patrol Mission');
  await missionDescriptionInput.fill('night patrol preview for browser publish');
  await routeLabelInput.fill('Night Patrol Route');
  await routeDescriptionInput.fill('night patrol route for browser publish');
  await routeAccentInput.fill('#7fd6ff');
  await missionFragmentIdInput.fill('echo-chamber');
  await missionFragmentLabelInput.fill('Echo Chamber Preview');
  await missionRouteIdInput.fill('residency-night-preview');
  await missionAccentInput.fill('#ffcc88');
  await missionZoneMapUrlInput.fill('/manifests/residency-night-preview.zones.json');
  await missionWorldAssetLabelInput.fill('Residency Marble Night');
  await missionWorldFrameIdInput.fill('residency_night_map');

  await expect(missionIdInput).toHaveValue('residency-night-patrol');
  await expect(missionRouteUrlInput).toHaveValue('/robot-routes/residency-night-preview.json');
  await expect(missionJson).toContainText('"id": "residency-night-patrol"');
  await expect(missionJson).toContainText(
    '"label": "Residency Night Patrol Mission"'
  );
  await expect(missionJson).toContainText(
    '"description": "night patrol preview for browser publish"'
  );
  await expect(missionJson).toContainText('"fragmentId": "echo-chamber"');
  await expect(missionJson).toContainText(
    '"fragmentLabel": "Echo Chamber Preview"'
  );
  await expect(missionJson).toContainText('"accent": "#ffcc88"');
  await expect(missionJson).toContainText(
    '"routeUrl": "/robot-routes/residency-night-preview.json"'
  );
  await expect(missionJson).toContainText(
    '"zoneMapUrl": "/manifests/residency-night-preview.zones.json"'
  );
  await expect(missionJson).toContainText('"assetLabel": "Residency Marble Night"');
  await expect(missionJson).toContainText('"frameId": "residency_night_map"');
  await expect(routeJson).toContainText('"label": "Night Patrol Route"');
  await expect(routeJson).toContainText(
    '"description": "night patrol route for browser publish"'
  );
  await expect(routeJson).toContainText('"accent": "#7fd6ff"');
  await expect(missionDraftBundleJson).toContainText(
    '"id": "residency-night-patrol"'
  );
  await expect(missionDraftBundleJson).toContainText(
    '"label": "Night Patrol Route"'
  );
  await expect(missionDraftBundleJson).toContainText('"accent": "#7fd6ff"');
  await expect(missionArtifactPackJson).toContainText(
    '"missionId": "residency-night-patrol"'
  );
  await expect(missionArtifactPackJson).toContainText(
    '"fileName": "dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.json"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"id": "residency-night-patrol"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"fragmentId": "echo-chamber"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"fragmentLabel": "Echo Chamber Preview"'
  );
  await expect(publishedMissionPreviewJson).toContainText('"accent": "#ffcc88"');
  await expect(publishedMissionPreviewJson).toContainText(
    '"routeUrl": "/robot-routes/residency-night-preview.json"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"zoneMapUrl": "/manifests/residency-night-preview.zones.json"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"assetLabel": "Residency Marble Night"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"frameId": "residency_night_map"'
  );
  await expect(publishReportJson).toContainText(
    '"fileName": "residency-night-preview.json"'
  );
  await expect(publishReportJson).toContainText(
    '"assetLabel": "Residency Marble Night"'
  );
  await expect(publishReportJson).toContainText(
    '"url": "/manifests/residency-night-preview.zones.json"'
  );
  await expect(publishedMissionPreviewJson).toContainText(
    '"launchUrl": "/?robotMission=%2Frobot-missions%2Fresidency-night-patrol.mission.json"'
  );
  await expect(
    rightPanel.getByText('published file residency-night-patrol.mission.json', {
      exact: true
    })
  ).toBeVisible();
  await expect(
    rightPanel.getByText(
      'report file dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.publish-report.json',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    rightPanel.getByText(
      'preflight fragment echo-chamber / route id residency-night-preview / mission id residency-night-patrol',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    rightPanel.getByText('route file residency-night-preview.json', {
      exact: true
    })
  ).toBeVisible();
  await expect(
    rightPanel.getByText(
      'world asset Residency Marble Night / frame residency_night_map / zone /manifests/residency-night-preview.zones.json',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    rightPanel.getByText(
      'preflight mission Residency Night Patrol Mission / desc night patrol preview for browser publish / fragment label Echo Chamber Preview',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    rightPanel.getByText(
      'preflight route meta Night Patrol Route / accent #7fd6ff / startup robot / preset window / robot camera chase / scene window-talk',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    rightPanel.getByText(
      'preflight route desc night patrol route for browser publish / preset label Window / robot camera label Chase Camera / scene label Window Talk',
      { exact: true }
    )
  ).toBeVisible();
  await expect(missionPublishCommand).toContainText(
    '# preflight: Mission Warning'
  );
  await expect(missionValidateCommand).toContainText(
    '# preflight: Mission Warning'
  );
  await expect(missionValidateCommand).toContainText(
    '# zone: /manifests/residency-night-preview.zones.json'
  );
  await expect(missionReleaseCommand).toContainText(
    '# zone: /manifests/residency-night-preview.zones.json'
  );
  await expect(missionReleaseCommand).toContainText(
    'npm run release:robot-mission --'
  );
  await expect(missionPublishCommand).toContainText(
    '# zone: /manifests/residency-night-preview.zones.json'
  );
  await expect(missionPublishCommand).toContainText(
    '--bundle /absolute/path/to/dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.artifact-pack.json'
  );
  await expect(missionExportPanel.getByText('Mission Warning', { exact: true })).toBeVisible();
  await expect(missionExportPanel).toContainText(
    'mission fragment=echo-chamber / route fragment=residency'
  );

  const launchDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Launch', exact: true })
    .click();
  const launchDownload = await launchDownloadPromise;
  expect(launchDownload.suggestedFilename()).toBe(
    'residency-night-patrol.launch-url.txt'
  );

  const preflightDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Preflight', exact: true })
    .click();
  const preflightDownload = await preflightDownloadPromise;
  expect(preflightDownload.suggestedFilename()).toBe(
    'dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.preflight.txt'
  );

  const publishReportDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Publish Report', exact: true })
    .click();
  const publishReportDownload = await publishReportDownloadPromise;
  expect(publishReportDownload.suggestedFilename()).toBe(
    'dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.publish-report.json'
  );

  const validateDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Validate', exact: true })
    .click();
  const validateDownload = await validateDownloadPromise;
  expect(validateDownload.suggestedFilename()).toBe(
    'dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.artifact-pack.validate-command.txt'
  );

  const releaseDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Release', exact: true })
    .click();
  const releaseDownload = await releaseDownloadPromise;
  expect(releaseDownload.suggestedFilename()).toBe(
    'dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.artifact-pack.release-command.txt'
  );

  const publishDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Publish Command', exact: true })
    .click();
  const publishDownload = await publishDownloadPromise;
  expect(publishDownload.suggestedFilename()).toBe(
    'dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.artifact-pack.publish-command.txt'
  );

  const artifactPackDownloadPromise = page.waitForEvent('download');
  await rightPanel
    .getByRole('button', { name: 'Download Artifact Pack', exact: true })
    .click();
  const artifactPackDownload = await artifactPackDownloadPromise;
  expect(artifactPackDownload.suggestedFilename()).toBe(
    'dreamwalker-live-echo-chamber-residency-night-patrol-draft-bundle.artifact-pack.json'
  );
});

test('Robot mission zone map overrides fragment zone source', async ({ page }) => {
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());

    if (url.pathname === '/robot-missions/test-zone-override.mission.json') {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          version: 1,
          protocol: 'dreamwalker-robot-mission/v1',
          id: 'test-zone-override',
          label: 'Test Zone Override Mission',
          description: 'mission-specific zone map override',
          fragmentId: 'residency',
          fragmentLabel: 'Residency',
          routeUrl: '/robot-routes/residency-window-loop.json',
          zoneMapUrl: '/manifests/test-robotics-residency-override.zones.json',
          world: {
            assetLabel: 'Residency Marble',
            frameId: 'dreamwalker_map'
          }
        })
      });
      return;
    }

    if (url.pathname === '/manifests/test-robotics-residency-override.zones.json') {
      await route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          frameId: 'dreamwalker_map',
          resolution: 0.5,
          defaultCost: 0,
          bounds: {
            minX: -6,
            maxX: 6,
            minZ: 0,
            maxZ: 14
          },
          zones: [
            {
              id: 'mission-override-zone',
              label: 'Mission Override Zone',
              shape: 'rect',
              center: [2.4, 0, 8.2],
              size: [4.2, 4.2],
              cost: 7,
              tags: ['mission', 'override']
            }
          ]
        })
      });
      return;
    }

    await route.continue();
  });

  await page.goto('/?robotMission=/robot-missions/test-zone-override.mission.json');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const zonePanel = rightPanel.locator('.robot-zone-panel').first();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(zonePanel).toContainText('/manifests/test-robotics-residency-override.zones.json');
  await expect(zonePanel).toContainText('Mission Zone Map');
  await expect(currentZoneCard).toContainText('Mission Override Zone');
  await expect(zonePanel).toContainText('cost 7 / tags mission, override');
});

test('Public robot mission catalog applies sample mission', async ({ page }) => {
  await page.goto('/');

  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const publicMissionCard = leftPanel
    .locator('.state-card')
    .filter({ hasText: 'Residency Window Loop Mission' });
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);

  await expect(leftPanel.getByText('Dynamic Map Viewer Public Robot Mission Catalog', { exact: true })).toBeVisible();
  await expect(leftPanel.getByText('Residency Window Loop Mission', { exact: true })).toBeVisible();
  await expect(publicMissionCard).toContainText('Mission Ready');

  await publicMissionCard.getByRole('button', { name: 'Apply', exact: true }).click();
  await expect(
    leftPanel.getByText('/robot-missions/residency-window-loop.mission.json', { exact: true })
  ).toBeVisible();
  await expect(rightPanel.getByRole('heading', { name: 'Robot Mode', exact: true })).toBeVisible();
  await expect(poseCard).toContainText('x 2.40 / z 8.20');
  await expect(routeCard).toContainText('4 nodes');
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Chase Camera');
  await expect(
    leftPanel.getByRole('button', { name: '2. Window', exact: true })
  ).toHaveClass(/active/);
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText('"cameraPresetId": "window"');
  await expect(rightPanel.getByLabel('Mission Publish Command')).toContainText(
    '# preflight: Mission Ready'
  );
  await expect(rightPanel.getByLabel('Mission Publish Command')).toContainText(
    '--bundle /absolute/path/to/dreamwalker-live-residency-residency-window-loop-draft-bundle.artifact-pack.json'
  );
});

test('Robot mission draft bundle import accepts artifact pack file and saves to shelf', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const leftPanel = page.locator('.left-panel');
  const rightPanel = page.locator('.right-panel');
  const missionExportPanel = rightPanel
    .locator('.robot-route-export-panel')
    .filter({ hasText: 'Mission Export' })
    .first();
  const routeJson = rightPanel.locator('#robot-route-json');
  const missionDraftBundleImport = rightPanel.getByLabel('Mission Draft Bundle Import');
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const missionDraftBundleFileInput = missionExportPanel.locator('input.manifest-file-input');

  const importBundle = {
    version: 1,
    label: 'Imported Residency Mission Draft Bundle',
    fragmentId: 'residency',
    mission: {
      version: 1,
      protocol: 'dreamwalker-robot-mission/v1',
      id: 'imported-residency-mission',
      label: 'Imported Residency Mission',
      description: 'mission draft bundle import smoke',
      fragmentId: 'residency',
      fragmentLabel: 'Residency',
      accent: '#85e3e1',
      routeUrl: '/robot-routes/imported-residency.json',
      zoneMapUrl: '/manifests/imported-residency-preview.zones.json',
      launchUrl: '/?robotMission=%2Frobot-missions%2Fimported-residency.mission.json',
      cameraPresetId: 'gate',
      robotCameraId: 'top',
      streamSceneId: 'gate-recap',
      startupMode: 'robot',
      world: {
        assetLabel: 'Residency Marble',
        frameId: 'dreamwalker_map'
      }
    },
    route: {
      version: 1,
      protocol: 'dreamwalker-robot-route/v1',
      label: 'Imported Residency Route',
      fragmentId: 'residency',
      fragmentLabel: 'Residency',
      frameId: 'dreamwalker_map',
      world: {
        fragmentId: 'residency',
        fragmentLabel: 'Residency',
        assetLabel: 'Residency Marble',
        frameId: 'dreamwalker_map',
        zoneMapUrl: '/manifests/imported-residency-preview.zones.json'
      },
      pose: {
        position: [0, 0, 9.6],
        yawDegrees: 180
      },
      waypoint: {
        position: [0, 0, 11.4]
      },
      route: [
        [0, 0, 6.4],
        [0, 0, 8.2],
        [0, 0, 9.6]
      ]
    },
    zones: {
      frameId: 'dreamwalker_map',
      resolution: 0.5,
      defaultCost: 0,
      bounds: {
        minX: -6,
        maxX: 6,
        minZ: 0,
        maxZ: 14
      },
      zones: [
        {
          id: 'import-zone',
          label: 'Import Test Zone',
          shape: 'rect',
          center: [0, 0, 9.6],
          size: [4.2, 4.2],
          cost: 5,
          tags: ['import', 'safe']
        }
      ]
    }
  };

  const artifactPack = {
    version: 1,
    protocol: 'dreamwalker-robot-mission-artifact-pack/v1',
    label: 'Imported Residency Artifact Pack',
    missionId: 'imported-residency-mission',
    fragmentId: 'residency',
    files: [
      {
        kind: 'draft-bundle',
        fileName: 'imported-residency-draft-bundle.json',
        mediaType: 'application/json',
        content: JSON.stringify(importBundle, null, 2)
      },
      {
        kind: 'preflight-summary',
        fileName: 'imported-residency.preflight.txt',
        mediaType: 'text/plain',
        content: [
          'status: Mission Ready',
          'detail: route/world aligned',
          'missionId: imported-residency-mission'
        ].join('\n')
      },
      {
        kind: 'publish-report',
        fileName: 'imported-residency.publish-report.json',
        mediaType: 'application/json',
        content: JSON.stringify(
          {
            version: 1,
            protocol: 'dreamwalker-robot-mission-publish-report/v1',
            mission: {
              id: 'imported-residency-mission',
              url: '/robot-missions/imported-residency.mission.json'
            },
            route: {
              id: 'imported-residency',
              fileName: 'imported-residency.json'
            }
          },
          null,
          2
        )
      }
    ]
  };

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await rightPanel
    .getByRole('button', { name: 'Import Draft Bundle File To Shelf', exact: true })
    .click();
  await missionDraftBundleFileInput.setInputFiles({
    name: 'imported-residency-artifact-pack.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(artifactPack, null, 2))
  });

  await expect(rightPanel.getByRole('heading', { name: 'Robot Mode', exact: true })).toBeVisible();
  await expect(poseCard).toContainText('x 0.00 / z 9.60');
  await expect(routeCard).toContainText('3 nodes');
  await expect(rightPanel.getByText('Imported Residency Artifact Pack', { exact: true })).toBeVisible();
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Top View');
  await expect(
    rightPanel.getByRole('button', { name: 'Top View', exact: true })
  ).toHaveClass(/active/);
  await expect(
    leftPanel.getByRole('button', { name: '3. Gate', exact: true })
  ).toHaveClass(/active/);
  await expect(currentZoneCard).toContainText('Import Test Zone');
  await expect(rightPanel.locator('.robot-zone-panel').first()).toContainText(
    'cost 5 / tags import, safe'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"id": "imported-residency-mission"'
  );
  await expect(missionDraftBundleImport).toContainText('"label": "Imported Residency Mission"');
  await missionDraftBundleImport.fill(JSON.stringify(artifactPack, null, 2));
  await expect(
    rightPanel.getByText('artifact pack preview Imported Residency Artifact Pack / 3 files', {
      exact: true
    })
  ).toBeVisible();
  await expect(
    rightPanel.getByLabel('Import Artifact Preflight')
  ).toContainText('status: Mission Ready');
  await expect(
    rightPanel.getByText('artifact preflight file imported-residency.preflight.txt', {
      exact: true
    })
  ).toBeVisible();
  await expect(
    rightPanel.getByLabel('Import Artifact Publish Report')
  ).toContainText('"protocol": "dreamwalker-robot-mission-publish-report/v1"');
  await expect(
    rightPanel.getByLabel('Import Artifact Publish Report')
  ).toContainText('"url": "/robot-missions/imported-residency.mission.json"');
  await expect(
    rightPanel.getByText('artifact report file imported-residency.publish-report.json', {
      exact: true
    })
  ).toBeVisible();
});

test('Robot mission draft bundle shelf saves persists and reapplies snapshots', async ({ page }) => {
  test.setTimeout(90000);

  await page.goto('/');
  await page.evaluate(({ robotMissionDraftBundleShelfStorageKeyValue }) => {
    window.localStorage.removeItem(robotMissionDraftBundleShelfStorageKeyValue);
  }, {
    robotMissionDraftBundleShelfStorageKeyValue:
      robotMissionDraftBundleShelfStorageKey
  });
  await page.reload();

  const leftPanel = page.locator('.left-panel');
  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const routeJson = rightPanel.locator('#robot-route-json');
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const routeCard = rightPanel.locator('.state-grid .state-card').nth(2);
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);

  await page.goto('/?robotMission=/robot-missions/residency-window-loop.mission.json');
  await expect(rightPanel.getByRole('heading', { name: 'Robot Mode', exact: true })).toBeVisible();

  await rightPanel
    .getByLabel('Draft Snapshot Label', { exact: true })
    .fill('Residency Mission Draft 1');
  await rightPanel
    .getByRole('button', { name: 'Save Draft Snapshot', exact: true })
    .click();
  await expect(rightPanel.getByText('Residency Mission Draft 1', { exact: true })).toBeVisible();

  await leftPanel.getByRole('button', { name: '1. Foyer', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Front Camera', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Reset Robot Pose', exact: true }).click();
  await expect(poseCard).toContainText('x 0.00 / z 5.80');

  await page.reload();
  await expect(rightPanel.getByRole('heading', { name: 'Robot Mode', exact: true })).toBeVisible();
  await expect(rightPanel.getByText('Residency Mission Draft 1', { exact: true })).toBeVisible();
  const draftSnapshotCard = rightPanel
    .locator('.state-list .state-card')
    .filter({ hasText: 'Residency Mission Draft 1' })
    .first();
  await expect(
    draftSnapshotCard
      .getByText(
        'draft file dreamwalker-live-residency-residency-window-loop-draft-bundle.json',
        { exact: true }
      )
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByText('published file residency-window-loop.mission.json', {
        exact: true
      })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByText(
        'launch /?robotMission=%2Frobot-missions%2Fresidency-window-loop.mission.json',
        { exact: true }
      )
  ).toBeVisible();

  await draftSnapshotCard
    .getByLabel('Snapshot Mission ID', { exact: true })
    .fill('Residency Shelf Patrol');
  await draftSnapshotCard
    .getByLabel('Snapshot Mission Label', { exact: true })
    .fill('Residency Shelf Patrol Mission');
  await draftSnapshotCard
    .getByLabel('Snapshot Mission Description', { exact: true })
    .fill('shelf-edited preview mission');
  await draftSnapshotCard
    .getByLabel('Snapshot Route Label', { exact: true })
    .fill('Shelf Patrol Route');
  await draftSnapshotCard
    .getByLabel('Snapshot Route Description', { exact: true })
    .fill('shelf route preview');
  await draftSnapshotCard
    .getByLabel('Snapshot Route Accent', { exact: true })
    .fill('#ffb27a');
  await draftSnapshotCard
    .getByLabel('Snapshot Mission Fragment ID', { exact: true })
    .fill('residency');
  await draftSnapshotCard
    .getByLabel('Snapshot Mission Fragment Label', { exact: true })
    .fill('Residency Shelf');
  await draftSnapshotCard
    .getByLabel('Snapshot Mission Accent', { exact: true })
    .fill('#ff9e72');
  await draftSnapshotCard
    .getByLabel('Snapshot Mission Route ID', { exact: true })
    .fill('residency-shelf-preview');
  await expect(
    draftSnapshotCard.getByLabel('Snapshot Mission Route URL', { exact: true })
  ).toHaveValue('/robot-routes/residency-shelf-preview.json');
  await draftSnapshotCard
    .getByLabel('Snapshot Zone Map URL', { exact: true })
    .fill('/manifests/residency-shelf-preview.zones.json');
  await draftSnapshotCard
    .getByLabel('Snapshot World Asset Label', { exact: true })
    .fill('Residency Shelf Marble');
  await draftSnapshotCard
    .getByLabel('Snapshot World Frame ID', { exact: true })
    .fill('residency_shelf_map');
  await draftSnapshotCard
    .getByLabel('Snapshot Startup Mode', { exact: true })
    .selectOption('robot');
  await draftSnapshotCard
    .getByLabel('Snapshot Camera Preset', { exact: true })
    .selectOption('gate');
  await draftSnapshotCard
    .getByLabel('Snapshot Robot Camera', { exact: true })
    .selectOption('top');
  await draftSnapshotCard
    .getByLabel('Snapshot Stream Scene', { exact: true })
    .selectOption('gate-recap');
  await expect(
    draftSnapshotCard.getByText('published file residency-shelf-patrol.mission.json', {
      exact: true
    })
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'preflight fragment residency / route id residency-shelf-preview / mission id residency-shelf-patrol',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'world asset Residency Shelf Marble / frame residency_shelf_map / zone /manifests/residency-shelf-preview.zones.json',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'preflight mission Residency Shelf Patrol Mission / desc shelf-edited preview mission / fragment label Residency Shelf',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'preflight route meta Shelf Patrol Route / accent #ffb27a / startup robot / preset gate / robot camera top / scene gate-recap',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'preflight route desc shelf route preview / preset label Gate / robot camera label Top View / scene label Gate Recap',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'launch /?robotMission=%2Frobot-missions%2Fresidency-shelf-patrol.mission.json',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'fragment residency / label Residency Shelf',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'route file residency-shelf-preview.json',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'route /robot-routes/residency-shelf-preview.json / zone /manifests/residency-shelf-preview.zones.json',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'route meta Shelf Patrol Route / accent #ffb27a',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'accent #ff9e72',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'world Residency Shelf Marble / frame residency_shelf_map',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.getByText(
      'effective robot / preset gate / robot camera top / scene gate-recap',
      { exact: true }
    )
  ).toBeVisible();
  await expect(
    draftSnapshotCard.locator('.health-badge').first()
  ).toBeVisible();

  await draftSnapshotCard
    .getByRole('button', { name: 'Apply', exact: true })
    .click();

  await expect(poseCard).toContainText('x 2.40 / z 8.20');
  await expect(routeCard).toContainText('4 nodes');
  await expect(rightPanel.locator('.robot-zone-panel').first()).toContainText(
    '/manifests/residency-shelf-preview.zones.json'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"accent": "#ff9e72"'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"fragmentId": "residency"'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"fragmentLabel": "Residency Shelf"'
  );
  await expect(routeJson).toContainText('"label": "Shelf Patrol Route"');
  await expect(routeJson).toContainText('"description": "shelf route preview"');
  await expect(routeJson).toContainText('"accent": "#ffb27a"');
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"routeUrl": "/robot-routes/residency-shelf-preview.json"'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"zoneMapUrl": "/manifests/residency-shelf-preview.zones.json"'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"assetLabel": "Residency Shelf Marble"'
  );
  await expect(rightPanel.getByLabel('Robot Mission JSON')).toContainText(
    '"frameId": "residency_shelf_map"'
  );
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Top View');
  await expect(
    rightPanel.getByRole('button', { name: 'Top View', exact: true })
  ).toHaveClass(/active/);
  await expect(
    leftPanel.getByRole('button', { name: '3. Gate', exact: true })
  ).toHaveClass(/active/);
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Bundle', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Mission', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Download Bundle', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Preview', exact: true })
  ).toBeVisible();
  const missionDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Mission', exact: true })
    .click();
  const missionDownload = await missionDownloadPromise;
  expect(missionDownload.suggestedFilename()).toBe(
    'residency-shelf-patrol.robot-mission.json'
  );
  const previewDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Preview', exact: true })
    .click();
  const previewDownload = await previewDownloadPromise;
  expect(previewDownload.suggestedFilename()).toBe(
    'residency-shelf-patrol.mission.json'
  );
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Launch', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Preflight', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Report', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Validate', exact: true })
  ).toBeVisible();
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Release', exact: true })
  ).toBeVisible();
  const launchDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Launch', exact: true })
    .click();
  const launchDownload = await launchDownloadPromise;
  expect(launchDownload.suggestedFilename()).toBe(
    'residency-shelf-patrol.launch-url.txt'
  );
  const preflightDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Preflight', exact: true })
    .click();
  const preflightDownload = await preflightDownloadPromise;
  expect(preflightDownload.suggestedFilename()).toBe(
    'dreamwalker-live-residency-residency-shelf-patrol-draft-bundle.preflight.txt'
  );
  const reportDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Report', exact: true })
    .click();
  const reportDownload = await reportDownloadPromise;
  expect(reportDownload.suggestedFilename()).toBe(
    'dreamwalker-live-residency-residency-shelf-patrol-draft-bundle.publish-report.json'
  );
  const validateDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Validate', exact: true })
    .click();
  const validateDownload = await validateDownloadPromise;
  expect(validateDownload.suggestedFilename()).toBe(
    'dreamwalker-live-residency-residency-shelf-patrol-draft-bundle.artifact-pack.validate-command.txt'
  );
  const releaseDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Release', exact: true })
    .click();
  const releaseDownload = await releaseDownloadPromise;
  expect(releaseDownload.suggestedFilename()).toBe(
    'dreamwalker-live-residency-residency-shelf-patrol-draft-bundle.artifact-pack.release-command.txt'
  );
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Publish', exact: true })
  ).toBeVisible();
  const publishDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Publish', exact: true })
    .click();
  const publishDownload = await publishDownloadPromise;
  expect(publishDownload.suggestedFilename()).toBe(
    'dreamwalker-live-residency-residency-shelf-patrol-draft-bundle.artifact-pack.publish-command.txt'
  );
  await expect(
    draftSnapshotCard
      .getByRole('button', { name: 'Copy Artifacts', exact: true })
  ).toBeVisible();
  const artifactsDownloadPromise = page.waitForEvent('download');
  await draftSnapshotCard
    .getByRole('button', { name: 'Download Artifacts', exact: true })
    .click();
  const artifactsDownload = await artifactsDownloadPromise;
  expect(artifactsDownload.suggestedFilename()).toBe(
    'dreamwalker-live-residency-residency-shelf-patrol-draft-bundle.artifact-pack.json'
  );
});

test('Robot mode loads semantic zones and updates current zone', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const zoneMapCard = rightPanel.locator('.state-grid .state-card').nth(5);
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const zonePanel = rightPanel.locator('.robot-zone-panel').first();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();

  await expect(zoneMapCard).toContainText('3 zones');
  await expect(currentZoneCard).toContainText('Residency Stage');
  await expect(zonePanel).toContainText('cost 15 / tags stream, safe');

  for (let index = 0; index < 5; index += 1) {
    await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();
  }

  await expect(currentZoneCard).toContainText('Gate Lane');
  await expect(zonePanel).toContainText('cost 45 / tags gate, transition');
  await rightPanel.getByRole('button', { name: 'Top View', exact: true }).click();
  await expect(page.locator('.semantic-zone-overlay')).toContainText('Gate Lane');
  await expect(page.locator('.semantic-zone-surface-overlay .semantic-zone-surface-shape')).toHaveCount(2);
  await expect(page.locator('.semantic-zone-surface-overlay .semantic-zone-surface-shape.active')).toHaveCount(1);
});

test('Robot mode semantic zone workspace edits and persists current fragment zones', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);
  const zoneStatusPanel = rightPanel.locator('.robot-zone-panel').first();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();

  await rightPanel.locator('#semantic-zone-label-residency-stage').fill('Stage Alpha');
  await rightPanel.locator('#semantic-zone-cost-residency-stage').fill('33');

  await expect(currentZoneCard).toContainText('Stage Alpha');
  await expect(zoneStatusPanel).toContainText('cost 33 / tags stream, safe');

  await rightPanel.getByRole('button', { name: 'Save Zone Workspace', exact: true }).click();
  await page.reload();
  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();

  await expect(currentZoneCard).toContainText('Stage Alpha');
  await expect(zoneStatusPanel).toContainText('cost 33 / tags stream, safe');
});

test('Robot mode semantic zone quick actions use robot pose and duplication', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const zoneMapCard = rightPanel.locator('.state-grid .state-card').nth(5);
  const stageCard = rightPanel.locator('.zone-editor-card').filter({ hasText: 'Residency Stage' });

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();

  await stageCard.getByRole('button', { name: 'Zone <- Robot', exact: true }).click();
  await expect(rightPanel.locator('#semantic-zone-center-z-residency-stage')).toHaveValue('6.6');

  await rightPanel.getByRole('button', { name: 'Add Zone At Robot', exact: true }).click();
  await expect(zoneMapCard).toContainText('4 zones');

  await stageCard.getByRole('button', { name: 'Duplicate', exact: true }).click();
  await expect(zoneMapCard).toContainText('5 zones');
});

test('Robot mode semantic zone batch ops derive zones from route and refit bounds', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const zoneMapCard = rightPanel.locator('.state-grid .state-card').nth(5);
  const currentZoneCard = rightPanel.locator('.state-grid .state-card').nth(6);

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();
  await rightPanel.getByRole('button', { name: 'Forward', exact: true }).click();

  await rightPanel.getByRole('button', { name: 'Add Zones From Route', exact: true }).click();
  await expect(zoneMapCard).toContainText('6 zones');

  await rightPanel.locator('#semantic-zone-center-x-residency-stage').fill('8');
  await rightPanel.getByRole('button', { name: 'Fit Bounds To Zones', exact: true }).click();
  await expect(rightPanel.locator('#semantic-zone-max-x')).toHaveValue('11.3');

  await rightPanel.getByRole('button', { name: 'Clear All Zones', exact: true }).click();
  await expect(zoneMapCard).toContainText('0 zones');
  await expect(currentZoneCard).toContainText('Outside Map');
});

test('Robot mode renders nav cost panel from semantic zones', async ({ page }) => {
  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const navPanel = rightPanel.locator('.robot-nav-panel');

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();

  await expect(navPanel).toBeVisible();
  await expect(navPanel.locator('.robot-nav-zone')).toHaveCount(3);
  await expect(navPanel.locator('.robot-nav-zone.active')).toHaveCount(1);
  await expect(navPanel.locator('.robot-nav-robot')).toBeVisible();

  await rightPanel.getByRole('button', { name: 'Drop Waypoint', exact: true }).click();
  await expect(navPanel.locator('.robot-nav-waypoint')).toBeVisible();
});

test('Robot mode supports gamepad teleop and camera cycling', async ({ page }) => {
  await page.addInitScript(() => {
    const defaultPad = {
      id: 'Virtual Pad',
      index: 0,
      connected: true,
      mapping: 'standard',
      axes: [0, 0, 0, 0],
      buttons: Array.from({ length: 16 }, () => ({
        pressed: false,
        touched: false,
        value: 0
      }))
    };
    const state = {
      pad: null
    };

    function clonePad(nextPad) {
      return nextPad
        ? {
            ...nextPad,
            axes: [...(nextPad.axes ?? defaultPad.axes)],
            buttons: (nextPad.buttons ?? defaultPad.buttons).map((button) => ({ ...button }))
          }
        : null;
    }

    Object.defineProperty(navigator, 'getGamepads', {
      configurable: true,
      value: () => (state.pad ? [state.pad] : [])
    });

    window.__setDreamwalkerGamepad = (pad = defaultPad) => {
      state.pad = clonePad({ ...defaultPad, ...pad });
    };

    window.__setDreamwalkerGamepadAxes = (axes) => {
      if (!state.pad) {
        return;
      }

      state.pad = clonePad({
        ...state.pad,
        axes
      });
    };

    window.__setDreamwalkerGamepadButton = (index, pressed) => {
      if (!state.pad) {
        return;
      }

      const buttons = state.pad.buttons.map((button, buttonIndex) =>
        buttonIndex === index
          ? {
              pressed,
              touched: pressed,
              value: pressed ? 1 : 0
            }
          : button
      );

      state.pad = clonePad({
        ...state.pad,
        buttons
      });
    };
  });

  await page.goto('/');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const poseCard = rightPanel.locator('.state-grid .state-card').first();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await page.evaluate(() => {
    window.__setDreamwalkerGamepad({
      id: 'Virtual Pad',
      mapping: 'standard'
    });
  });

  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('gamepad Virtual Pad / standard');

  const initialPose = await poseCard.textContent();
  await page.evaluate(() => {
    window.__setDreamwalkerGamepadAxes([0, -1, 0, 0]);
  });
  await page.waitForTimeout(240);
  await page.evaluate(() => {
    window.__setDreamwalkerGamepadAxes([0, 0, 0, 0]);
  });
  await expect(poseCard).not.toHaveText(initialPose ?? '');

  await page.evaluate(() => {
    window.__setDreamwalkerGamepadButton(0, true);
  });
  await page.waitForTimeout(120);
  await page.evaluate(() => {
    window.__setDreamwalkerGamepadButton(0, false);
  });
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('m ahead');

  await page.evaluate(() => {
    window.__setDreamwalkerGamepadButton(5, true);
  });
  await page.waitForTimeout(120);
  await page.evaluate(() => {
    window.__setDreamwalkerGamepadButton(5, false);
  });
  await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Chase Camera');
});

test('Robot bridge exchanges state snapshots and inbound commands', async ({ page }) => {
  await page.goto('/?robotBridge=1');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const poseCard = rightPanel.locator('.state-grid .state-card').first();
  const bridgeCard = rightPanel.locator('.state-grid .state-card').nth(3);

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(bridgeCard).toContainText('Connected');

  const bridgeSocket = new WebSocket('ws://127.0.0.1:8790/robotics');
  await waitForBridgeOpen(bridgeSocket);

  try {
    const readyMessage = await waitForBridgeMessage(
      bridgeSocket,
      (message) => message.type === 'bridge-ready'
    );
    expect(readyMessage.protocol).toBe(robotBridgeProtocolId);

    bridgeSocket.send(JSON.stringify({ type: 'request-state' }));
    const snapshot = await waitForBridgeMessage(
      bridgeSocket,
      (message) => message.type === 'robot-state'
    );
    expect(snapshot.protocol).toBe(robotBridgeProtocolId);
    expect(snapshot.source).toBe('dreamwalker-live');
    expect(snapshot.fragmentId).toBe('residency');
    expect(snapshot.pose.position[2]).toBeCloseTo(5.8, 2);

    bridgeSocket.send(
      JSON.stringify({
        type: 'set-pose',
        pose: {
          position: [1.25, 0, 6.4],
          yawDegrees: 90
        },
        resetRoute: true
      })
    );
    await expect(poseCard).toContainText('x 1.25 / z 6.40');

    bridgeSocket.send(
      JSON.stringify({
        type: 'set-waypoint',
        position: [1.25, 0, 8.6]
      })
    );
    await expect(rightPanel.locator('.robot-camera-panel')).toContainText('m ahead');
  } finally {
    bridgeSocket.close();
  }
});

test('Robot mode sim2real panel renders websocket previews', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');
    await expect(sim2realPanel).toContainText('backend gsplat');

    await sim2realPanel.getByRole('button', { name: 'Render Current Pose' }).click();
    await waitForCondition(
      () => sim2realServer.messages.some((message) => message.type === 'render'),
      'sim2real render request'
    );

    const renderRequest = findLastMessage(
      sim2realServer.messages,
      (message) => message.type === 'render'
    );

    expect(renderRequest).not.toBeNull();
    expect(renderRequest.protocol).toBe(sim2realQueryProtocolId);
    expect(renderRequest.width).toBe(64);
    expect(renderRequest.height).toBe(48);
    expect(renderRequest.pose.position[2]).toBeCloseTo(5.8, 1);
    expect(renderRequest.pose.orientation).toHaveLength(4);

    await expect(sim2realPanel).toContainText('64 x 48');
    await expect(sim2realPanel.getByAltText('Sim2Real RGB Preview')).toHaveAttribute(
      'src',
      /data:image\/jpeg;base64,/
    );
    await expect(sim2realPanel.getByAltText('Sim2Real Depth Preview')).toHaveAttribute(
      'src',
      /data:image\/png;base64,/
    );
  } finally {
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel captures route bundles as downloadable JSON', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await waitForCondition(
      () => sim2realServer.messages.some((message) => message.type === 'render'),
      'sim2real route capture request'
    );

    await expect(sim2realPanel).toContainText('1 frames');
    await expect(sim2realPanel).toContainText('residency / 64 x 48');

    const downloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const download = await downloadPromise;
    const downloadPath = await download.path();
    const bundle = JSON.parse(await fs.readFile(downloadPath, 'utf8'));

    expect(bundle.protocol).toBe('dreamwalker-sim2real-capture/v1');
    expect(bundle.type).toBe('route-capture-bundle');
    expect(bundle.fragmentId).toBe('residency');
    expect(bundle.endpoint).toBe(sim2realServer.url);
    expect(bundle.captures).toHaveLength(1);
    expect(bundle.captures[0].response.type).toBe('render-result');
    expect(bundle.captures[0].response.width).toBe(64);
    expect(bundle.captures[0].response.height).toBe(48);

    await sim2realPanel.locator('#sim2real-capture-shelf-label').fill('Residency Route Dataset A');
    await sim2realPanel.getByRole('button', { name: 'Save Capture Snapshot' }).click();
    await expect(sim2realPanel).toContainText('Residency Route Dataset A');
    await expect(sim2realPanel).toContainText('saved');

    await sim2realPanel.getByRole('button', { name: 'Clear Capture', exact: true }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('Idle');
    await expect(sim2realPanel).toContainText('Residency Route Dataset A');

    await sim2realPanel.getByRole('button', { name: 'Preview Last Frame' }).click();
    await expect(sim2realPanel).toContainText('Residency Route Dataset A / Preview');
  } finally {
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel benchmarks imported localization trajectories', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');
    const navPanel = rightPanel.locator('.robot-nav-panel');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();

    await sim2realPanel.locator('#sim2real-benchmark-import').setInputFiles(captureDownloadPath);
    await expect(benchmarkCard).toContainText('1 matched poses');
    await expect(benchmarkCard).toContainText('ATE RMSE');
    await expect(benchmarkCard).toContainText('0.000 m');
    await expect(benchmarkCard).toContainText('matching 1 / gt 1 / estimate 1');
    await expect(navPanel).toContainText('benchmark ATE 0.000 m / match 1');
    await expect(navPanel.locator('.robot-nav-benchmark-ground-truth-node')).toHaveCount(1);
    await expect(navPanel.locator('.robot-nav-benchmark-estimate-node')).toHaveCount(1);

    const benchmarkDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Benchmark Report' }).click();
    const benchmarkDownload = await benchmarkDownloadPromise;
    const benchmarkDownloadPath = await benchmarkDownload.path();
    const report = JSON.parse(await fs.readFile(benchmarkDownloadPath, 'utf8'));

    expect(report.protocol).toBe('dreamwalker-localization-benchmark/v1');
    expect(report.type).toBe('localization-benchmark-report');
    expect(report.matching.matchedCount).toBe(1);
    expect(report.metrics.ateRmseMeters).toBe(0);
    expect(report.metrics.rpeTranslationRmseMeters).toBeNull();
  } finally {
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel imports TUM style localization text trajectories', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');
    const navPanel = rightPanel.locator('.robot-nav-panel');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const captureBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const capturePose = captureBundle.captures[0].pose;
    const yawRadians = (capturePose.yawDegrees * Math.PI) / 180;
    const tumTrajectory = [
      `0 ${capturePose.position[0]} ${capturePose.position[1]} ${capturePose.position[2]} 0 ${Math.sin(yawRadians / 2)} 0 ${Math.cos(yawRadians / 2)}`
    ].join('\n');

    await sim2realPanel.locator('#sim2real-benchmark-import').setInputFiles({
      name: 'orbslam3_camera_trajectory.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(tumTrajectory, 'utf8')
    });

    await expect(benchmarkCard).toContainText('orbslam3_camera_trajectory');
    await expect(benchmarkCard).toContainText('1 matched poses');
    await expect(benchmarkCard).toContainText('0.000 m');
    await expect(navPanel).toContainText('benchmark ATE 0.000 m / match 1');
    await expect(navPanel.locator('.robot-nav-benchmark-ground-truth-node')).toHaveCount(1);
    await expect(navPanel.locator('.robot-nav-benchmark-estimate-node')).toHaveCount(1);

    const benchmarkDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Benchmark Report' }).click();
    const benchmarkDownload = await benchmarkDownloadPromise;
    const benchmarkDownloadPath = await benchmarkDownload.path();
    const report = JSON.parse(await fs.readFile(benchmarkDownloadPath, 'utf8'));

    expect(report.estimate.label).toBe('orbslam3_camera_trajectory');
    expect(report.estimate.sourceType).toBe('tum-trajectory-text');
    expect(report.metrics.ateRmseMeters).toBe(0);
  } finally {
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel auto-aligns localization benchmark by timestamps', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const templateBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const templateCapture = templateBundle.captures[0];
    const templatePosition = templateCapture.pose.position;
    const templateYawDegrees = templateCapture.pose.yawDegrees;
    const offsetPosition = [templatePosition[0] + 2, templatePosition[1], templatePosition[2]];
    const yawRadians = (templateYawDegrees * Math.PI) / 180;
    const syntheticBundle = {
      ...templateBundle,
      capturedAt: '2026-04-02T00:00:01.000Z',
      fragmentLabel: 'Residency Timestamp GT',
      route: [
        {
          index: 0,
          position: [...templatePosition],
          yawDegrees: templateYawDegrees,
          relativeTimeSeconds: 0
        },
        {
          index: 1,
          position: [...offsetPosition],
          yawDegrees: templateYawDegrees,
          relativeTimeSeconds: 1
        }
      ],
      captures: [
        {
          ...templateCapture,
          index: 1,
          label: 'gt:late',
          capturedAt: '2026-04-02T00:00:01.000Z',
          relativeTimeSeconds: 1,
          pose: {
            position: [...offsetPosition],
            yawDegrees: templateYawDegrees
          }
        },
        {
          ...templateCapture,
          index: 0,
          label: 'gt:early',
          capturedAt: '2026-04-02T00:00:00.000Z',
          relativeTimeSeconds: 0,
          pose: {
            position: [...templatePosition],
            yawDegrees: templateYawDegrees
          }
        }
      ]
    };
    const tumTrajectory = [
      `0 ${templatePosition[0]} ${templatePosition[1]} ${templatePosition[2]} 0 ${Math.sin(yawRadians / 2)} 0 ${Math.cos(yawRadians / 2)}`,
      `1 ${offsetPosition[0]} ${offsetPosition[1]} ${offsetPosition[2]} 0 ${Math.sin(yawRadians / 2)} 0 ${Math.cos(yawRadians / 2)}`
    ].join('\n');

    await sim2realPanel.locator('#sim2real-capture-shelf-label').fill('Timestamp Ground Truth Bundle');
    await sim2realPanel.locator('#sim2real-capture-import').setInputFiles({
      name: 'timestamp-ground-truth.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(syntheticBundle), 'utf8')
    });

    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('2 frames');
    await benchmarkCard.locator('#sim2real-benchmark-alignment').selectOption('index');
    await sim2realPanel.locator('#sim2real-benchmark-import').setInputFiles({
      name: 'timestamp-aligned-estimate.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(tumTrajectory, 'utf8')
    });

    await expect(benchmarkCard).toContainText('2.000 m');
    await benchmarkCard.locator('#sim2real-benchmark-alignment').selectOption('auto');
    await expect(benchmarkCard).toContainText('alignment timestamp');
    await expect(benchmarkCard).toContainText('0.000 m');
    await expect(benchmarkCard).toContainText('time delta mean 0.000 s / max 0.000 s');

    const benchmarkDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Benchmark Report' }).click();
    const benchmarkDownload = await benchmarkDownloadPromise;
    const benchmarkDownloadPath = await benchmarkDownload.path();
    const report = JSON.parse(await fs.readFile(benchmarkDownloadPath, 'utf8'));

    expect(report.requestedAlignment).toBe('auto');
    expect(report.alignment).toBe('timestamp');
    expect(report.matching.matchedCount).toBe(2);
    expect(report.metrics.ateRmseMeters).toBe(0);
    expect(report.metrics.timeDelta.rmse).toBe(0);
  } finally {
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel interpolates sparse timestamp trajectories', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const templateBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const templateCapture = templateBundle.captures[0];
    const startPosition = templateCapture.pose.position;
    const middlePosition = [startPosition[0] + 1, startPosition[1], startPosition[2]];
    const endPosition = [startPosition[0] + 2, startPosition[1], startPosition[2]];
    const yawDegrees = templateCapture.pose.yawDegrees;
    const yawRadians = (yawDegrees * Math.PI) / 180;
    const sparseGroundTruthBundle = {
      ...templateBundle,
      capturedAt: '2026-04-02T00:00:00.000Z',
      fragmentLabel: 'Residency Sparse Timestamp GT',
      route: [
        {
          index: 0,
          position: [...startPosition],
          yawDegrees,
          relativeTimeSeconds: 0
        },
        {
          index: 1,
          position: [...middlePosition],
          yawDegrees,
          relativeTimeSeconds: 0.5
        },
        {
          index: 2,
          position: [...endPosition],
          yawDegrees,
          relativeTimeSeconds: 1
        }
      ],
      captures: [
        {
          ...templateCapture,
          index: 0,
          label: 'gt:start',
          capturedAt: '2026-04-02T00:00:00.000Z',
          relativeTimeSeconds: 0,
          pose: {
            position: [...startPosition],
            yawDegrees
          }
        },
        {
          ...templateCapture,
          index: 1,
          label: 'gt:mid',
          capturedAt: '2026-04-02T00:00:00.500Z',
          relativeTimeSeconds: 0.5,
          pose: {
            position: [...middlePosition],
            yawDegrees
          }
        },
        {
          ...templateCapture,
          index: 2,
          label: 'gt:end',
          capturedAt: '2026-04-02T00:00:01.000Z',
          relativeTimeSeconds: 1,
          pose: {
            position: [...endPosition],
            yawDegrees
          }
        }
      ]
    };
    const sparseTumTrajectory = [
      `0 ${startPosition[0]} ${startPosition[1]} ${startPosition[2]} 0 ${Math.sin(yawRadians / 2)} 0 ${Math.cos(yawRadians / 2)}`,
      `1 ${endPosition[0]} ${endPosition[1]} ${endPosition[2]} 0 ${Math.sin(yawRadians / 2)} 0 ${Math.cos(yawRadians / 2)}`
    ].join('\n');

    await sim2realPanel.locator('#sim2real-capture-import').setInputFiles({
      name: 'sparse-ground-truth.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(sparseGroundTruthBundle), 'utf8')
    });

    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('3 frames');
    await benchmarkCard.locator('#sim2real-benchmark-alignment').selectOption('index');
    await sim2realPanel.locator('#sim2real-benchmark-import').setInputFiles({
      name: 'sparse-estimate.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(sparseTumTrajectory, 'utf8')
    });

    await expect(benchmarkCard).toContainText('0.707 m');
    await expect(benchmarkCard).toContainText('matching 2 / gt 3 / estimate 2 / gt remainder 1');

    await benchmarkCard.locator('#sim2real-benchmark-alignment').selectOption('auto');
    await expect(benchmarkCard).toContainText('alignment timestamp / interpolation linear');
    await expect(benchmarkCard).toContainText('matching 3 / gt 3 / estimate 2 / interpolated 1');
    await expect(benchmarkCard).toContainText('0.000 m');
    await expect(benchmarkCard).toContainText('time delta mean 0.000 s / max 0.000 s');

    const benchmarkDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Benchmark Report' }).click();
    const benchmarkDownload = await benchmarkDownloadPromise;
    const benchmarkDownloadPath = await benchmarkDownload.path();
    const report = JSON.parse(await fs.readFile(benchmarkDownloadPath, 'utf8'));

    expect(report.alignment).toBe('timestamp');
    expect(report.estimate.interpolationMode).toBe('linear');
    expect(report.matching.matchedCount).toBe(3);
    expect(report.matching.interpolatedCount).toBe(1);
    expect(report.metrics.ateRmseMeters).toBe(0);
  } finally {
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel monitors live localization estimate streams', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();
  const liveLocalizationServer = await createLiveLocalizationMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const templateBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const templateCapture = templateBundle.captures[0];
    const startPosition = templateCapture.pose.position;
    const endPosition = [startPosition[0] + 1.5, startPosition[1], startPosition[2]];
    const yawDegrees = templateCapture.pose.yawDegrees;
    const yawRadians = (yawDegrees * Math.PI) / 180;
    const liveGroundTruthBundle = {
      ...templateBundle,
      capturedAt: '2026-04-02T00:00:00.000Z',
      fragmentLabel: 'Residency Live Monitor GT',
      route: [
        {
          index: 0,
          position: [...startPosition],
          yawDegrees,
          relativeTimeSeconds: 0
        },
        {
          index: 1,
          position: [...endPosition],
          yawDegrees,
          relativeTimeSeconds: 1
        }
      ],
      captures: [
        {
          ...templateCapture,
          index: 0,
          label: 'gt:start',
          capturedAt: '2026-04-02T00:00:00.000Z',
          relativeTimeSeconds: 0,
          pose: {
            position: [...startPosition],
            yawDegrees
          }
        },
        {
          ...templateCapture,
          index: 1,
          label: 'gt:end',
          capturedAt: '2026-04-02T00:00:01.000Z',
          relativeTimeSeconds: 1,
          pose: {
            position: [...endPosition],
            yawDegrees
          }
        }
      ]
    };

    await sim2realPanel.locator('#sim2real-capture-import').setInputFiles({
      name: 'live-ground-truth.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(liveGroundTruthBundle), 'utf8')
    });

    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('2 frames');
    await benchmarkCard.locator('#sim2real-live-estimate-url').fill(liveLocalizationServer.url);
    await benchmarkCard
      .getByRole('button', { name: 'Connect Live Monitor', exact: true })
      .click();
    await expect(benchmarkCard).toContainText('live monitor Connected');

    await liveLocalizationServer.send({
      type: 'reset',
      label: 'ORB-SLAM3 Live'
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Live',
      pose: {
        position: [...startPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 0
      }
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Live',
      pose: {
        position: [...endPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 1
      }
    });

    await expect(benchmarkCard).toContainText('estimate ORB-SLAM3 Live / 2 poses / source auto-live');
    await expect(benchmarkCard).toContainText('2 matched poses');
    await expect(benchmarkCard).toContainText('0.000 m');
    await expect(benchmarkCard).toContainText('messages 2');

    const benchmarkDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Benchmark Report' }).click();
    const benchmarkDownload = await benchmarkDownloadPromise;
    const benchmarkDownloadPath = await benchmarkDownload.path();
    const report = JSON.parse(await fs.readFile(benchmarkDownloadPath, 'utf8'));

    expect(report.estimate.label).toBe('ORB-SLAM3 Live');
    expect(report.estimate.sourceType).toBe('live-stream');
    expect(report.matching.matchedCount).toBe(2);
    expect(report.metrics.ateRmseMeters).toBe(0);

    await benchmarkCard
      .getByRole('button', { name: 'Disconnect Live Monitor', exact: true })
      .click();
    await expect(benchmarkCard).toContainText('live monitor Disabled');
  } finally {
    await liveLocalizationServer.close();
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel saves and reloads localization benchmark runs', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();
  const liveLocalizationServer = await createLiveLocalizationMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const templateBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const templateCapture = templateBundle.captures[0];
    const startPosition = templateCapture.pose.position;
    const endPosition = [startPosition[0] + 1.5, startPosition[1], startPosition[2]];
    const yawDegrees = templateCapture.pose.yawDegrees;
    const yawRadians = (yawDegrees * Math.PI) / 180;
    const groundTruthBundle = {
      ...templateBundle,
      capturedAt: '2026-04-02T00:00:00.000Z',
      fragmentLabel: 'Residency Run Shelf GT',
      route: [
        {
          index: 0,
          position: [...startPosition],
          yawDegrees,
          relativeTimeSeconds: 0
        },
        {
          index: 1,
          position: [...endPosition],
          yawDegrees,
          relativeTimeSeconds: 1
        }
      ],
      captures: [
        {
          ...templateCapture,
          index: 0,
          label: 'gt:start',
          capturedAt: '2026-04-02T00:00:00.000Z',
          relativeTimeSeconds: 0,
          pose: {
            position: [...startPosition],
            yawDegrees
          }
        },
        {
          ...templateCapture,
          index: 1,
          label: 'gt:end',
          capturedAt: '2026-04-02T00:00:01.000Z',
          relativeTimeSeconds: 1,
          pose: {
            position: [...endPosition],
            yawDegrees
          }
        }
      ]
    };

    await sim2realPanel.locator('#sim2real-capture-import').setInputFiles({
      name: 'run-shelf-ground-truth.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(groundTruthBundle), 'utf8')
    });

    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('2 frames');
    await benchmarkCard.locator('#sim2real-live-estimate-url').fill(liveLocalizationServer.url);
    await benchmarkCard
      .getByRole('button', { name: 'Connect Live Monitor', exact: true })
      .click();
    await expect(benchmarkCard).toContainText('live monitor Connected');

    await liveLocalizationServer.send({
      type: 'reset',
      label: 'ORB-SLAM3 Live'
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Live',
      pose: {
        position: [...startPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 0
      }
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Live',
      pose: {
        position: [...endPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 1
      }
    });

    await expect(benchmarkCard).toContainText('estimate ORB-SLAM3 Live / 2 poses / source auto-live');
    await expect(benchmarkCard).toContainText('2 matched poses');
    await expect(benchmarkCard).toContainText('0.000 m');

    await benchmarkCard
      .locator('#sim2real-localization-run-shelf-label')
      .fill('ORB-SLAM3 Live / Run A');
    await benchmarkCard.getByRole('button', { name: 'Save Benchmark Run' }).click();
    await expect(benchmarkCard).toContainText('1 saved runs');

    const shiftedStartPosition = [startPosition[0] + 0.5, startPosition[1], startPosition[2]];
    const shiftedEndPosition = [endPosition[0] + 0.5, endPosition[1], endPosition[2]];

    await liveLocalizationServer.send({
      type: 'reset',
      label: 'ORB-SLAM3 Live Offset'
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Live Offset',
      pose: {
        position: [...shiftedStartPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 0
      }
    });
    await expect(benchmarkCard).toContainText('estimate ORB-SLAM3 Live Offset / 1 poses / source auto-live');
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Live Offset',
      pose: {
        position: [...shiftedEndPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 1
      }
    });

    await expect(benchmarkCard).toContainText('estimate ORB-SLAM3 Live Offset / 2 poses / source auto-live');
    await expect(benchmarkCard).toContainText('0.500 m');
    await benchmarkCard
      .locator('#sim2real-localization-run-shelf-label')
      .fill('ORB-SLAM3 Live / Run B');
    await benchmarkCard.getByRole('button', { name: 'Save Benchmark Run' }).click();
    await expect(benchmarkCard).toContainText('2 saved runs');
    const imageBenchmarkReportRunA = {
      protocol: 'dreamwalker-localization-image-benchmark/v1',
      type: 'localization-image-benchmark-report',
      createdAt: '2026-04-02T00:10:00.000Z',
      endpoint: sim2realServer.url,
      alignment: 'timestamp',
      groundTruth: {
        fragmentId: 'residency',
        fragmentLabel: 'Residency Run Shelf GT'
      },
      estimate: {
        label: 'ORB-SLAM3 Live',
        sourceType: 'live-stream'
      },
      matching: {
        matchedCount: 2
      },
      metrics: {
        summary: {
          lpips: { mean: 0.123 },
          psnr: { mean: 28.5 },
          ssim: { mean: 0.932 }
        },
        highlights: {
          lpips: {
            ordering: 'max',
            frameIndex: 1,
            value: 0.245,
            groundTruthLabel: 'gt:end',
            estimateLabel: 'ORB-SLAM3 Live',
            groundTruthColorJpegBase64: templateCapture.response.colorJpegBase64,
            renderedColorJpegBase64: templateCapture.response.colorJpegBase64
          }
        }
      },
      frames: []
    };
    const imageBenchmarkReportRunB = {
      protocol: 'dreamwalker-localization-image-benchmark/v1',
      type: 'localization-image-benchmark-report',
      createdAt: '2026-04-02T00:11:00.000Z',
      endpoint: sim2realServer.url,
      alignment: 'timestamp',
      groundTruth: {
        fragmentId: 'residency',
        fragmentLabel: 'Residency Run Shelf GT'
      },
      estimate: {
        label: 'ORB-SLAM3 Live Offset',
        sourceType: 'live-stream'
      },
      matching: {
        matchedCount: 2
      },
      metrics: {
        summary: {
          lpips: { mean: 0.5 },
          psnr: { mean: 19.8 },
          ssim: { mean: 0.744 }
        },
        highlights: {
          lpips: {
            ordering: 'max',
            frameIndex: 0,
            value: 0.661,
            groundTruthLabel: 'gt:start',
            estimateLabel: 'ORB-SLAM3 Live Offset',
            groundTruthColorJpegBase64: templateCapture.response.colorJpegBase64,
            renderedColorJpegBase64: templateCapture.response.colorJpegBase64
          }
        }
      },
      frames: []
    };

    await benchmarkCard.locator('#sim2real-image-benchmark-import').setInputFiles({
      name: 'run-a-image-report.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(imageBenchmarkReportRunA), 'utf8')
    });
    await benchmarkCard.locator('#sim2real-image-benchmark-import').setInputFiles({
      name: 'run-b-image-report.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(imageBenchmarkReportRunB), 'utf8')
    });

    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText('2 runs ranked by ATE');
    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText('best ATE ORB-SLAM3 Live / Run A / 0.000 m');
    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText('latest ORB-SLAM3 Live / Run B / ate spread 0.500 m / best LPIPS ORB-SLAM3 Live / Run A / 0.123 / lpips spread 0.377');

    const runEntry = benchmarkCard
      .locator('.state-list .state-card')
      .filter({ hasText: 'ORB-SLAM3 Live / Run A' })
      .first();
    await expect(runEntry).toContainText('live-stream');
    await expect(runEntry).toContainText('2 matched');
    await expect(runEntry).toContainText('ATE 0.000 m');
    await expect(runEntry).toContainText('image lpips 0.123 / psnr 28.50 dB / ssim 0.932');
    await expect(runEntry).toContainText('worst lpips 0.245 / frame 2');
    const compareRowA = benchmarkCard
      .locator('.sim2real-run-compare-row')
      .filter({ hasText: 'ORB-SLAM3 Live / Run A' })
      .first();
    const compareRowB = benchmarkCard
      .locator('.sim2real-run-compare-row')
      .filter({ hasText: 'ORB-SLAM3 Live / Run B' })
      .first();
    await expect(compareRowA).toContainText('#1');
    await expect(compareRowA).toContainText('Baseline');
    await expect(compareRowA).toContainText('Best ATE');
    await expect(compareRowA).toContainText('Best LPIPS');
    await expect(compareRowA).toContainText('0.000 m');
    await expect(compareRowA).toContainText('0.123');
    await expect(compareRowB).toContainText('#2');
    await expect(compareRowB).toContainText('0.500 m');
    await expect(compareRowB).toContainText('0.500');
    await expect(compareRowB).toContainText('+0.500 m');
    await expect(compareRowB).toContainText('+0.377');

    await benchmarkCard.locator('#sim2real-run-compare-baseline').selectOption({
      label: 'ORB-SLAM3 Live / Run B'
    });
    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText(
      'baseline ORB-SLAM3 Live / Run B / delta columns show run minus baseline'
    );
    await expect(compareRowB).toContainText('Baseline');
    await expect(compareRowA).toContainText('-0.500 m');
    await expect(compareRowA).toContainText('-0.377');

    const compareJsonDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Compare JSON' }).click();
    const compareJsonDownload = await compareJsonDownloadPromise;
    const compareJsonDownloadPath = await compareJsonDownload.path();
    const compareJson = JSON.parse(await fs.readFile(compareJsonDownloadPath, 'utf8'));

    expect(compareJson.type).toBe('localization-run-compare-report');
    expect(compareJson.baselineLabel).toBe('ORB-SLAM3 Live / Run B');
    expect(compareJson.baseline.label).toBe('ORB-SLAM3 Live / Run B');
    expect(compareJson.highlights.bestLpips.label).toBe('ORB-SLAM3 Live / Run A');
    expect(compareJson.highlights.bestLpips.worstFrameIndex).toBe(1);
    expect(compareJson.highlights.bestLpips.worstFrameLabel).toBe('gt:end');
    expect(compareJson.rows).toHaveLength(2);
    expect(compareJson.rows[0].deltas.ateMeters).toBe(-0.5);
    expect(compareJson.rows[0].deltas.lpips).toBeCloseTo(-0.377, 6);
    expect(compareJson.rows[0].imageBenchmarkSummary.psnrMean).toBe(28.5);
    expect(compareJson.rows[0].imageBenchmarkSummary.worstLpipsFrameIndex).toBe(1);
    expect(compareJson.rows[0].imageBenchmarkSummary.worstLpipsGroundTruthLabel).toBe('gt:end');

    const compareCsvDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Compare CSV' }).click();
    const compareCsvDownload = await compareCsvDownloadPromise;
    const compareCsvDownloadPath = await compareCsvDownload.path();
    const compareCsv = await fs.readFile(compareCsvDownloadPath, 'utf8');
    const compareCsvLines = compareCsv.trim().split('\n');

    expect(compareCsvLines[0]).toContain(
      'rank,label,sourceType,matchedCount,ateRmseMeters,ateDeltaMeters,yawRmseDegrees,yawDeltaDegrees,psnrMean,ssimMean,lpipsMean,lpipsDelta,worstLpipsValue,worstLpipsFrame'
    );
    expect(compareCsvLines[1]).toContain('1,ORB-SLAM3 Live / Run A,live-stream,2,0,-0.5');
    expect(compareCsvLines[1]).toContain(',28.5,0.932,0.123,-0.377,0.245,2,gt:end,ORB-SLAM3 Live,');
    expect(compareCsvLines[2]).toContain('2,ORB-SLAM3 Live / Run B,live-stream,2,0.5,0');
    expect(compareCsvLines[2]).toContain(',19.8,0.744,0.5,0,0.661,1,gt:start,ORB-SLAM3 Live Offset,');

    const compareMarkdownDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Compare Markdown' }).click();
    const compareMarkdownDownload = await compareMarkdownDownloadPromise;
    const compareMarkdownDownloadPath = await compareMarkdownDownload.path();
    const compareMarkdown = await fs.readFile(compareMarkdownDownloadPath, 'utf8');

    expect(compareMarkdown).toContain('# Localization Run Compare');
    expect(compareMarkdown).toMatch(/- Fragment: (Residency|residency)/);
    expect(compareMarkdown).toContain('- Baseline: ORB-SLAM3 Live / Run B');
    expect(compareMarkdown).toContain('- Best LPIPS: ORB-SLAM3 Live / Run A (0.123)');
    expect(compareMarkdown).toContain('| Rank | Run | Source | Matched | ATE | ΔATE | Yaw | ΔYaw | PSNR | SSIM | LPIPS | ΔLPIPS | Worst LPIPS |');
    expect(compareMarkdown).toContain('| 1 | ORB-SLAM3 Live / Run A / best-ate / best-yaw / best-lpips | live-stream | 2 | 0.000 m | -0.500 m | 0.00 deg | 0.00 deg | 28.50 dB | 0.932 | 0.123 | -0.377 | 0.245 @ frame 2 |');
    expect(compareMarkdown).toContain('| 2 | ORB-SLAM3 Live / Run B / baseline / latest / active | live-stream | 2 | 0.500 m | 0.000 m | 0.00 deg | 0.00 deg | 19.80 dB | 0.744 | 0.500 | 0.000 | 0.661 @ frame 1 |');

    const reviewBundleDownloadPromise = page.waitForEvent('download');
    await benchmarkCard.getByRole('button', { name: 'Download Review Bundle' }).click();
    const reviewBundleDownload = await reviewBundleDownloadPromise;
    const reviewBundleDownloadPath = await reviewBundleDownload.path();
    const reviewBundle = JSON.parse(await fs.readFile(reviewBundleDownloadPath, 'utf8'));

    expect(reviewBundle.type).toBe('localization-review-bundle');
    expect(reviewBundle.selection.baselineLabel).toBe('ORB-SLAM3 Live / Run B');
    expect(reviewBundle.compareReport.type).toBe('localization-run-compare-report');
    expect(reviewBundle.artifacts.compareCsv).toContain('rank,label,sourceType,matchedCount');
    expect(reviewBundle.artifacts.compareMarkdown).toContain('# Localization Run Compare');
    expect(reviewBundle.artifacts.worstLpipsPreviews).toHaveLength(2);
    expect(reviewBundle.artifacts.worstLpipsPreviews[0].runLabel).toBe('ORB-SLAM3 Live / Run A');
    expect(reviewBundle.artifacts.worstLpipsPreviews[0].frameIndex).toBe(1);
    expect(reviewBundle.artifacts.worstLpipsPreviews[0].groundTruthColorJpegBase64).toBe(
      templateCapture.response.colorJpegBase64
    );
    expect(reviewBundle.artifacts.worstLpipsPreviews[0].renderedColorJpegBase64).toBe(
      templateCapture.response.colorJpegBase64
    );
    expect(reviewBundle.runs).toHaveLength(2);
    expect(reviewBundle.runs[0].snapshot.type).toBe('localization-run-snapshot');
    expect(reviewBundle.runs[0].snapshot.imageBenchmark.summary.lpipsMean).toBe(0.123);
    expect(reviewBundle.runs[0].reviewArtifacts.worstLpipsPreview.value).toBe(0.245);
    expect(reviewBundle.linkedCaptures).toHaveLength(1);
    expect(reviewBundle.linkedCaptures[0].sourceId).toBe('current-capture');
    expect(reviewBundle.linkedCaptures[0].bundle.type).toBe('route-capture-bundle');
    expect(reviewBundle.linkedCaptures[0].captureCount).toBe(2);

    await benchmarkCard.getByRole('button', { name: 'Clear Run Shelf' }).click();
    await page.getByRole('button', { name: 'Clear Capture Shelf' }).click();
    await expect(benchmarkCard).toContainText('No saved runs');

    await benchmarkCard.locator('#sim2real-review-bundle-import').setInputFiles(
      reviewBundleDownloadPath
    );
    await expect(benchmarkCard).toContainText('2 saved runs');
    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText(
      'baseline ORB-SLAM3 Live / Run B / delta columns show run minus baseline'
    );
    await expect(
      benchmarkCard
        .locator('.sim2real-run-compare-row')
        .filter({ hasText: 'ORB-SLAM3 Live / Run B' })
        .first()
    ).toContainText('Baseline');
    await expect(
      page
        .locator('.state-card')
        .filter({ hasText: 'Current Capture / 2 frames' })
        .getByRole('button', { name: 'Download Bundle JSON' })
    ).toBeVisible();

    const importedRunEntry = benchmarkCard
      .locator('.state-list .state-card')
      .filter({ hasText: 'ORB-SLAM3 Live / Run A' })
      .first();

    await importedRunEntry.getByRole('button', { name: 'Preview Worst LPIPS', exact: true }).click();
    await expect(benchmarkCard.locator('.sim2real-run-image-preview-card')).toContainText('Worst LPIPS Preview');
    await expect(benchmarkCard.locator('.sim2real-run-image-preview-card')).toContainText('LPIPS 0.245 / frame 2');
    await expect(page.getByAltText('Localization Ground Truth Preview')).toBeVisible();
    await expect(page.getByAltText('Localization Rendered Preview')).toBeVisible();

    await benchmarkCard.getByRole('button', { name: 'Clear Live Estimate' }).click();
    await expect(benchmarkCard).toContainText('Idle');

    await importedRunEntry.getByRole('button', { name: 'Load Run', exact: true }).click();
    await expect(benchmarkCard).toContainText('estimate ORB-SLAM3 Live / 2 poses / source imported');
    await expect(benchmarkCard).toContainText('2 matched poses');
    await expect(benchmarkCard).toContainText('0.000 m');
    await expect(
      benchmarkCard
        .locator('.sim2real-run-compare-row')
        .filter({ hasText: 'ORB-SLAM3 Live / Run A' })
        .first()
    ).toContainText('Active');

    const runDownloadPromise = page.waitForEvent('download');
    await importedRunEntry.getByRole('button', { name: 'Download Run JSON', exact: true }).click();
    const runDownload = await runDownloadPromise;
    const runDownloadPath = await runDownload.path();
    const runSnapshot = JSON.parse(await fs.readFile(runDownloadPath, 'utf8'));

    expect(runSnapshot.type).toBe('localization-run-snapshot');
    expect(runSnapshot.summary.matchedCount).toBe(2);
    expect(runSnapshot.estimate.sourceType).toBe('live-stream');
    expect(runSnapshot.groundTruth.bundle.type).toBe('route-capture-bundle');
    expect(runSnapshot.report.metrics.ateRmseMeters).toBe(0);
    expect(runSnapshot.imageBenchmark.summary.lpipsMean).toBe(0.123);
  } finally {
    await liveLocalizationServer.close();
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel runs websocket localization image benchmarks for saved runs', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();
  const liveLocalizationServer = await createLiveLocalizationMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const templateBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const templateCapture = templateBundle.captures[0];
    const startPosition = templateCapture.pose.position;
    const endPosition = [startPosition[0] + 1.5, startPosition[1], startPosition[2]];
    const yawDegrees = templateCapture.pose.yawDegrees;
    const yawRadians = (yawDegrees * Math.PI) / 180;
    const groundTruthBundle = {
      ...templateBundle,
      capturedAt: '2026-04-02T00:00:00.000Z',
      fragmentLabel: 'Residency Direct Image GT',
      route: [
        {
          index: 0,
          position: [...startPosition],
          yawDegrees,
          relativeTimeSeconds: 0
        },
        {
          index: 1,
          position: [...endPosition],
          yawDegrees,
          relativeTimeSeconds: 1
        }
      ],
      captures: [
        {
          ...templateCapture,
          index: 0,
          label: 'gt:start',
          capturedAt: '2026-04-02T00:00:00.000Z',
          relativeTimeSeconds: 0,
          pose: {
            position: [...startPosition],
            yawDegrees
          }
        },
        {
          ...templateCapture,
          index: 1,
          label: 'gt:end',
          capturedAt: '2026-04-02T00:00:01.000Z',
          relativeTimeSeconds: 1,
          pose: {
            position: [...endPosition],
            yawDegrees
          }
        }
      ]
    };

    await sim2realPanel.locator('#sim2real-capture-import').setInputFiles({
      name: 'direct-image-ground-truth.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(groundTruthBundle), 'utf8')
    });

    await benchmarkCard.locator('#sim2real-live-estimate-url').fill(liveLocalizationServer.url);
    await benchmarkCard.getByRole('button', { name: 'Connect Live Monitor', exact: true }).click();
    await expect(benchmarkCard).toContainText('live monitor Connected');

    await liveLocalizationServer.send({
      type: 'reset',
      label: 'ORB-SLAM3 Direct Image'
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Direct Image',
      pose: {
        position: [...startPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 0
      }
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Direct Image',
      pose: {
        position: [...endPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 1
      }
    });

    await expect(benchmarkCard).toContainText('estimate ORB-SLAM3 Direct Image / 2 poses / source auto-live');
    await expect(benchmarkCard).toContainText('2 matched poses');

    await benchmarkCard
      .locator('#sim2real-localization-run-shelf-label')
      .fill('ORB-SLAM3 Direct Image / Run A');
    await benchmarkCard.getByRole('button', { name: 'Save Benchmark Run' }).click();

    const runEntry = benchmarkCard
      .locator('.state-list .state-card')
      .filter({ hasText: 'ORB-SLAM3 Direct Image / Run A' })
      .first();

    await runEntry.getByRole('button', { name: 'Run Image Benchmark', exact: true }).click();
    await waitForCondition(
      () =>
        sim2realServer.messages.some((message) => message.type === 'localization-image-benchmark'),
      'sim2real localization image benchmark request'
    );

    const benchmarkRequest = sim2realServer.messages.find(
      (message) => message.type === 'localization-image-benchmark'
    );
    expect(benchmarkRequest.estimate.label).toBe('ORB-SLAM3 Direct Image');
    expect(benchmarkRequest.groundTruthBundle.type).toBe('route-capture-bundle');
    expect(benchmarkRequest.groundTruthBundle.captures).toHaveLength(2);
    expect(benchmarkRequest.metrics).toEqual(['psnr', 'ssim', 'lpips']);

    await expect(runEntry).toContainText('image lpips 0.123 / psnr 28.50 dB / ssim 0.932');
    await expect(runEntry).toContainText('worst lpips 0.245 / frame 2');
    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText('best LPIPS ORB-SLAM3 Direct Image / Run A / 0.123');
    await expect(benchmarkCard.locator('.sim2real-run-image-preview-card')).toContainText('Worst LPIPS Preview');
    await expect(page.getByAltText('Localization Ground Truth Preview')).toBeVisible();
    await expect(page.getByAltText('Localization Rendered Preview')).toBeVisible();

    const runDownloadPromise = page.waitForEvent('download');
    await runEntry.getByRole('button', { name: 'Download Run JSON', exact: true }).click();
    const runDownload = await runDownloadPromise;
    const runDownloadPath = await runDownload.path();
    const runSnapshot = JSON.parse(await fs.readFile(runDownloadPath, 'utf8'));

    expect(runSnapshot.imageBenchmark.summary.lpipsMean).toBe(0.123);
    expect(runSnapshot.imageBenchmark.endpoint).toBe(sim2realServer.url);
  } finally {
    await liveLocalizationServer.close();
    await sim2realServer.close();
  }
});

test('Robot mode sim2real panel batches missing websocket localization image benchmarks', async ({ page }) => {
  const sim2realServer = await createSim2RealMockServer();
  const liveLocalizationServer = await createLiveLocalizationMockServer();

  try {
    await page.goto(`/?sim2real=1&sim2realUrl=${encodeURIComponent(sim2realServer.url)}`);

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const sim2realPanel = rightPanel.locator('.sim2real-panel');
    const benchmarkCard = sim2realPanel.locator('.sim2real-benchmark-card');

    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(sim2realPanel).toContainText('Connected');

    await sim2realPanel.getByRole('button', { name: 'Capture Route Bundle' }).click();
    await expect(sim2realPanel.locator('.sim2real-capture-card')).toContainText('1 frames');

    const captureDownloadPromise = page.waitForEvent('download');
    await sim2realPanel.getByRole('button', { name: 'Download Capture JSON' }).click();
    const captureDownload = await captureDownloadPromise;
    const captureDownloadPath = await captureDownload.path();
    const templateBundle = JSON.parse(await fs.readFile(captureDownloadPath, 'utf8'));
    const templateCapture = templateBundle.captures[0];
    const startPosition = templateCapture.pose.position;
    const endPosition = [startPosition[0] + 1.5, startPosition[1], startPosition[2]];
    const yawDegrees = templateCapture.pose.yawDegrees;
    const yawRadians = (yawDegrees * Math.PI) / 180;
    const groundTruthBundle = {
      ...templateBundle,
      capturedAt: '2026-04-02T00:00:00.000Z',
      fragmentLabel: 'Residency Batch Image GT',
      route: [
        {
          index: 0,
          position: [...startPosition],
          yawDegrees,
          relativeTimeSeconds: 0
        },
        {
          index: 1,
          position: [...endPosition],
          yawDegrees,
          relativeTimeSeconds: 1
        }
      ],
      captures: [
        {
          ...templateCapture,
          index: 0,
          label: 'gt:start',
          capturedAt: '2026-04-02T00:00:00.000Z',
          relativeTimeSeconds: 0,
          pose: {
            position: [...startPosition],
            yawDegrees
          }
        },
        {
          ...templateCapture,
          index: 1,
          label: 'gt:end',
          capturedAt: '2026-04-02T00:00:01.000Z',
          relativeTimeSeconds: 1,
          pose: {
            position: [...endPosition],
            yawDegrees
          }
        }
      ]
    };

    await sim2realPanel.locator('#sim2real-capture-import').setInputFiles({
      name: 'batch-image-ground-truth.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(groundTruthBundle), 'utf8')
    });

    await benchmarkCard.locator('#sim2real-live-estimate-url').fill(liveLocalizationServer.url);
    await benchmarkCard.getByRole('button', { name: 'Connect Live Monitor', exact: true }).click();
    await expect(benchmarkCard).toContainText('live monitor Connected');

    await liveLocalizationServer.send({
      type: 'reset',
      label: 'ORB-SLAM3 Batch A'
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Batch A',
      pose: {
        position: [...startPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 0
      }
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Batch A',
      pose: {
        position: [...endPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 1
      }
    });

    await benchmarkCard.locator('#sim2real-localization-run-shelf-label').fill('ORB-SLAM3 Batch / Run A');
    await benchmarkCard.getByRole('button', { name: 'Save Benchmark Run' }).click();
    await expect(benchmarkCard).toContainText('1 saved runs');

    const shiftedStartPosition = [startPosition[0] + 0.5, startPosition[1], startPosition[2]];
    const shiftedEndPosition = [endPosition[0] + 0.5, endPosition[1], endPosition[2]];

    await liveLocalizationServer.send({
      type: 'reset',
      label: 'ORB-SLAM3 Batch Offset'
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Batch Offset',
      pose: {
        position: [...shiftedStartPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 0
      }
    });
    await liveLocalizationServer.send({
      type: 'pose-estimate',
      label: 'ORB-SLAM3 Batch Offset',
      pose: {
        position: [...shiftedEndPosition],
        orientation: [0, Math.sin(yawRadians / 2), 0, Math.cos(yawRadians / 2)],
        timestampSeconds: 1
      }
    });

    await benchmarkCard.locator('#sim2real-localization-run-shelf-label').fill('ORB-SLAM3 Batch / Run B');
    await benchmarkCard.getByRole('button', { name: 'Save Benchmark Run' }).click();
    await expect(benchmarkCard).toContainText('2 saved runs');
    await expect(benchmarkCard).toContainText('image metrics 0 / missing 2');

    await benchmarkCard.getByRole('button', { name: 'Run Missing Image Benchmarks' }).click();
    await expect(benchmarkCard).toContainText('image benchmark batch missing-only');

    await waitForCondition(
      () =>
        sim2realServer.messages.filter((message) => message.type === 'localization-image-benchmark').length === 2,
      'sim2real localization image benchmark batch requests'
    );

    const benchmarkRequests = sim2realServer.messages.filter(
      (message) => message.type === 'localization-image-benchmark'
    );
    expect(benchmarkRequests).toHaveLength(2);
    expect(
      benchmarkRequests.map((message) => message.estimate.label).sort()
    ).toEqual(['ORB-SLAM3 Batch A', 'ORB-SLAM3 Batch Offset']);

    const runEntryA = benchmarkCard
      .locator('.state-list .state-card')
      .filter({ hasText: 'ORB-SLAM3 Batch / Run A' })
      .first();
    const runEntryB = benchmarkCard
      .locator('.state-list .state-card')
      .filter({ hasText: 'ORB-SLAM3 Batch / Run B' })
      .first();

    await expect(runEntryA).toContainText('image lpips 0.123 / psnr 28.50 dB / ssim 0.932');
    await expect(runEntryB).toContainText('image lpips 0.500 / psnr 19.80 dB / ssim 0.744');
    await expect(benchmarkCard).toContainText('image metrics 2 / missing 0');
    await expect(benchmarkCard.locator('.sim2real-run-compare-card')).toContainText(
      'best LPIPS ORB-SLAM3 Batch / Run A / 0.123 / lpips spread 0.377'
    );
    await expect(benchmarkCard.locator('.sim2real-run-image-preview-card')).toContainText('Worst LPIPS Preview');
  } finally {
    await liveLocalizationServer.close();
    await sim2realServer.close();
  }
});

test('Robot bridge CLI can request v1 state snapshots', async ({ page }) => {
  await page.goto('/?robotBridge=1');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const bridgeCard = rightPanel.locator('.state-grid .state-card').nth(3);

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(bridgeCard).toContainText('Connected');

  const { stdout } = await execFileAsync(
    'node',
    ['./tools/robotics-bridge-client.mjs', 'request-state'],
    {
      cwd: dreamwalkerWebRoot
    }
  );
  const snapshot = JSON.parse(stdout);

  expect(snapshot.type).toBe('robot-state');
  expect(snapshot.protocol).toBe(robotBridgeProtocolId);
  expect(snapshot.source).toBe('dreamwalker-live');
  expect(snapshot.fragmentId).toBe('residency');
});

test('ROSBridge relay publishes robot state and accepts ROS-side commands', async ({ page }) => {
  await page.goto('/?robotBridge=1');

  const topbar = page.locator('.topbar');
  const rightPanel = page.locator('.right-panel');
  const bridgeCard = rightPanel.locator('.state-grid .state-card').nth(3);
  const poseCard = rightPanel.locator('.state-grid .state-card').first();

  await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
  await expect(bridgeCard).toContainText('Connected');

  const rosbridgeServer = await createRosbridgeMockServer();
  const relayProcess = spawnRosbridgeRelay(['--rosbridge', rosbridgeServer.url]);

  try {
    const relaySocket = await waitForCondition(
      () => rosbridgeServer.getSocket(),
      'rosbridge relay connection'
    );

    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'subscribe' && frame.topic === '/dreamwalker/cmd_pose2d'
        ),
      'rosbridge relay subscribe'
    );

    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_state_json'
        ),
      'rosbridge robot-state publish'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_route_path'
        ),
      'rosbridge robot-route-path publish'
    );

    const stateFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_state_json'
    );
    const poseFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_pose2d'
    );
    const poseStampedFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_pose_stamped'
    );
    const routePathFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_route_path'
    );

    expect(stateFrame).not.toBeNull();
    expect(poseFrame).not.toBeNull();
    expect(poseStampedFrame).not.toBeNull();
    expect(routePathFrame).not.toBeNull();

    const stateJson = JSON.parse(stateFrame.msg.data);
    expect(stateJson.protocol).toBe(robotBridgeProtocolId);
    expect(stateJson.fragmentId).toBe('residency');
    expect(poseFrame.msg.x).toBeCloseTo(0, 2);
    expect(poseFrame.msg.y).toBeCloseTo(5.8, 2);
    expect(poseStampedFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(poseStampedFrame.msg.pose.position.x).toBeCloseTo(0, 2);
    expect(poseStampedFrame.msg.pose.position.y).toBeCloseTo(5.8, 2);
    expect(routePathFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(routePathFrame.msg.poses).toHaveLength(1);

    relaySocket.send(
      JSON.stringify({
        op: 'publish',
        topic: '/dreamwalker/cmd_pose2d',
        msg: {
          x: 2.5,
          y: 7.1,
          theta: Math.PI / 2
        }
      })
    );
    await expect(poseCard).toContainText('x 2.50 / z 7.10');

    relaySocket.send(
      JSON.stringify({
        op: 'publish',
        topic: '/dreamwalker/cmd_waypoint',
        msg: {
          x: 2.5,
          y: 0,
          z: 9.3
        }
      })
    );
    await expect(rightPanel.locator('.robot-camera-panel')).toContainText('m ahead');
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_goal_pose_stamped'
        ),
      'rosbridge goal pose publish'
    );
    const goalPoseFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/robot_goal_pose_stamped'
    );
    expect(goalPoseFrame).not.toBeNull();
    expect(goalPoseFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(goalPoseFrame.msg.pose.position.x).toBeCloseTo(2.5, 2);
    expect(goalPoseFrame.msg.pose.position.y).toBeCloseTo(9.3, 2);

    relaySocket.send(
      JSON.stringify({
        op: 'publish',
        topic: '/dreamwalker/cmd_json',
        msg: {
          data: JSON.stringify({
            type: 'set-camera',
            cameraId: 'top'
          })
        }
      })
    );
    await expect(rightPanel.locator('.robot-camera-panel')).toContainText('Top View');
  } finally {
    await stopChildProcess(relayProcess.child);
    await rosbridgeServer.close();
  }
});

test('Robot frame streaming from browser reaches ROS image topics through relay', async ({ page }) => {
  const rosbridgeServer = await createRosbridgeMockServer();
  const relayProcess = spawnRosbridgeRelay(['--rosbridge', rosbridgeServer.url]);

  try {
    await page.goto('/?robotBridge=1&robotFrameStream=1&robotFrameFps=5');

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const bridgeCard = rightPanel.locator('.state-grid .state-card').nth(3);

    await expect(page.getByRole('heading', { name: 'Dynamic Map Viewer' })).toBeVisible();
    await expect(page.locator('.dreamwalker-stage canvas')).toBeVisible({ timeout: 20_000 });
    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(bridgeCard).toContainText('Connected');

    await waitForCondition(
      () => rosbridgeServer.getSocket(),
      'rosbridge relay connection'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'advertise' && frame.topic === '/dreamwalker/camera/compressed'
        ),
      'camera compressed advertise'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'advertise' && frame.topic === '/dreamwalker/camera/camera_info'
        ),
      'camera info advertise'
    );

    await waitForCondition(
      () =>
        rosbridgeServer.frames.filter(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/compressed'
        ).length >= 2,
      'streamed camera compressed publishes',
      15_000
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.filter(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/camera_info'
        ).length >= 2,
      'streamed camera info publishes',
      15_000
    );

    const compressedFrames = rosbridgeServer.frames.filter(
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/compressed'
    );
    const cameraInfoFrames = rosbridgeServer.frames.filter(
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/camera_info'
    );
    const compressedFrame = compressedFrames.at(-1);
    const previousCompressedFrame = compressedFrames.at(-2);
    const cameraInfoFrame = cameraInfoFrames.at(-1);

    expect(previousCompressedFrame).not.toBeNull();
    expect(compressedFrame).not.toBeNull();
    expect(cameraInfoFrame).not.toBeNull();
    expect(compressedFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(compressedFrame.msg.format).toBe('jpeg');

    const jpegBytes = Buffer.from(compressedFrame.msg.data, 'base64');
    expect(jpegBytes.length).toBeGreaterThan(100);
    expect(jpegBytes[0]).toBe(0xff);
    expect(jpegBytes[1]).toBe(0xd8);
    expect(
      compressedFrame.msg.header.stamp.sec !== previousCompressedFrame.msg.header.stamp.sec ||
        compressedFrame.msg.header.stamp.nanosec !== previousCompressedFrame.msg.header.stamp.nanosec
    ).toBe(true);

    expect(cameraInfoFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(cameraInfoFrame.msg.width).toBeGreaterThan(0);
    expect(cameraInfoFrame.msg.height).toBeGreaterThan(0);
    expect(cameraInfoFrame.msg.width).toBeGreaterThanOrEqual(cameraInfoFrame.msg.height);
    expect(cameraInfoFrame.msg.distortion_model).toBe('plumb_bob');
    expect(cameraInfoFrame.msg.k).toHaveLength(9);
    expect(cameraInfoFrame.msg.p).toHaveLength(12);
    expect(cameraInfoFrame.msg.header.stamp.sec).toBe(compressedFrame.msg.header.stamp.sec);
    expect(cameraInfoFrame.msg.header.stamp.nanosec).toBe(compressedFrame.msg.header.stamp.nanosec);
  } finally {
    await stopChildProcess(relayProcess.child);
    await rosbridgeServer.close();
  }
});

test('Robot depth streaming from browser reaches ROS depth topics through relay', async ({
  page
}) => {
  const rosbridgeServer = await createRosbridgeMockServer();
  const relayProcess = spawnRosbridgeRelay(['--rosbridge', rosbridgeServer.url]);

  try {
    await page.goto('/?robotBridge=1&robotFrameStream=1&robotDepthStream=1&robotFrameFps=5');

    const topbar = page.locator('.topbar');
    const rightPanel = page.locator('.right-panel');
    const bridgeCard = rightPanel.locator('.state-grid .state-card').nth(3);

    await expect(page.getByRole('heading', { name: 'Dynamic Map Viewer' })).toBeVisible();
    await expect(page.locator('.dreamwalker-stage canvas')).toBeVisible({ timeout: 20_000 });
    await topbar.getByRole('button', { name: 'Robot', exact: true }).click();
    await expect(bridgeCard).toContainText('Connected');

    await waitForCondition(
      () => rosbridgeServer.getSocket(),
      'rosbridge relay connection'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'advertise' && frame.topic === '/dreamwalker/depth/image'
        ),
      'depth image advertise'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.filter(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/depth/image'
        ).length >= 2,
      'streamed depth image publishes',
      20_000
    );

    const depthFrames = rosbridgeServer.frames.filter(
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/depth/image'
    );
    const depthFrame = depthFrames.at(-1);
    const previousDepthFrame = depthFrames.at(-2);

    expect(previousDepthFrame).not.toBeNull();
    expect(depthFrame).not.toBeNull();
    expect(depthFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(depthFrame.msg.encoding).toBe('32FC1');
    expect(depthFrame.msg.is_bigendian).toBe(0);
    expect(depthFrame.msg.width).toBeGreaterThan(0);
    expect(depthFrame.msg.height).toBeGreaterThan(0);
    expect(depthFrame.msg.step).toBe(depthFrame.msg.width * 4);

    const depthBytes = Buffer.from(depthFrame.msg.data, 'base64');
    expect(depthBytes.length).toBe(depthFrame.msg.width * depthFrame.msg.height * 4);

    const depthSamples = decodeFloat32LeSamples(depthBytes);
    expect(depthSamples.length).toBeGreaterThan(0);
    expect(depthSamples.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    expect(
      depthFrame.msg.header.stamp.sec !== previousDepthFrame.msg.header.stamp.sec ||
        depthFrame.msg.header.stamp.nanosec !== previousDepthFrame.msg.header.stamp.nanosec
    ).toBe(true);
  } finally {
    await stopChildProcess(relayProcess.child);
    await rosbridgeServer.close();
  }
});

test('ROSBridge relay publishes camera binary frames as ROS image topics', async () => {
  const roboticsServer = await createRoboticsMockServer();
  const rosbridgeServer = await createRosbridgeMockServer();
  const relayProcess = spawnRosbridgeRelay([
    '--robotics',
    roboticsServer.url,
    '--rosbridge',
    rosbridgeServer.url
  ]);

  try {
    const roboticsSocket = await waitForCondition(
      () => roboticsServer.getSocket(),
      'robotics relay connection'
    );
    await waitForCondition(
      () => rosbridgeServer.getSocket(),
      'rosbridge relay connection'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'advertise' && frame.topic === '/dreamwalker/camera/compressed'
        ),
      'camera compressed advertise'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'advertise' && frame.topic === '/dreamwalker/camera/camera_info'
        ),
      'camera info advertise'
    );

    const timestamp = '2026-04-01T12:34:56.789Z';
    const timestampMs = Date.parse(timestamp);
    const jpegBytes = Buffer.from([0xff, 0xd8, 0xff, 0xdb, 0x00, 0x43, 0x00, 0xff, 0xd9]);
    const cameraFrame = await buildCameraFrameMessage(new Blob([jpegBytes]), {
      timestamp,
      cameraId: 'front',
      width: 640,
      height: 480,
      fov: 60,
      pose: {
        position: [1, 2, 3],
        orientation: [0, 0, 0, 1]
      }
    });

    roboticsSocket.send(cameraFrame);

    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/compressed'
        ),
      'camera compressed publish'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/camera_info'
        ),
      'camera info publish'
    );

    const compressedFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/compressed'
    );
    const cameraInfoFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/camera/camera_info'
    );

    expect(compressedFrame).not.toBeNull();
    expect(cameraInfoFrame).not.toBeNull();
    expect(compressedFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(compressedFrame.msg.header.stamp.sec).toBe(Math.floor(timestampMs / 1_000));
    expect(compressedFrame.msg.header.stamp.nanosec).toBe((timestampMs % 1_000) * 1_000_000);
    expect(compressedFrame.msg.format).toBe('jpeg');
    expect(compressedFrame.msg.data).toBe(jpegBytes.toString('base64'));

    const fy = 480 / (2 * Math.tan((60 * Math.PI) / 360));
    const fx = fy * (640 / 480);
    const cx = 640 / 2;
    const cy = 480 / 2;

    expect(cameraInfoFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(cameraInfoFrame.msg.header.stamp.sec).toBe(Math.floor(timestampMs / 1_000));
    expect(cameraInfoFrame.msg.header.stamp.nanosec).toBe((timestampMs % 1_000) * 1_000_000);
    expect(cameraInfoFrame.msg.width).toBe(640);
    expect(cameraInfoFrame.msg.height).toBe(480);
    expect(cameraInfoFrame.msg.distortion_model).toBe('plumb_bob');
    expect(cameraInfoFrame.msg.d).toEqual([0, 0, 0, 0, 0]);
    expect(cameraInfoFrame.msg.k[0]).toBeCloseTo(fx, 6);
    expect(cameraInfoFrame.msg.k[2]).toBeCloseTo(cx, 6);
    expect(cameraInfoFrame.msg.k[4]).toBeCloseTo(fy, 6);
    expect(cameraInfoFrame.msg.k[5]).toBeCloseTo(cy, 6);
    expect(cameraInfoFrame.msg.r).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(cameraInfoFrame.msg.p[0]).toBeCloseTo(fx, 6);
    expect(cameraInfoFrame.msg.p[2]).toBeCloseTo(cx, 6);
    expect(cameraInfoFrame.msg.p[5]).toBeCloseTo(fy, 6);
    expect(cameraInfoFrame.msg.p[6]).toBeCloseTo(cy, 6);
    expect(cameraInfoFrame.msg.p[10]).toBe(1);
  } finally {
    await stopChildProcess(relayProcess.child);
    await roboticsServer.close();
    await rosbridgeServer.close();
  }
});

test('ROSBridge relay publishes depth binary frames as ROS image topics', async () => {
  const roboticsServer = await createRoboticsMockServer();
  const rosbridgeServer = await createRosbridgeMockServer();
  const relayProcess = spawnRosbridgeRelay([
    '--robotics',
    roboticsServer.url,
    '--rosbridge',
    rosbridgeServer.url
  ]);

  try {
    const roboticsSocket = await waitForCondition(
      () => roboticsServer.getSocket(),
      'robotics relay connection'
    );
    await waitForCondition(
      () => rosbridgeServer.getSocket(),
      'rosbridge relay connection'
    );
    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'advertise' && frame.topic === '/dreamwalker/depth/image'
        ),
      'depth image advertise'
    );

    const timestamp = '2026-04-01T12:34:56.789Z';
    const timestampMs = Date.parse(timestamp);
    const depthValues = new Float32Array([1.5, 2.25, 3.75, 5.5]);
    const depthFrame = await buildDepthFrameMessage(depthValues, {
      timestamp,
      cameraId: 'front',
      width: 2,
      height: 2,
      fov: 60,
      nearClip: 0.1,
      farClip: 50,
      pose: {
        position: [1, 2, 3],
        orientation: [0, 0, 0, 1]
      }
    });

    roboticsSocket.send(depthFrame);

    await waitForCondition(
      () =>
        rosbridgeServer.frames.some(
          (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/depth/image'
        ),
      'depth image publish'
    );

    const publishedDepthFrame = findLastMessage(
      rosbridgeServer.frames,
      (frame) => frame.op === 'publish' && frame.topic === '/dreamwalker/depth/image'
    );

    expect(publishedDepthFrame).not.toBeNull();
    expect(publishedDepthFrame.msg.header.frame_id).toBe('dreamwalker_map');
    expect(publishedDepthFrame.msg.header.stamp.sec).toBe(Math.floor(timestampMs / 1_000));
    expect(publishedDepthFrame.msg.header.stamp.nanosec).toBe(
      (timestampMs % 1_000) * 1_000_000
    );
    expect(publishedDepthFrame.msg.width).toBe(2);
    expect(publishedDepthFrame.msg.height).toBe(2);
    expect(publishedDepthFrame.msg.encoding).toBe('32FC1');
    expect(publishedDepthFrame.msg.is_bigendian).toBe(0);
    expect(publishedDepthFrame.msg.step).toBe(8);
    expect(publishedDepthFrame.msg.data).toBe(
      Buffer.from(depthValues.buffer, depthValues.byteOffset, depthValues.byteLength).toString(
        'base64'
      )
    );
  } finally {
    await stopChildProcess(relayProcess.child);
    await roboticsServer.close();
    await rosbridgeServer.close();
  }
});
