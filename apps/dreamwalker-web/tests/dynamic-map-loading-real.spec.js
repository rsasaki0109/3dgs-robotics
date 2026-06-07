import { expect, test } from '@playwright/test';
import { existsSync, statSync } from 'node:fs';
import WebSocket from 'ws';
import {
  robotBridgeBrowserSource,
  robotBridgeProtocolId
} from '../src/robotics-bridge.js';

const catalogUrl = '/manifests/dev-large-scale-3dgs-tile-catalog.json';
const westSplatUrl = '/splats/dev-large-scale/tile_west.splat';
const eastSplatUrl = '/splats/dev-large-scale/tile_east.splat';
const catalogPath = new URL('../public/manifests/dev-large-scale-3dgs-tile-catalog.json', import.meta.url);
const westSplatPath = new URL('../public/splats/dev-large-scale/tile_west.splat', import.meta.url);
const eastSplatPath = new URL('../public/splats/dev-large-scale/tile_east.splat', import.meta.url);
const smokeCatalogUrl = '/manifests/smoke-real-train-tile-catalog.json';
const smokeSplatUrl = '/splats/smoke-real-train/tile_x000_y000.splat';
const smokeCatalogPath = new URL('../public/manifests/smoke-real-train-tile-catalog.json', import.meta.url);
const smokeSplatPath = new URL('../public/splats/smoke-real-train/tile_x000_y000.splat', import.meta.url);
const smokeTwoTileCatalogUrl = '/manifests/smoke-gsplat-2tile-browser-catalog.json';
const smokeTwoTileWestSplatUrl = '/splats/smoke-gsplat-2tile/tile_x000_y000.splat';
const smokeTwoTileEastSplatUrl = '/splats/smoke-gsplat-2tile/tile_x001_y000.splat';
const smokeTwoTileCatalogPath = new URL(
  '../public/manifests/smoke-gsplat-2tile-browser-catalog.json',
  import.meta.url
);
const smokeTwoTileWestSplatPath = new URL(
  '../public/splats/smoke-gsplat-2tile/tile_x000_y000.splat',
  import.meta.url
);
const smokeTwoTileEastSplatPath = new URL(
  '../public/splats/smoke-gsplat-2tile/tile_x001_y000.splat',
  import.meta.url
);
const smokeGridCatalogUrl = '/manifests/smoke-gsplat-grid6-catalog.json';
const smokeGridFarTileId = 'tile_x002_z001';
const smokeGridFarSplatUrl = `/splats/smoke-gsplat-grid6/${smokeGridFarTileId}.splat`;
const smokeGridRouteUrl = '/robot-routes/smoke-gsplat-grid6-route.json';
const smokeGridCatalogPath = new URL(
  '../public/manifests/smoke-gsplat-grid6-catalog.json',
  import.meta.url
);
const smokeGridRoutePath = new URL(
  '../public/robot-routes/smoke-gsplat-grid6-route.json',
  import.meta.url
);
const smokeGridSplatPaths = [
  'tile_x000_z000',
  'tile_x000_z001',
  'tile_x001_z000',
  'tile_x001_z001',
  'tile_x002_z000',
  'tile_x002_z001'
].map(
  (tileId) => new URL(`../public/splats/smoke-gsplat-grid6/${tileId}.splat`, import.meta.url)
);
const smokeGridPositionSequence = [
  { tileId: 'tile_x000_z000', position: { x: 7.75, y: 0, z: 7.75 } },
  { tileId: 'tile_x000_z001', position: { x: 7.75, y: 0, z: 15.75 } },
  { tileId: 'tile_x001_z001', position: { x: 15.75, y: 0, z: 15.75 } },
  { tileId: 'tile_x002_z001', position: { x: 23.75, y: 0, z: 15.75 } },
  { tileId: 'tile_x002_z000', position: { x: 23.75, y: 0, z: 7.75 } },
  { tileId: 'tile_x001_z000', position: { x: 15.75, y: 0, z: 7.75 } }
];
const smokeGridInitialRoutePreloadTileIds = [
  'tile_x000_z001',
  'tile_x001_z001',
  'tile_x002_z001',
  'tile_x002_z000'
];
const smokeGridRoutePlaybackMs = 2000;

function hasRealDynamicMapFixture() {
  return [catalogPath, westSplatPath, eastSplatPath].every((url) => existsSync(url));
}

function hasSmokeTrainFixture() {
  return [smokeCatalogPath, smokeSplatPath].every((url) => existsSync(url));
}

function hasSmokeTwoTileFixture() {
  return [smokeTwoTileCatalogPath, smokeTwoTileWestSplatPath, smokeTwoTileEastSplatPath].every(
    (url) => existsSync(url)
  );
}

function hasSmokeGridFixture() {
  return [smokeGridCatalogPath, smokeGridRoutePath, ...smokeGridSplatPaths].every((url) =>
    existsSync(url)
  );
}

function contentLength(response) {
  const raw = response.headers()['content-length'];
  const value = Number(raw);
  return Number.isFinite(value) ? value : 0;
}

function waitForBridgeOpen(socket) {
  return new Promise((resolve, reject) => {
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
    const messages = [];

    const timeoutId = setInterval(() => {
      const matchIndex = messages.findIndex((message) => predicate(message));
      if (matchIndex >= 0) {
        clearInterval(timeoutId);
        cleanup();
        resolve(messages[matchIndex]);
        return;
      }

      if (Date.now() - startedAt > 10_000) {
        clearInterval(timeoutId);
        cleanup();
        reject(new Error('robot bridge message timed out'));
      }
    }, 20);

    function cleanup() {
      socket.off('message', handleMessage);
      socket.off('error', handleError);
    }

    function handleMessage(buffer) {
      try {
        messages.push(JSON.parse(buffer.toString()));
      } catch (error) {
        clearInterval(timeoutId);
        cleanup();
        reject(error);
      }
    }

    function handleError(error) {
      clearInterval(timeoutId);
      cleanup();
      reject(error);
    }

    socket.on('message', handleMessage);
    socket.on('error', handleError);
  });
}

test.describe('real dynamic map tile fixture', () => {
  test.skip(!hasRealDynamicMapFixture(), 'dev large-scale .splat fixture is local and gitignored');

  test('loads real adjacent splats, switches active tile, and unmounts the previous tile entity', async ({
    baseURL,
    page
  }) => {
    test.setTimeout(120_000);

    const splatResponses = [];
    page.on('response', (response) => {
      const responseUrl = new URL(response.url());

      if (!responseUrl.pathname.startsWith('/splats/dev-large-scale/')) {
        return;
      }

      splatResponses.push({
        length: contentLength(response),
        method: response.request().method(),
        path: responseUrl.pathname,
        status: response.status()
      });
    });

    await page.goto(
      `${baseURL}/?tileCatalog=${encodeURIComponent(catalogUrl)}&tilePreload=metadata&dynamicMapDiagnostics=1`,
      { waitUntil: 'domcontentloaded' }
    );

    await expect(page.getByText('Active tile: tile_west').first()).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText(`Active splat: ${westSplatUrl}`).first()).toBeVisible();
    await expect(page.getByText(/Preload status: tile_east: ready/).first()).toBeVisible();
    await page.waitForFunction(
      (westUrl) =>
        window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
        window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === westUrl,
      westSplatUrl
    );

    await page.getByRole('button', { name: /^2\. Window$/ }).first().click();

    await expect(page.getByText('Active tile: tile_east').first()).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText(`Active splat: ${eastSplatUrl}`).first()).toBeVisible();
    await expect(page.getByText(/Preload status: tile_west: ready/).first()).toBeVisible();
    await page.waitForFunction(
      (eastUrl) =>
        window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
        window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === eastUrl,
      eastSplatUrl
    );

    const diagnostics = await page.evaluate(() => window.__dreamwalkerDynamicMapDiagnostics);
    expect(diagnostics.events.some((event) => event.type === 'unmount' && event.url === westSplatUrl)).toBe(true);

    const westSize = statSync(westSplatPath).size;
    const eastSize = statSync(eastSplatPath).size;
    const westGet = splatResponses.find(
      (response) => response.method === 'GET' && response.path === westSplatUrl && response.status === 200
    );
    const eastGet = splatResponses.find(
      (response) => response.method === 'GET' && response.path === eastSplatUrl && response.status === 200
    );

    expect(westGet).toBeTruthy();
    expect(eastGet).toBeTruthy();
    expect(westGet.length).toBe(westSize);
    expect(eastGet.length).toBe(eastSize);
  });
});

test.describe('real train/export dynamic map smoke fixture', () => {
  test.skip(!hasSmokeTrainFixture(), 'smoke train/export fixture is generated locally');

  test('loads a splat produced by large-scale-3dgs-run through the tile catalog', async ({
    baseURL,
    page
  }) => {
    const smokeGetResponse = page.waitForResponse((response) => {
      const responseUrl = new URL(response.url());

      return (
        responseUrl.pathname === smokeSplatUrl &&
        response.request().method() === 'GET' &&
        response.status() === 200
      );
    });

    await page.goto(
      `${baseURL}/?tileCatalog=${encodeURIComponent(smokeCatalogUrl)}&tilePreload=metadata&dynamicMapDiagnostics=1`,
      { waitUntil: 'domcontentloaded' }
    );

    await expect(page.getByText('Active tile: tile_x000_y000').first()).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText(`Active splat: ${smokeSplatUrl}`).first()).toBeVisible();
    await page.waitForFunction(
      (splatUrl) =>
        window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
        window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === splatUrl,
      smokeSplatUrl
    );

    const smokeGet = await smokeGetResponse;
    expect(contentLength(smokeGet)).toBe(statSync(smokeSplatPath).size);
  });
});

test.describe('real gsplat two-tile dynamic map smoke fixture', () => {
  test.skip(!hasSmokeTwoTileFixture(), 'two-tile gsplat fixture is generated locally');

  test('switches between two splats produced by the gsplat train/export path', async ({
    baseURL,
    page
  }) => {
    test.setTimeout(120_000);

    const splatResponses = [];
    page.on('response', (response) => {
      const responseUrl = new URL(response.url());

      if (!responseUrl.pathname.startsWith('/splats/smoke-gsplat-2tile/')) {
        return;
      }

      splatResponses.push({
        length: contentLength(response),
        method: response.request().method(),
        path: responseUrl.pathname,
        status: response.status()
      });
    });

    await page.goto(
      `${baseURL}/?tileCatalog=${encodeURIComponent(smokeTwoTileCatalogUrl)}&tilePreload=metadata&dynamicMapDiagnostics=1`,
      { waitUntil: 'domcontentloaded' }
    );

    await expect(page.getByText('Active tile: tile_x000_y000').first()).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText(`Active splat: ${smokeTwoTileWestSplatUrl}`).first()).toBeVisible();
    await expect(page.getByText(/Preload status: tile_x001_y000: ready/).first()).toBeVisible();
    await page.waitForFunction(
      (westSplatUrl) =>
        window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
        window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === westSplatUrl,
      smokeTwoTileWestSplatUrl
    );

    await page.getByRole('button', { name: /^2\. Window$/ }).first().click();

    await expect(page.getByText('Active tile: tile_x001_y000').first()).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText(`Active splat: ${smokeTwoTileEastSplatUrl}`).first()).toBeVisible();
    await expect(page.getByText(/Preload status: tile_x000_y000: ready/).first()).toBeVisible();
    await page.waitForFunction(
      (eastSplatUrl) =>
        window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
        window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === eastSplatUrl,
      smokeTwoTileEastSplatUrl
    );

    const diagnostics = await page.evaluate(() => window.__dreamwalkerDynamicMapDiagnostics);
    expect(
      diagnostics.events.some(
        (event) => event.type === 'unmount' && event.url === smokeTwoTileWestSplatUrl
      )
    ).toBe(true);

    const westGet = splatResponses.find(
      (response) =>
        response.method === 'GET' &&
        response.path === smokeTwoTileWestSplatUrl &&
        response.status === 200
    );
    const eastGet = splatResponses.find(
      (response) =>
        response.method === 'GET' &&
        response.path === smokeTwoTileEastSplatUrl &&
        response.status === 200
    );

    expect(westGet).toBeTruthy();
    expect(eastGet).toBeTruthy();
    expect(westGet.length).toBe(statSync(smokeTwoTileWestSplatPath).size);
    expect(eastGet.length).toBe(statSync(smokeTwoTileEastSplatPath).size);
  });
});

test.describe('real gsplat grid dynamic map smoke fixture', () => {
  test.skip(!hasSmokeGridFixture(), 'six-tile gsplat fixture is generated locally');

  test('loads an explicitly selected far tile from a six-tile gsplat catalog', async ({
    baseURL,
    page
  }) => {
    const farTileResponse = page.waitForResponse((response) => {
      const responseUrl = new URL(response.url());

      return (
        responseUrl.pathname === smokeGridFarSplatUrl &&
        response.request().method() === 'GET' &&
        response.status() === 200
      );
    });

    const pageUrl =
      `${baseURL}/?tileCatalog=${encodeURIComponent(smokeGridCatalogUrl)}` +
      `&tileId=${smokeGridFarTileId}&tilePreload=metadata&tilePreloadLimit=1` +
      '&dynamicMapDiagnostics=1';

    await page.goto(pageUrl, { waitUntil: 'domcontentloaded' });

    await expect(page.getByText('Active tile: tile_x002_z001').first()).toBeVisible({
      timeout: 60_000
    });
    await expect(page.getByText('6 ready / 6 tiles').first()).toBeVisible();
    await expect(page.getByText('Preload limit: 1').first()).toBeVisible();
    await expect(page.getByText(`Active splat: ${smokeGridFarSplatUrl}`).first()).toBeVisible();
    await page.waitForFunction(
      (splatUrl) =>
        window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
        window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === splatUrl,
      smokeGridFarSplatUrl
    );

    const response = await farTileResponse;
    const farTilePath = smokeGridSplatPaths.find((path) =>
      path.pathname.endsWith(`${smokeGridFarTileId}.splat`)
    );
    expect(farTilePath).toBeTruthy();
    expect(contentLength(response)).toBe(statSync(farTilePath).size);
  });

  test('switches through six gsplat tiles from diagnostic map position updates', async ({
    baseURL,
    page
  }) => {
    test.setTimeout(120_000);

    await page.goto(
      `${baseURL}/?tileCatalog=${encodeURIComponent(smokeGridCatalogUrl)}&tilePreload=metadata&dynamicMapDiagnostics=1`,
      { waitUntil: 'domcontentloaded' }
    );
    await page.waitForFunction(
      () => typeof window.__dreamwalkerDynamicMapDiagnostics?.setMapPosition === 'function'
    );

    for (const step of smokeGridPositionSequence) {
      const splatUrl = `/splats/smoke-gsplat-grid6/${step.tileId}.splat`;

      await page.evaluate(
        ({ position }) =>
          window.__dreamwalkerDynamicMapDiagnostics.setMapPosition(position, 'grid-test'),
        step
      );

      await expect(page.getByText(`Active tile: ${step.tileId}`).first()).toBeVisible({
        timeout: 60_000
      });
      await expect(page.getByText(`Active splat: ${splatUrl}`).first()).toBeVisible();
      await expect(page.getByText(/Tile residency: active 1 \/ preload/).first()).toBeVisible();
      await expect(
        page.getByText(new RegExp(`Tile residency list: .*${step.tileId}:active`)).first()
      ).toBeVisible();
      await expect(
        page.getByText(
          `Position grid-test: ${step.position.x}, ${step.position.y}, ${step.position.z}`
        ).first()
      ).toBeVisible();
      await page.waitForFunction(
        (activeSplatUrl) =>
          window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
          window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === activeSplatUrl,
        splatUrl
      );
    }

    const diagnostics = await page.evaluate(() => window.__dreamwalkerDynamicMapDiagnostics);
    const unmountedUrls = diagnostics.events
      .filter((event) => event.type === 'unmount')
      .map((event) => event.url);

    expect(new Set(unmountedUrls).size).toBeGreaterThanOrEqual(5);
    await expect(page.getByText(/Preload status: .* cached/).first()).toBeVisible();
  });

  test('switches through six gsplat tiles from robot bridge pose updates', async ({
    baseURL,
    page
  }) => {
    test.setTimeout(120_000);

    await page.goto(
      `${baseURL}/?tileCatalog=${encodeURIComponent(smokeGridCatalogUrl)}` +
        '&tilePreload=metadata&dynamicMapDiagnostics=1&robotBridge=1',
      { waitUntil: 'domcontentloaded' }
    );

    const bridgeSocket = new WebSocket('ws://127.0.0.1:8790/robotics');
    await waitForBridgeOpen(bridgeSocket);

    try {
      const stateSnapshotPromise = waitForBridgeMessage(
        bridgeSocket,
        (message) =>
          message.type === 'robot-state' &&
          message.protocol === robotBridgeProtocolId &&
          message.source === robotBridgeBrowserSource
      );
      bridgeSocket.send(
        JSON.stringify({
          protocol: robotBridgeProtocolId,
          type: 'request-state'
        })
      );
      await stateSnapshotPromise;

      for (const step of smokeGridPositionSequence) {
        const splatUrl = `/splats/smoke-gsplat-grid6/${step.tileId}.splat`;

        bridgeSocket.send(
          JSON.stringify({
            protocol: robotBridgeProtocolId,
            type: 'set-pose',
            pose: {
              position: [step.position.x, step.position.y, step.position.z],
              yawDegrees: 0
            },
            resetRoute: true
          })
        );

        await expect(page.locator('.dreamwalker-shell')).toHaveClass(/mode-robot/);
        await expect(page.getByText(`Active tile: ${step.tileId}`).first()).toBeVisible({
          timeout: 60_000
        });
        await expect(page.getByText(`Active splat: ${splatUrl}`).first()).toBeVisible();
        await expect(
          page.getByText(
            `Position robot: ${step.position.x}, ${step.position.y}, ${step.position.z}`
          ).first()
        ).toBeVisible();
        await expect(
          page.getByText(new RegExp(`Tile residency list: .*${step.tileId}:active`)).first()
        ).toBeVisible();
        await page.waitForFunction(
          (activeSplatUrl) =>
            window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
            window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === activeSplatUrl,
          splatUrl
        );
      }

      const diagnostics = await page.evaluate(() => window.__dreamwalkerDynamicMapDiagnostics);
      const unmountedUrls = diagnostics.events
        .filter((event) => event.type === 'unmount')
        .map((event) => event.url);

      expect(new Set(unmountedUrls).size).toBeGreaterThanOrEqual(5);
    } finally {
      bridgeSocket.close();
    }
  });

  test('plays a robot route and switches six gsplat tiles from route playback', async ({
    baseURL,
    page
  }) => {
    test.setTimeout(120_000);

    await page.goto(
      `${baseURL}/?tileCatalog=${encodeURIComponent(smokeGridCatalogUrl)}` +
        '&tilePreload=metadata&dynamicMapDiagnostics=1' +
        `&robotRoute=${encodeURIComponent(smokeGridRouteUrl)}` +
        `&robotRoutePlayback=1&robotRoutePlaybackMs=${smokeGridRoutePlaybackMs}` +
        '&robotRoutePlaybackLoop=1',
      { waitUntil: 'domcontentloaded' }
    );

    await expect(page.getByText('Route Loaded', { exact: true }).first()).toBeVisible({
      timeout: 60_000
    });
    await expect(
      page.getByText(`Robot route playback: ${smokeGridRoutePlaybackMs} ms / loop`).first()
    ).toBeVisible();
    await expect(page.locator('.dreamwalker-shell')).toHaveClass(/mode-robot/);

    for (const [index, step] of smokeGridPositionSequence.entries()) {
      const splatUrl = `/splats/smoke-gsplat-grid6/${step.tileId}.splat`;

      await expect(page.getByText(`Active tile: ${step.tileId}`).first()).toBeVisible({
        timeout: 60_000
      });
      await expect(page.getByText(`Active splat: ${splatUrl}`).first()).toBeVisible();
      if (index === 0) {
        await expect(
          page.getByText(
            'Preload tiles: tile_x000_z001, tile_x001_z001, tile_x002_z001, tile_x002_z000'
          ).first()
        ).toBeVisible();
        const initialPreloadCandidatesHandle = await page.waitForFunction(
          (expectedTileIds) => {
            const candidates =
              window.__dreamwalkerDynamicMapDiagnostics?.tilePreloadCandidates ?? [];
            const preloadCandidates = candidates.map((candidate) => ({
              role: candidate.role,
              tileId: candidate.tileId
            }));

            const hasExpectedCandidates = expectedTileIds.every(
              (tileId, candidateIndex) =>
                preloadCandidates[candidateIndex]?.tileId === tileId &&
                preloadCandidates[candidateIndex]?.role === 'route-preload-tile'
            );

            return hasExpectedCandidates ? preloadCandidates.slice(0, expectedTileIds.length) : false;
          },
          smokeGridInitialRoutePreloadTileIds
        );
        const initialPreloadCandidates = await initialPreloadCandidatesHandle.jsonValue();
        expect(initialPreloadCandidates).toEqual(
          smokeGridInitialRoutePreloadTileIds.map((tileId) => ({
            role: 'route-preload-tile',
            tileId
          }))
        );
      }
      await expect(
        page.getByText(new RegExp(`Tile residency list: .*${step.tileId}:active`)).first()
      ).toBeVisible();
      await page.waitForFunction(
        (activeSplatUrl) =>
          window.__dreamwalkerDynamicMapDiagnostics?.activeSplatUrls?.length === 1 &&
          window.__dreamwalkerDynamicMapDiagnostics.activeSplatUrls[0] === activeSplatUrl,
        splatUrl
      );
    }
  });
});
