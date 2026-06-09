import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  buildDynamicMapCatalogLaunchUrl,
  validateDynamicMapCatalog
} from '../tools/validate-dynamic-map-catalog.mjs';

const appPublicRoot = fileURLToPath(new URL('../public', import.meta.url));

async function createPublicRoot() {
  const root = await mkdtemp(path.join(tmpdir(), 'dreamwalker-catalog-'));
  const publicRoot = path.join(root, 'public');
  await mkdir(path.join(publicRoot, 'manifests'), { recursive: true });
  await mkdir(path.join(publicRoot, 'robot-routes'), { recursive: true });
  await mkdir(path.join(publicRoot, 'splats', 'demo-scene'), { recursive: true });

  return {
    root,
    publicRoot,
    async cleanup() {
      await rm(root, { force: true, recursive: true });
    }
  };
}

async function writeJson(filePath, value) {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function createCatalog(overrides = {}) {
  return {
    version: 1,
    type: 'large-scale-3dgs-tile-catalog',
    sceneId: 'demo-scene',
    label: 'Demo Scene',
    tiling: {
      strategy: 'test-grid',
      axes: 'xz',
      tileSize: 10,
      overlap: 2,
      minImages: 1,
      worldBounds: { minX: -10, maxX: 10, minZ: 0, maxZ: 10 }
    },
    tiles: [
      {
        id: 'tile_west',
        status: 'ready',
        splatUrl: '/splats/demo-scene/tile_west.splat',
        axes: 'xz',
        tileIndex: { x: 0, z: 0 },
        coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
        expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
      },
      {
        id: 'tile_east',
        status: 'ready',
        splatUrl: '/splats/demo-scene/tile_east.splat',
        axes: 'xz',
        tileIndex: { x: 1, z: 0 },
        coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
        expandedBounds: { minX: -2, maxX: 12, minZ: -2, maxZ: 12 }
      }
    ],
    ...overrides
  };
}

function createRoute(route = [[-5, 0, 5], [5, 0, 5]]) {
  const lastPoint = route[route.length - 1] ?? [0, 0, 0];

  return {
    version: 1,
    protocol: 'dreamwalker-robot-route/v1',
    label: 'Demo Route',
    fragmentId: 'residency',
    frameId: 'dreamwalker_map',
    pose: {
      position: lastPoint,
      yawDegrees: 90
    },
    route
  };
}

test('validateDynamicMapCatalog validates ready local splats and prints a launch URL', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_east.splat'), 'east');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(catalogPath, createCatalog());

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot,
      siteUrl: 'http://127.0.0.1:5173/?fragment=residency',
      preloadMode: 'cache'
    });

    assert.equal(result.ok, true);
    assert.equal(result.errorCount, 0);
    assert.equal(result.warningCount, 0);
    assert.equal(result.catalog.summary.readyTileCount, 2);
    assert.ok(result.findings.some((finding) => finding.scope === 'catalog:grid'));
    assert.ok(result.findings.some((finding) => finding.scope === 'catalog:adjacency'));
    assert.match(result.launchUrl, /^http:\/\/127\.0\.0\.1:5173\/\?fragment=residency&/);
    assert.match(result.launchUrl, /tileCatalog=%2Fmanifests%2Fcatalog\.json/);
    assert.match(result.launchUrl, /tilePreload=cache/);
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog fails when a ready local splat is missing', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(catalogPath, createCatalog());

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot
    });

    assert.equal(result.ok, false);
    assert.equal(result.errorCount, 1);
    assert.ok(result.findings.some((finding) => finding.detail.includes('tile_east.splat')));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog checks optional viewer splat assets', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_east.splat'), 'east');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.ply'), 'ply');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(
      catalogPath,
      createCatalog({
        tiles: [
          {
            id: 'tile_west',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_west.splat',
            viewerSplatUrl: '/splats/demo-scene/tile_west.ply',
            axes: 'xz',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
          },
          {
            id: 'tile_east',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_east.splat',
            viewerSplatUrl: '/splats/demo-scene/tile_east.ply',
            axes: 'xz',
            tileIndex: { x: 1, z: 0 },
            coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: -2, maxX: 12, minZ: -2, maxZ: 12 }
          }
        ]
      })
    );

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot
    });

    assert.equal(result.ok, false);
    assert.equal(result.errorCount, 1);
    assert.ok(result.findings.some((finding) => finding.detail.includes('tile_east.ply')));
    assert.ok(result.findings.some((finding) => finding.detail.includes('tile_west.ply')));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog fails when the catalog has no ready tiles', async () => {
  const fixture = await createPublicRoot();

  try {
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(
      catalogPath,
      createCatalog({
        tiles: [
          {
            id: 'tile_missing',
            status: 'missing-splat',
            splatUrl: ''
          }
        ]
      })
    );

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot
    });

    assert.equal(result.ok, false);
    assert.equal(result.errorCount, 1);
    assert.ok(result.findings.some((finding) => finding.scope === 'catalog:ready'));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog fails on duplicate ready splat URLs and tile indexes', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(
      catalogPath,
      createCatalog({
        tiles: [
          {
            id: 'tile_west',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_west.splat',
            axes: 'xz',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
          },
          {
            id: 'tile_east',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_west.splat',
            axes: 'xz',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: -2, maxX: 12, minZ: -2, maxZ: 12 }
          }
        ]
      })
    );

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot
    });

    assert.equal(result.ok, false);
    assert.ok(result.findings.some((finding) => finding.detail.includes('duplicate splatUrl')));
    assert.ok(result.findings.some((finding) => finding.detail.includes('duplicate tileIndex')));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog fails on invalid tile bounds', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_east.splat'), 'east');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(
      catalogPath,
      createCatalog({
        tiles: [
          {
            id: 'tile_west',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_west.splat',
            axes: 'xz',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
          },
          {
            id: 'tile_east',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_east.splat',
            axes: 'xz',
            tileIndex: { x: 1, z: 0 },
            coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: 1, maxX: 8, minZ: 2, maxZ: 8 }
          }
        ]
      })
    );

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot
    });

    assert.equal(result.ok, false);
    assert.ok(result.findings.some((finding) => finding.detail.includes('expandedBounds does not contain coreBounds')));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog warns about sparse rectangular tile indexes', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_far_east.splat'), 'far east');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    await writeJson(
      catalogPath,
      createCatalog({
        tiling: {
          strategy: 'test-grid',
          axes: 'xz',
          tileSize: 10,
          overlap: 0,
          minImages: 1,
          worldBounds: { minX: -10, maxX: 30, minZ: 0, maxZ: 10 }
        },
        tiles: [
          {
            id: 'tile_west',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_west.splat',
            axes: 'xz',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 }
          },
          {
            id: 'tile_far_east',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_far_east.splat',
            axes: 'xz',
            tileIndex: { x: 2, z: 0 },
            coreBounds: { minX: 20, maxX: 30, minZ: 0, maxZ: 10 },
            expandedBounds: { minX: 20, maxX: 30, minZ: 0, maxZ: 10 }
          }
        ]
      })
    );

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot
    });

    assert.equal(result.ok, true);
    assert.ok(result.warningCount >= 1);
    assert.ok(result.findings.some((finding) => finding.scope === 'catalog:grid' && finding.detail.includes('missing x1/z0')));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog validates robot route coverage and prints a playback launch URL', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_east.splat'), 'east');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    const routePath = path.join(fixture.publicRoot, 'robot-routes', 'demo-route.json');
    await writeJson(catalogPath, createCatalog());
    await writeJson(routePath, createRoute());

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot,
      routeInput: routePath,
      routePlayback: true,
      routePlaybackLoop: true,
      routePlaybackMs: 500,
      siteUrl: 'http://127.0.0.1:5173/'
    });

    assert.equal(result.ok, true);
    assert.equal(result.errorCount, 0);
    assert.deepEqual(result.routeTileSequence, ['tile_west', 'tile_east']);
    assert.ok(result.findings.some((finding) =>
      finding.scope === 'route:coverage' &&
      finding.detail.includes('all route points')
    ));
    assert.match(result.launchUrl, /robotRoute=%2Frobot-routes%2Fdemo-route\.json/);
    assert.match(result.launchUrl, /robotRoutePlayback=1/);
    assert.match(result.launchUrl, /robotRoutePlaybackMs=500/);
    assert.match(result.launchUrl, /robotRoutePlaybackLoop=1/);
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog validates viewer XZ route coverage for XY source tiles', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_a.splat'), 'a');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_b.splat'), 'b');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_a.ply'), 'ply:a');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_b.ply'), 'ply:b');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    const routePath = path.join(fixture.publicRoot, 'robot-routes', 'demo-route.json');
    await writeJson(
      catalogPath,
      createCatalog({
        tiling: {
          strategy: 'test-grid',
          axes: 'xy',
          viewerAxes: 'xz',
          tileSize: 10,
          overlap: 2,
          minImages: 1,
          worldBounds: { minX: 0, maxX: 20, minY: -20, maxY: -10 },
          viewerWorldBounds: { minX: 0, maxX: 20, minZ: -20, maxZ: -10 }
        },
        tiles: [
          {
            id: 'tile_a',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_a.splat',
            viewerSplatUrl: '/splats/demo-scene/tile_a.ply',
            axes: 'xy',
            viewerAxes: 'xz',
            tileIndex: { x: 0, y: 1 },
            viewerTileIndex: { x: 0, z: 1 },
            coreBounds: { minX: 0, maxX: 10, minY: -20, maxY: -10 },
            expandedBounds: { minX: -2, maxX: 12, minY: -22, maxY: -8 },
            viewerCoreBounds: { minX: 0, maxX: 10, minZ: -20, maxZ: -10 },
            viewerExpandedBounds: { minX: -2, maxX: 12, minZ: -22, maxZ: -8 }
          },
          {
            id: 'tile_b',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_b.splat',
            viewerSplatUrl: '/splats/demo-scene/tile_b.ply',
            axes: 'xy',
            viewerAxes: 'xz',
            tileIndex: { x: 1, y: 1 },
            viewerTileIndex: { x: 1, z: 1 },
            coreBounds: { minX: 10, maxX: 20, minY: -20, maxY: -10 },
            expandedBounds: { minX: 8, maxX: 22, minY: -22, maxY: -8 },
            viewerCoreBounds: { minX: 10, maxX: 20, minZ: -20, maxZ: -10 },
            viewerExpandedBounds: { minX: 8, maxX: 22, minZ: -22, maxZ: -8 }
          }
        ]
      })
    );
    await writeJson(routePath, createRoute([[5, 0, -15], [15, 0, -15]]));

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot,
      routeInput: routePath
    });

    assert.equal(result.ok, true);
    assert.equal(result.errorCount, 0);
    assert.deepEqual(result.routeTileSequence, ['tile_a', 'tile_b']);
    assert.ok(result.findings.some((finding) =>
      finding.scope === 'route:coverage' &&
      finding.detail.includes('all route points')
    ));
  } finally {
    await fixture.cleanup();
  }
});

test('validateDynamicMapCatalog validates the bundled large outdoor route playback demo', async () => {
  const result = await validateDynamicMapCatalog(
    '/manifests/outdoor-production-grid-large-tile-catalog.json',
    {
      publicRoot: appPublicRoot,
      routeInput: '/robot-routes/outdoor-production-grid-large-route.json',
      routePlayback: true,
      routePlaybackLoop: true,
      routePlaybackMs: 1200,
      siteUrl: 'http://127.0.0.1:5173/'
    }
  );

  assert.equal(result.ok, true);
  assert.equal(result.errorCount, 0);
  assert.equal(result.routeTileSequence.length, 87);
  assert.equal(result.routeTileSequence[0], 'tile_x000_z001');
  assert.equal(result.routeTileSequence.at(-1), 'tile_x023_z012');
  assert.ok(result.findings.some((finding) =>
    finding.scope === 'route:coverage' &&
    finding.detail.includes('all route points')
  ));
  assert.match(
    result.launchUrl,
    /tileCatalog=%2Fmanifests%2Foutdoor-production-grid-large-tile-catalog\.json/
  );
  assert.match(
    result.launchUrl,
    /robotRoute=%2Frobot-routes%2Foutdoor-production-grid-large-route\.json/
  );
  assert.match(result.launchUrl, /robotRoutePlayback=1/);
  assert.match(result.launchUrl, /robotRoutePlaybackMs=1200/);
  assert.match(result.launchUrl, /robotRoutePlaybackLoop=1/);
});

test('validateDynamicMapCatalog fails when a robot route leaves ready tile coverage', async () => {
  const fixture = await createPublicRoot();

  try {
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_west.splat'), 'west');
    await writeFile(path.join(fixture.publicRoot, 'splats', 'demo-scene', 'tile_east.splat'), 'east');
    const catalogPath = path.join(fixture.publicRoot, 'manifests', 'catalog.json');
    const routePath = path.join(fixture.publicRoot, 'robot-routes', 'bad-route.json');
    await writeJson(catalogPath, createCatalog());
    await writeJson(routePath, createRoute([[-5, 0, 5], [40, 0, 5]]));

    const result = await validateDynamicMapCatalog(catalogPath, {
      publicRoot: fixture.publicRoot,
      routeInput: routePath
    });

    assert.equal(result.ok, false);
    assert.ok(result.findings.some((finding) =>
      finding.scope === 'route:point:2' &&
      finding.detail.includes('outside ready tile coverage')
    ));
  } finally {
    await fixture.cleanup();
  }
});

test('buildDynamicMapCatalogLaunchUrl preserves existing query and supports tile override', () => {
  const launchUrl = buildDynamicMapCatalogLaunchUrl('/manifests/catalog.json', {
    siteUrl: 'http://example.test/dreamwalker?fragment=residency',
    preloadMode: 'metadata',
    tileId: 'tile_east'
  });

  assert.equal(
    launchUrl,
    'http://example.test/dreamwalker?fragment=residency&tileCatalog=%2Fmanifests%2Fcatalog.json&tilePreload=metadata&tileId=tile_east'
  );
});

test('buildDynamicMapCatalogLaunchUrl supports relative site URLs', () => {
  const launchUrl = buildDynamicMapCatalogLaunchUrl('/manifests/catalog.json', {
    siteUrl: '/',
    preloadMode: 'off'
  });

  assert.equal(
    launchUrl,
    '/?tileCatalog=%2Fmanifests%2Fcatalog.json&tilePreload=off'
  );
});
