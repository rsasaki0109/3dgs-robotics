import assert from 'node:assert/strict';
import { test } from 'node:test';
import { resolveDreamwalkerConfig } from '../src/app-config.js';
import {
  buildDynamicMapLoadPlan,
  collectDynamicMapTargetFragmentIds,
  preloadDynamicMapEntry,
  selectDynamicMapTile
} from '../src/dynamic-map-loading.js';

test('collectDynamicMapTargetFragmentIds includes active fragment and known gate target', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');

  assert.deepEqual(collectDynamicMapTargetFragmentIds(activeConfig), [
    'residency',
    'echo-chamber'
  ]);
});

test('collectDynamicMapTargetFragmentIds can skip gate target and dedupe extras', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');

  assert.deepEqual(
    collectDynamicMapTargetFragmentIds(activeConfig, {
      includeGateTarget: false,
      extraFragmentIds: ['residency', 'echo-chamber', 'unknown']
    }),
    ['residency', 'echo-chamber']
  );
});

test('buildDynamicMapLoadPlan resolves active map and gate preload candidate from manifest', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');
  const plan = buildDynamicMapLoadPlan(activeConfig, {
    label: 'Runtime Map Catalog',
    fragments: {
      residency: {
        label: 'Runtime Residency',
        splatUrl: '/splats/runtime-residency.sog',
        colliderMeshUrl: '/colliders/runtime-residency.glb'
      },
      'echo-chamber': {
        label: 'Runtime Echo',
        splatUrl: 'https://cdn.example.test/echo.sog'
      }
    }
  });

  assert.equal(plan.strategy, 'active-fragment-on-demand');
  assert.equal(plan.sourceLabel, 'Runtime Map Catalog');
  assert.equal(plan.active.fragmentId, 'residency');
  assert.equal(plan.active.assetLabel, 'Runtime Residency');
  assert.equal(plan.active.splatAssetKind, 'local');
  assert.equal(plan.active.colliderAssetKind, 'local');
  assert.equal(plan.preloadCandidates.length, 1);
  assert.equal(plan.preloadCandidates[0].fragmentId, 'echo-chamber');
  assert.equal(plan.preloadCandidates[0].assetLabel, 'Runtime Echo');
  assert.equal(plan.preloadCandidates[0].splatAssetKind, 'remote');
  assert.equal(
    plan.runtimeKey,
    'residency:/splats/runtime-residency.sog:/colliders/runtime-residency.glb'
  );
});

test('buildDynamicMapLoadPlan marks demo fallback when no configured splat exists', () => {
  const activeConfig = resolveDreamwalkerConfig('echo-chamber');
  const plan = buildDynamicMapLoadPlan(activeConfig, {
    fragments: {
      'echo-chamber': {
        label: 'Echo Without Splat',
        splatUrl: '',
        colliderMeshUrl: ''
      }
    }
  });

  assert.equal(plan.active.fragmentId, 'echo-chamber');
  assert.equal(plan.active.usesDemoFallback, true);
  assert.equal(plan.active.hasConfiguredSplat, false);
  assert.equal(plan.active.splatSource, 'demo');
  assert.equal(plan.active.splatAssetKind, 'remote');
});

test('buildDynamicMapLoadPlan uses ready large-scale tile catalog as active splat', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');
  const plan = buildDynamicMapLoadPlan(
    activeConfig,
    {
      fragments: {
        residency: {
          label: 'Runtime Residency',
          splatUrl: '/splats/runtime-residency.sog'
        }
      }
    },
    {
      tileCatalog: {
        sceneId: 'demo-scene',
        label: 'Demo Large Scene',
        tiles: [
          {
            id: 'tile_x000_y000',
            status: 'missing-splat',
            splatUrl: ''
          },
          {
            id: 'tile_x001_y000',
            label: 'Tile East',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_x001_y000.splat'
          },
          {
            id: 'tile_x002_y000',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_x002_y000.splat'
          }
        ]
      }
    }
  );

  assert.equal(plan.strategy, 'large-scale-3dgs-tile-catalog');
  assert.equal(plan.active.splatUrl, '/splats/demo-scene/tile_x001_y000.splat');
  assert.equal(plan.active.splatSource, 'tile-catalog');
  assert.equal(plan.active.assetLabel, 'Demo Large Scene / Tile East');
  assert.equal(plan.active.assetBundle.splatSource, 'tile-catalog');
  assert.equal(plan.active.assetBundle.usesDemoFallback, false);
  assert.equal(plan.activeTile.id, 'tile_x001_y000');
  assert.equal(plan.tileCatalog.summary.readyTileCount, 2);
  assert.deepEqual(
    plan.tilePreloadCandidates.map((entry) => entry.tileId),
    ['tile_x002_y000']
  );
  assert.equal(
    plan.runtimeKey,
    'residency:tile_x001_y000:/splats/demo-scene/tile_x001_y000.splat:'
  );
});

test('buildDynamicMapLoadPlan limits tile preload candidates', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');
  const plan = buildDynamicMapLoadPlan(
    activeConfig,
    {
      fragments: {
        residency: {
          label: 'Runtime Residency',
          splatUrl: '/splats/runtime-residency.sog'
        }
      }
    },
    {
      maxTilePreloadCandidates: 1,
      position: { x: 1, y: 0, z: 1 },
      tileCatalog: {
        sceneId: 'demo-scene',
        label: 'Demo Large Scene',
        tiles: [
          {
            id: 'tile_active',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_active.splat',
            coreBounds: { minX: 0, maxX: 2, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_near',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_near.splat',
            coreBounds: { minX: 2, maxX: 4, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_far',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/demo-scene/tile_far.splat',
            coreBounds: { minX: 20, maxX: 22, minZ: 0, maxZ: 2 }
          }
        ]
      }
    }
  );

  assert.equal(plan.activeTile.id, 'tile_active');
  assert.deepEqual(
    plan.tilePreloadCandidates.map((entry) => entry.tileId),
    ['tile_near']
  );
});

test('buildDynamicMapLoadPlan prioritizes adjacent tileIndex preload candidates', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');
  const plan = buildDynamicMapLoadPlan(
    activeConfig,
    {
      fragments: {
        residency: {
          label: 'Runtime Residency',
          splatUrl: '/splats/runtime-residency.sog'
        }
      }
    },
    {
      maxTilePreloadCandidates: 3,
      position: { x: 1, y: 0, z: 1 },
      tileCatalog: {
        sceneId: 'grid-scene',
        label: 'Grid Scene',
        tiles: [
          {
            id: 'tile_active',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_active.splat',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: 0, maxX: 2, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_cardinal',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_cardinal.splat',
            tileIndex: { x: 1, z: 0 },
            coreBounds: { minX: 30, maxX: 32, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_diagonal',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_diagonal.splat',
            tileIndex: { x: 1, z: 1 },
            coreBounds: { minX: 2, maxX: 4, minZ: 2, maxZ: 4 }
          },
          {
            id: 'tile_far',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_far.splat',
            tileIndex: { x: 2, z: 0 },
            coreBounds: { minX: 4, maxX: 6, minZ: 0, maxZ: 2 }
          }
        ]
      }
    }
  );

  assert.deepEqual(
    plan.tilePreloadCandidates.map((entry) => entry.tileId),
    ['tile_cardinal', 'tile_diagonal', 'tile_far']
  );
});

test('buildDynamicMapLoadPlan prioritizes route preview tiles before adjacent tiles', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');
  const plan = buildDynamicMapLoadPlan(
    activeConfig,
    {
      fragments: {
        residency: {
          label: 'Runtime Residency',
          splatUrl: '/splats/runtime-residency.sog'
        }
      }
    },
    {
      maxTilePreloadCandidates: 2,
      position: { x: 1, y: 0, z: 1 },
      routePreviewPositions: [
        { x: 11, y: 0, z: 1 },
        { x: 5, y: 0, z: 1 }
      ],
      tileCatalog: {
        sceneId: 'route-grid-scene',
        label: 'Route Grid Scene',
        tiles: [
          {
            id: 'tile_active',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_active.splat',
            tileIndex: { x: 0, z: 0 },
            coreBounds: { minX: 0, maxX: 2, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_adjacent',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_adjacent.splat',
            tileIndex: { x: 1, z: 0 },
            coreBounds: { minX: 2, maxX: 4, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_route_mid',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_route_mid.splat',
            tileIndex: { x: 2, z: 0 },
            coreBounds: { minX: 4, maxX: 6, minZ: 0, maxZ: 2 }
          },
          {
            id: 'tile_route_far',
            axes: 'xz',
            status: 'ready',
            splatUrl: '/splats/grid/tile_route_far.splat',
            tileIndex: { x: 5, z: 0 },
            coreBounds: { minX: 10, maxX: 12, minZ: 0, maxZ: 2 }
          }
        ]
      }
    }
  );

  assert.deepEqual(
    plan.tilePreloadCandidates.map((entry) => entry.tileId),
    ['tile_route_far', 'tile_route_mid']
  );
  assert.deepEqual(
    plan.tilePreloadCandidates.map((entry) => entry.role),
    ['route-preload-tile', 'route-preload-tile']
  );
});

test('selectDynamicMapTile honors an explicit ready tile id', () => {
  const selected = selectDynamicMapTile(
    {
      tiles: [
        {
          id: 'tile_a',
          status: 'ready',
          splatUrl: '/splats/tile_a.splat'
        },
        {
          id: 'tile_b',
          status: 'ready',
          splatUrl: '/splats/tile_b.splat'
        }
      ]
    },
    { tileId: 'tile_b' }
  );

  assert.equal(selected.id, 'tile_b');
  assert.equal(selected.splatUrl, '/splats/tile_b.splat');
});

test('selectDynamicMapTile chooses a ready tile from map position', () => {
  const selected = selectDynamicMapTile(
    {
      tiles: [
        {
          id: 'tile_west',
          axes: 'xz',
          status: 'ready',
          splatUrl: '/splats/tile_west.splat',
          coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
          expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
        },
        {
          id: 'tile_east',
          axes: 'xz',
          status: 'ready',
          splatUrl: '/splats/tile_east.splat',
          coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
          expandedBounds: { minX: -2, maxX: 12, minZ: -2, maxZ: 12 }
        }
      ]
    },
    {
      position: { x: 6, y: 0, z: 5 }
    }
  );

  assert.equal(selected.id, 'tile_east');
});

test('selectDynamicMapTile keeps current tile while position remains in expanded bounds', () => {
  const selected = selectDynamicMapTile(
    {
      tiles: [
        {
          id: 'tile_west',
          axes: 'xz',
          status: 'ready',
          splatUrl: '/splats/tile_west.splat',
          coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
          expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
        },
        {
          id: 'tile_east',
          axes: 'xz',
          status: 'ready',
          splatUrl: '/splats/tile_east.splat',
          coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
          expandedBounds: { minX: -2, maxX: 12, minZ: -2, maxZ: 12 }
        }
      ]
    },
    {
      currentTileId: 'tile_west',
      position: { x: 1.5, y: 0, z: 5 }
    }
  );

  assert.equal(selected.id, 'tile_west');
});

test('buildDynamicMapLoadPlan switches auto tile after leaving current expanded bounds', () => {
  const activeConfig = resolveDreamwalkerConfig('residency');
  const tileCatalog = {
    sceneId: 'demo-scene',
    label: 'Demo Large Scene',
    tiles: [
      {
        id: 'tile_west',
        axes: 'xz',
        status: 'ready',
        splatUrl: '/splats/tile_west.splat',
        coreBounds: { minX: -10, maxX: 0, minZ: 0, maxZ: 10 },
        expandedBounds: { minX: -12, maxX: 2, minZ: -2, maxZ: 12 }
      },
      {
        id: 'tile_east',
        axes: 'xz',
        status: 'ready',
        splatUrl: '/splats/tile_east.splat',
        coreBounds: { minX: 0, maxX: 10, minZ: 0, maxZ: 10 },
        expandedBounds: { minX: -2, maxX: 12, minZ: -2, maxZ: 12 }
      }
    ]
  };
  const plan = buildDynamicMapLoadPlan(
    activeConfig,
    { fragments: { residency: { label: 'Runtime Residency' } } },
    {
      currentTileId: 'tile_west',
      position: { x: 4, y: 0, z: 5 },
      tileCatalog
    }
  );

  assert.equal(plan.activeTile.id, 'tile_east');
  assert.equal(plan.active.splatUrl, '/splats/tile_east.splat');
});

test('preloadDynamicMapEntry checks tile splat metadata with HEAD', async () => {
  const calls = [];
  const result = await preloadDynamicMapEntry(
    {
      fragmentId: 'residency',
      fragmentLabel: 'Residency',
      splatUrl: '/splats/tile_east.splat'
    },
    {
      fetchImpl: async (url, options) => {
        calls.push({ url, method: options.method });
        return { ok: true, status: 200, statusText: 'OK' };
      },
      mode: 'metadata'
    }
  );

  assert.equal(result.status, 'ready');
  assert.deepEqual(calls, [{ url: '/splats/tile_east.splat', method: 'HEAD' }]);
});

test('preloadDynamicMapEntry falls back to range GET when HEAD is unavailable', async () => {
  const calls = [];
  const result = await preloadDynamicMapEntry(
    {
      fragmentId: 'residency',
      fragmentLabel: 'Residency',
      splatUrl: '/splats/tile_east.splat'
    },
    {
      fetchImpl: async (url, options) => {
        calls.push({
          url,
          method: options.method,
          range: options.headers?.Range ?? ''
        });

        if (options.method === 'HEAD') {
          return { ok: false, status: 405, statusText: 'Method Not Allowed' };
        }

        return { ok: true, status: 206, statusText: 'Partial Content' };
      },
      mode: 'metadata'
    }
  );

  assert.equal(result.status, 'ready');
  assert.equal(result.assets[0].method, 'GET');
  assert.deepEqual(calls, [
    { url: '/splats/tile_east.splat', method: 'HEAD', range: '' },
    { url: '/splats/tile_east.splat', method: 'GET', range: 'bytes=0-0' }
  ]);
});
