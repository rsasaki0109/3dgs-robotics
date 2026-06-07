import { expect, test } from '@playwright/test';

const outdoorDemoCatalogUrl = '/manifests/outdoor-demo-dust3r-tile-catalog.json';
const outdoorDemoRouteUrl = '/robot-routes/outdoor-demo-dust3r-tile-loop.json';
const outdoorDemoTileRoutePattern = '**/splats/outdoor-demo-dust3r-tiled/*.splat';

function buildTileCatalogDataUrl() {
  const catalog = {
    version: 1,
    type: 'large-scale-3dgs-tile-catalog',
    sceneId: 'browser-test-large-scale',
    label: 'Browser Test Large-scale Tiles',
    tiling: {
      strategy: 'browser-test-fixture',
      axes: 'xz',
      tileSize: 8,
      overlap: 2
    },
    tiles: [
      {
        id: 'tile_west',
        label: 'Browser Test West',
        axes: 'xz',
        status: 'ready',
        runStatus: 'test',
        splatUrl: '/splats/browser-test/tile_west.splat',
        coreBounds: { minX: -6, maxX: 2, minZ: 0, maxZ: 12 },
        expandedBounds: { minX: -8, maxX: 3.4, minZ: -2, maxZ: 14 }
      },
      {
        id: 'tile_east',
        label: 'Browser Test East',
        axes: 'xz',
        status: 'ready',
        runStatus: 'test',
        splatUrl: '/splats/browser-test/tile_east.splat',
        coreBounds: { minX: 2, maxX: 10, minZ: 0, maxZ: 12 },
        expandedBounds: { minX: 0.6, maxX: 12, minZ: -2, maxZ: 14 }
      }
    ]
  };

  return `data:application/json,${encodeURIComponent(JSON.stringify(catalog))}`;
}

test('large-scale demo button launches the bundled outdoor tile catalog', async ({
  page,
  baseURL
}) => {
  await page.route(outdoorDemoTileRoutePattern, async (route) => {
    const request = route.request();

    if (request.method() === 'HEAD') {
      await route.fulfill({
        headers: {
          'Content-Length': '32'
        },
        status: 200
      });
      return;
    }

    await route.fulfill({
      body: Buffer.alloc(32),
      contentType: 'application/octet-stream',
      status: 200
    });
  });

  await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Open Large-scale Demo' }).click();

  await expect(page).toHaveURL(
    new RegExp(`tileCatalog=.*${encodeURIComponent(outdoorDemoCatalogUrl)}`)
  );
  await expect(page).toHaveURL(new RegExp(`robotRoute=.*${encodeURIComponent(outdoorDemoRouteUrl)}`));
  await expect(page.getByText('Outdoor Demo DUSt3R Tiled').first()).toBeVisible();
  await expect(page.getByText('Route Loaded', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('Robot route playback: 1200 ms / loop').first()).toBeVisible();
  await expect(page.locator('.dreamwalker-shell')).toHaveClass(/mode-robot/);
  await expect(page.getByText('4 ready / 4 tiles').first()).toBeVisible();
  await expect(page.getByText('Preload limit: 2').first()).toBeVisible();
  await expect(page.getByText('Resident limit: 3').first()).toBeVisible();
  await expect(page.getByText(/Active tile: tile_x\d{3}_z\d{3}/).first()).toBeVisible();
  await expect(
    page
      .getByText(/Active splat: \/splats\/outdoor-demo-dust3r-tiled\/tile_x\d{3}_z\d{3}\.splat/)
      .first()
  ).toBeVisible();
});

test('dynamic map tile catalog auto-switches and preloads adjacent tile metadata', async ({ page, baseURL }) => {
  const splatRequests = [];

  await page.route('**/splats/browser-test/*.splat', async (route) => {
    const request = route.request();
    splatRequests.push({
      method: request.method(),
      url: request.url()
    });

    if (request.method() === 'HEAD') {
      await route.fulfill({
        headers: {
          'Content-Length': '32'
        },
        status: 200
      });
      return;
    }

    await route.fulfill({
      body: Buffer.alloc(32),
      contentType: 'application/octet-stream',
      status: 200
    });
  });

  await page.goto(
    `${baseURL}/?tileCatalog=${encodeURIComponent(buildTileCatalogDataUrl())}`,
    { waitUntil: 'domcontentloaded' }
  );

  await expect(page.getByText('Tile Ready').first()).toBeVisible();
  await expect(page.getByText('Active tile: tile_west').first()).toBeVisible();
  await expect(page.getByText('Preload Ready').first()).toBeVisible();
  await expect(page.getByText(/Preload status: tile_east: ready/).first()).toBeVisible();

  await page.getByRole('button', { name: /^2\. Window$/ }).first().click();

  await expect(page.getByText('Active tile: tile_east').first()).toBeVisible();
  await expect(page.getByText(/Preload status: tile_west: ready/).first()).toBeVisible();

  expect(splatRequests.some((request) => request.method === 'HEAD')).toBe(true);
});
