import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import net from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';

const REPO_ROOT = new URL('../../../', import.meta.url);

async function reservePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  await new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  if (!port) {
    throw new Error('Failed to reserve an ephemeral port for docs viewer smoke test');
  }
  return port;
}

async function waitForServer(url, child, timeoutMs = 10000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (child.exitCode !== null) {
      throw new Error(`docs viewer server exited early with code ${child.exitCode}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // retry
    }
    await delay(200);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

test('docs viewer loads the published outdoor scene bundle', async () => {
  const docsPort = await reservePort();
  const docsUrl = `http://127.0.0.1:${docsPort}/`;
  const server = spawn('python3', ['-m', 'http.server', String(docsPort), '-d', 'docs'], {
    cwd: REPO_ROOT,
    stdio: 'ignore',
  });

  let browser;
  try {
    await waitForServer(docsUrl, server);
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto(docsUrl, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => {
      const pointCount = document.getElementById('point-count')?.textContent || '';
      const status = document.getElementById('viewer-status')?.textContent || '';
      return /points/.test(pointCount) && pointCount !== '0 points' && /Showing Outdoor GS Demo/.test(status);
    });

    const snapshot = await page.evaluate(() => ({
      pointCount: document.getElementById('point-count')?.textContent || '',
      status: document.getElementById('viewer-status')?.textContent || '',
      tabs: Array.from(document.querySelectorAll('.scene-tabs .scene-tab')).map((element) =>
        element.textContent.trim()
      ),
    }));

    assert.equal(snapshot.status, 'Showing Outdoor GS Demo from published');
    assert.ok(/\d[\d,]* points/.test(snapshot.pointCount));
    assert.deepEqual(snapshot.tabs, ['Outdoor GS Demo']);
  } finally {
    if (browser) {
      await browser.close();
    }
    if (server.exitCode === null) {
      server.kill('SIGTERM');
      await Promise.race([
        new Promise((resolve) => server.once('exit', resolve)),
        delay(1000),
      ]);
    }
  }
});
