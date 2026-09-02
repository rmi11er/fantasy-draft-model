import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders the Levered Leopolds emergency draft room", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Levered Leopolds Draft Room<\/title>/i);
  assert.match(html, /Loading emergency board/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("ships a populated static fallback contract and no starter preview", async () => {
  const payload = JSON.parse(
    await readFile(new URL("../public/draft-board.json", import.meta.url), "utf8"),
  );
  assert.equal(payload.report.franchise, "Levered Leopolds");
  assert.equal(payload.draft_order.length, 160);
  assert.ok(payload.players.length > 500);
  assert.ok(payload.report.recommendations.length > 0);
  await assert.rejects(
    access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)),
  );

  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /manual_state\.json/);
  assert.match(page, /Record pick/);
  assert.match(page, /Recompute EV/);
  assert.match(page, /Absolute odds remain uncalibrated/);
});
