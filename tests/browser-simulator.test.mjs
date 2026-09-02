import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { recomputeDraftEV } from "../app/draft-simulator.ts";

const board = JSON.parse(
  await readFile(new URL("../public/draft-board.json", import.meta.url), "utf8"),
);

test("browser simulator prices the user's next pick from static state", () => {
  const result = recomputeDraftEV({
    players: board.players,
    draftOrder: board.draft_order,
    manual: {},
    xouts: new Set(board.xouts),
    simulations: 5,
  });
  assert.equal(result.targetOverall, 2);
  assert.equal(result.recommendations.length, 8);
  assert.ok(result.recommendations.every((row) => row.payout_equity >= 0));
});

test("manual 1.01 changes the available pool before recomputation", () => {
  const gibbs = board.players.find((player) => player.name === "Jahmyr Gibbs");
  const result = recomputeDraftEV({
    players: board.players,
    draftOrder: board.draft_order,
    manual: { "1": gibbs.mfl_player_id },
    xouts: new Set(board.xouts),
    simulations: 10,
  });
  assert.equal(result.targetOverall, 2);
  assert.ok(result.recommendations.every((row) => row.player_id !== gibbs.mfl_player_id));
  assert.equal(result.recommendations[0].availability_probability, 1);
});
