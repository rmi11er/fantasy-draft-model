export type BrowserPlayer = {
  mfl_player_id: string;
  name: string;
  position: string;
  team: string;
  display_adp: number;
  mean_season_points: number;
  sd_season_points: number;
};

export type BrowserPick = {
  overall: number;
  round: number;
  slot: number;
  franchise_id: string;
  player_id: string | null;
};

export type BrowserRecommendation = {
  rank: number;
  player_id: string;
  name: string;
  position: string;
  team: string;
  adp: number;
  payout_equity: number;
  marginal_ev: number;
  availability_probability: number;
  first_probability: number;
  second_probability: number;
  third_probability: number;
  marginal_ev_standard_error: number;
  best_ev_gap_standard_error: number;
  statistically_tied_for_best: boolean;
};

export type BrowserSimulationResult = {
  recommendations: BrowserRecommendation[];
  referencePlayer: string;
  referencePayoutEquity: number;
  simulations: number;
  targetOverall: number;
};

type Position = "QB" | "RB" | "WR" | "TE" | "PK" | "DEF";
type Player = BrowserPlayer & { positionKey: Position; index: number };
type Scenario = {
  noise: Float32Array;
  outcomes: Float32Array;
  externalScenarios: Uint16Array;
  externalOpponents: Uint8Array;
};
type Equity = {
  payout: number;
  first: number;
  second: number;
  third: number;
  availability: number;
  commonSelection: string;
  payoutSamples: number[];
};

const POSITIONS: Position[] = ["QB", "RB", "WR", "TE", "PK", "DEF"];
const MINIMUMS: Record<Position, number> = { QB: 1, RB: 2, WR: 3, TE: 1, PK: 1, DEF: 1 };
const CAPS: Record<Position, number> = { QB: 2, RB: 8, WR: 8, TE: 4, PK: 2, DEF: 2 };
const DEMAND: Record<Position, number> = { QB: 1, RB: 3, WR: 4, TE: 1, PK: 1, DEF: 1 };
const USER_ID = "0016";
const FIELD_SIZE = 20;

function positionKey(value: string): Position {
  const normalized = value.toUpperCase().replaceAll("/", "");
  if (normalized === "K") return "PK";
  if (normalized === "DST" || normalized === "D-ST" || normalized === "DEF") return "DEF";
  if (POSITIONS.includes(normalized as Position)) return normalized as Position;
  throw new Error(`Unsupported position ${value}`);
}

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random: () => number) {
  const first = Math.max(random(), Number.EPSILON);
  const second = random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

function pairedStandardError(first: number[], second: number[]) {
  if (first.length < 2 || first.length !== second.length) return 0;
  const differences = first.map((value, index) => value - second[index]);
  const mean = differences.reduce((total, value) => total + value, 0) / differences.length;
  const variance = differences.reduce((total, value) => total + (value - mean) ** 2, 0) / (differences.length - 1);
  return Math.sqrt(variance / differences.length);
}

function counts(roster: string[], byId: Map<string, Player>) {
  const result: Record<Position, number> = { QB: 0, RB: 0, WR: 0, TE: 0, PK: 0, DEF: 0 };
  roster.forEach((id) => { result[byId.get(id)!.positionKey] += 1; });
  return result;
}

function feasible(position: Position, roster: string[], picksAfter: number, byId: Map<string, Player>) {
  const rosterCounts = counts(roster, byId);
  if (rosterCounts[position] >= CAPS[position]) return false;
  rosterCounts[position] += 1;
  return POSITIONS.reduce((total, key) => total + Math.max(0, MINIMUMS[key] - rosterCounts[key]), 0) <= picksAfter;
}

function samplePlayer(player: Player, z: number) {
  const mean = player.mean_season_points;
  const sd = player.sd_season_points || 0;
  if (!mean || !sd) return mean;
  const ratio = (sd / mean) ** 2;
  const logVariance = Math.log1p(ratio);
  return Math.exp(Math.log(mean) - logVariance / 2 + Math.sqrt(logVariance) * z);
}

function rosterScore(roster: string[], scenario: Scenario, byId: Map<string, Player>) {
  const values: Record<Position, number[]> = { QB: [], RB: [], WR: [], TE: [], PK: [], DEF: [] };
  roster.forEach((id) => {
    const player = byId.get(id)!;
    values[player.positionKey].push(samplePlayer(player, scenario.outcomes[player.index]));
  });
  POSITIONS.forEach((position) => values[position].sort((a, b) => b - a));
  if (!values.QB.length || !values.PK.length || !values.DEF.length) return -1e9;
  let bestSkill = -1e9;
  for (let rb = 2; rb <= 4; rb += 1) {
    for (let wr = 3; wr <= 5; wr += 1) {
      for (let te = 1; te <= 3; te += 1) {
        if (rb + wr + te !== 8 || values.RB.length < rb || values.WR.length < wr || values.TE.length < te) continue;
        const skill = values.RB.slice(0, rb).reduce((a, b) => a + b, 0)
          + values.WR.slice(0, wr).reduce((a, b) => a + b, 0)
          + values.TE.slice(0, te).reduce((a, b) => a + b, 0);
        bestSkill = Math.max(bestSkill, skill);
      }
    }
  }
  return values.QB[0] + values.PK[0] + values.DEF[0] + bestSkill;
}

function selectPlayerPool(allPlayers: BrowserPlayer[], required: Set<string>) {
  const valid = allPlayers.filter((player) =>
    Number.isFinite(player.display_adp) && player.display_adp > 0
    && Number.isFinite(player.mean_season_points) && player.mean_season_points > 0
  );
  const ids = new Set(valid.slice().sort((a, b) => a.display_adp - b.display_adp).slice(0, 260).map((p) => p.mfl_player_id));
  POSITIONS.forEach((position) => {
    valid.filter((player) => positionKey(player.position) === position)
      .sort((a, b) => b.mean_season_points - a.mean_season_points)
      .slice(0, 60)
      .forEach((player) => ids.add(player.mfl_player_id));
  });
  required.forEach((id) => ids.add(id));
  const sourceById = new Map(allPlayers.map((player) => [player.mfl_player_id, player]));
  return [...ids].map((id, index) => {
    const player = sourceById.get(id);
    if (!player) throw new Error(`Picked player ${id} is absent from the static projection board`);
    return { ...player, positionKey: positionKey(player.position), index } as Player;
  });
}

export function recomputeDraftEV({
  players: allPlayers,
  draftOrder,
  externalDraftOrder = [],
  manual,
  xouts,
  simulations = 60,
}: {
  players: BrowserPlayer[];
  draftOrder: BrowserPick[];
  externalDraftOrder?: BrowserPick[];
  manual: Record<string, string>;
  xouts: Set<string>;
  simulations?: number;
}): BrowserSimulationResult {
  const filled = new Map<number, string>();
  draftOrder.forEach((pick) => {
    const playerId = pick.player_id || manual[String(pick.overall)];
    if (playerId) filled.set(pick.overall, playerId);
  });
  const filledIds = new Set(filled.values());
  if (filledIds.size !== filled.size) throw new Error("The manual draft contains a duplicate player");
  const highestFilled = Math.max(0, ...filled.keys());
  const future = draftOrder.filter((pick) => pick.overall > highestFilled);
  const target = future.find((pick) => pick.franchise_id === USER_ID);
  if (!target) throw new Error("Levered Leopolds has no remaining pick");

  const externalFilled = new Map<number, string>();
  externalDraftOrder.forEach((pick) => {
    if (pick.player_id) externalFilled.set(pick.overall, pick.player_id);
  });
  const externalFilledIds = new Set(externalFilled.values());
  if (externalFilledIds.size !== externalFilled.size) throw new Error("The other conference contains a duplicate player");
  const externalHighestFilled = Math.max(0, ...externalFilled.keys());
  const externalFuture = externalDraftOrder.filter((pick) => pick.overall > externalHighestFilled);

  const requiredIds = new Set([...filledIds, ...externalFilledIds]);
  const players = selectPlayerPool(allPlayers, requiredIds);
  const byId = new Map(players.map((player) => [player.mfl_player_id, player]));
  const franchises = [...new Set(draftOrder.map((pick) => pick.franchise_id))];
  if (franchises.length !== 10) throw new Error("The static simulator requires ten conference franchises");
  const externalFranchises = [...new Set(externalDraftOrder.map((pick) => pick.franchise_id))];
  const hasExactExternal = externalFranchises.length === 10;
  if (externalDraftOrder.length && !hasExactExternal) throw new Error("The other conference must contain ten franchises");
  const allFranchises = [...franchises, ...externalFranchises];
  const franchiseIndex = new Map(allFranchises.map((id, index) => [id, index]));
  const rosters: Record<string, string[]> = Object.fromEntries(franchises.map((id) => [id, []]));
  filled.forEach((playerId, overall) => {
    const slot = draftOrder.find((pick) => pick.overall === overall)!;
    if (!byId.has(playerId)) throw new Error(`Picked player ${playerId} is outside the simulation pool`);
    rosters[slot.franchise_id].push(playerId);
  });
  const externalRosters: Record<string, string[]> = Object.fromEntries(externalFranchises.map((id) => [id, []]));
  externalFilled.forEach((playerId, overall) => {
    const slot = externalDraftOrder.find((pick) => pick.overall === overall)!;
    if (!byId.has(playerId)) throw new Error(`Other-conference player ${playerId} is outside the simulation pool`);
    externalRosters[slot.franchise_id].push(playerId);
  });
  const picksAfter = new Map(future.map((slot) => [
    slot.overall,
    future.filter((later) => later.overall > slot.overall && later.franchise_id === slot.franchise_id).length,
  ]));
  const externalPicksAfter = new Map(externalFuture.map((slot) => [
    slot.overall,
    externalFuture.filter((later) => later.overall > slot.overall && later.franchise_id === slot.franchise_id).length,
  ]));

  const replacement: Record<Position, number> = { QB: 0, RB: 0, WR: 0, TE: 0, PK: 0, DEF: 0 };
  POSITIONS.forEach((position) => {
    const values = players.filter((p) => p.positionKey === position).map((p) => p.mean_season_points).sort((a, b) => b - a);
    replacement[position] = values[Math.min(DEMAND[position] * 10, values.length) - 1] || 0;
  });
  const draftValue = (player: Player) => player.mean_season_points - replacement[player.positionKey];
  const initiallySelectable = players.filter((player) =>
    !filledIds.has(player.mfl_player_id) && !xouts.has(player.mfl_player_id)
    && feasible(player.positionKey, rosters[USER_ID], picksAfter.get(target.overall) || 0, byId)
  );
  const market = initiallySelectable.slice().sort((a, b) => a.display_adp - b.display_adp);
  const model = initiallySelectable.slice().sort((a, b) => draftValue(b) - draftValue(a) || b.mean_season_points - a.mean_season_points);
  const candidates: Player[] = [];
  [...market.slice(0, 4), ...model].forEach((player) => {
    if (candidates.length < 8 && !candidates.some((existing) => existing.mfl_player_id === player.mfl_player_id)) candidates.push(player);
  });

  const random = mulberry32(50_473 + highestFilled * 997);
  const scenarios: Scenario[] = [];
  while (scenarios.length < simulations) {
    const noise = new Float32Array(allFranchises.length * players.length);
    const outcomes = new Float32Array(players.length);
    for (let index = 0; index < noise.length; index += 1) noise[index] = normal(random) * 0.38;
    for (let index = 0; index < outcomes.length; index += 1) outcomes[index] = normal(random);
    const externalIndices = (scenarioIndex: number) => {
      const externalScenarios = new Uint16Array(FIELD_SIZE - franchises.length);
      const externalOpponents = new Uint8Array(FIELD_SIZE - franchises.length);
      for (let index = 0; index < externalScenarios.length; index += 1) {
        externalScenarios[index] = (scenarioIndex + Math.floor(((index + 1) * simulations) / (externalScenarios.length + 1))) % simulations;
        externalOpponents[index] = (scenarioIndex + index) % (franchises.length - 1);
      }
      return { externalScenarios, externalOpponents };
    }
    const { externalScenarios, externalOpponents } = externalIndices(scenarios.length);
    scenarios.push({ noise, outcomes, externalScenarios, externalOpponents });
    if (scenarios.length < simulations) {
      const pairedExternal = externalIndices(scenarios.length);
      scenarios.push({
        noise: Float32Array.from(noise, (value) => -value),
        outcomes: Float32Array.from(outcomes, (value) => -value),
        ...pairedExternal,
      });
    }
  }

  const exactExternalScoreRows: Record<string, number>[] = hasExactExternal
    ? scenarios.map((scenario) => {
      const scenarioRosters: Record<string, string[]> = Object.fromEntries(
        externalFranchises.map((id) => [id, [...externalRosters[id]]]),
      );
      const selectedIds = new Set(externalFilledIds);
      externalFuture.forEach((slot) => {
        const roster = scenarioRosters[slot.franchise_id];
        const remaining = externalPicksAfter.get(slot.overall) || 0;
        const availablePlayers = players.filter((player) =>
          !selectedIds.has(player.mfl_player_id)
          && feasible(player.positionKey, roster, remaining, byId)
        );
        if (!availablePlayers.length) throw new Error(`No legal other-conference player remains for pick ${slot.overall}`);
        const team = franchiseIndex.get(slot.franchise_id)!;
        const choice = availablePlayers.reduce((best, player) => {
          const playerScore = Math.log(player.display_adp) + scenario.noise[team * players.length + player.index];
          const bestScore = Math.log(best.display_adp) + scenario.noise[team * players.length + best.index];
          return playerScore < bestScore ? player : best;
        });
        roster.push(choice.mfl_player_id);
        selectedIds.add(choice.mfl_player_id);
      });
      return Object.fromEntries(
        externalFranchises.map((id) => [id, rosterScore(scenarioRosters[id], scenario, byId)]),
      );
    })
    : [];

  function equity(forcedId: string | null): Equity {
    const scoreRows: Record<string, number>[] = [];
    const selections = new Map<string, number>();
    let available = 0;
    scenarios.forEach((scenario) => {
      const scenarioRosters: Record<string, string[]> = Object.fromEntries(franchises.map((id) => [id, [...rosters[id]]]));
      const selectedIds = new Set(filledIds);
      let targetSelection = "";
      future.forEach((slot) => {
        const roster = scenarioRosters[slot.franchise_id];
        const remaining = picksAfter.get(slot.overall) || 0;
        const availablePlayers = players.filter((player) =>
          !selectedIds.has(player.mfl_player_id)
          && (slot.franchise_id !== USER_ID || !xouts.has(player.mfl_player_id))
          && feasible(player.positionKey, roster, remaining, byId)
        );
        if (!availablePlayers.length) throw new Error(`No legal player remains for pick ${slot.overall}`);
        let choice: Player;
        if (slot.overall === target.overall && forcedId && availablePlayers.some((p) => p.mfl_player_id === forcedId)) {
          choice = byId.get(forcedId)!;
        } else if (slot.franchise_id === USER_ID) {
          choice = availablePlayers.reduce((best, player) => {
            const delta = draftValue(player) - draftValue(best);
            return delta > 0 || (delta === 0 && player.display_adp < best.display_adp) ? player : best;
          });
        } else {
          const team = franchiseIndex.get(slot.franchise_id)!;
          choice = availablePlayers.reduce((best, player) => {
            const playerScore = Math.log(player.display_adp) + scenario.noise[team * players.length + player.index];
            const bestScore = Math.log(best.display_adp) + scenario.noise[team * players.length + best.index];
            return playerScore < bestScore ? player : best;
          });
        }
        roster.push(choice.mfl_player_id);
        selectedIds.add(choice.mfl_player_id);
        if (slot.overall === target.overall) targetSelection = choice.mfl_player_id;
      });
      selections.set(targetSelection, (selections.get(targetSelection) || 0) + 1);
      if (!forcedId || targetSelection === forcedId) available += 1;
      scoreRows.push(Object.fromEntries(franchises.map((id) => [id, rosterScore(scenarioRosters[id], scenario, byId)])));
    });

    const opponentIds = franchises.filter((id) => id !== USER_ID);
    let first = 0; let second = 0; let third = 0; let payout = 0;
    const payoutSamples: number[] = [];
    scenarios.forEach((scenario, scenarioIndex) => {
      const ourScore = scoreRows[scenarioIndex][USER_ID];
      const opponents = opponentIds.map((id) => scoreRows[scenarioIndex][id]);
      if (hasExactExternal) {
        externalFranchises.forEach((id) => opponents.push(exactExternalScoreRows[scenarioIndex][id]));
      } else {
        for (let index = 0; index < scenario.externalScenarios.length; index += 1) {
          opponents.push(scoreRows[scenario.externalScenarios[index]][opponentIds[scenario.externalOpponents[index]]]);
        }
      }
      const greater = opponents.filter((score) => score > ourScore).length;
      const tied = 1 + opponents.filter((score) => score === ourScore).length;
      const share = (place: number) => place > greater && place <= greater + tied ? 1 / tied : 0;
      first += share(1); second += share(2); third += share(3);
      const scenarioPayout = share(1) * 0.7 + share(2) * 0.2 + share(3) * 0.1;
      payout += scenarioPayout;
      payoutSamples.push(scenarioPayout);
    });
    const commonSelection = [...selections.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "";
    return { payout: payout / simulations, first: first / simulations, second: second / simulations, third: third / simulations, availability: available / simulations, commonSelection, payoutSamples };
  }

  const reference = equity(null);
  const priced = candidates.map((candidate) => {
    const result = equity(candidate.mfl_player_id);
    return { candidate, result };
  }).sort((a, b) => b.result.payout - a.result.payout || a.candidate.display_adp - b.candidate.display_adp);
  const best = priced[0].result;
  const recommendations = priced.map(({ candidate, result }) => {
    const bestGapStandardError = pairedStandardError(result.payoutSamples, best.payoutSamples);
    return {
      rank: 0,
      player_id: candidate.mfl_player_id,
      name: candidate.name,
      position: candidate.position,
      team: candidate.team,
      adp: candidate.display_adp,
      payout_equity: result.payout,
      marginal_ev: result.payout - reference.payout,
      availability_probability: result.availability,
      first_probability: result.first,
      second_probability: result.second,
      third_probability: result.third,
      marginal_ev_standard_error: pairedStandardError(result.payoutSamples, reference.payoutSamples),
      best_ev_gap_standard_error: bestGapStandardError,
      statistically_tied_for_best: best.payout - result.payout <= 1.96 * bestGapStandardError,
    };
  }).map((row, index) => ({ ...row, rank: index + 1 }));

  return {
    recommendations,
    referencePlayer: byId.get(reference.commonSelection)?.name || reference.commonSelection,
    referencePayoutEquity: reference.payout,
    simulations,
    targetOverall: target.overall,
  };
}
