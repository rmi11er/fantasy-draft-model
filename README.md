# Levered Leopolds draft model

Read-only draft-day recommendations for MFL league `50473`. The model applies
the league's exact scoring, builds point-in-time player distributions, completes
the remaining 10-team conference draft, and ranks the next selection by
expected share of a 70/20/10 prize pool.

## Draft-day command

```bash
uv run --locked fantasy-draft
```

The normal command refreshes MFL draft results, rosters, players, and ADP. It
does not submit picks. Results are written to `outputs/draft_day/` and the
manual board contract is refreshed at `public/draft-board.json`.

The payout simulation conditions both ten-team conference drafts independently
and uses the same NFL outcome for duplicate cross-conference players. Opponent
future picks follow noisy ADP subject to roster feasibility; Levered Leopolds'
future picks follow projection value over replacement. Personal X-outs affect
only Levered Leopolds—other teams may still select those players.

Useful controls:

```bash
# Compare named players explicitly.
uv run --locked fantasy-draft --candidate "Jahmyr Gibbs" --candidate "Drake Maye"

# Persistently X out or restore a player.
uv run --locked fantasy-draft --exclude "Player Name"
uv run --locked fantasy-draft --include "Player Name"

# Never contact MFL; use the last-good cache and the manual ledger.
uv run --locked fantasy-draft --offline

# Record and undo conference picks in offline mode.
uv run --locked fantasy-draft --offline --record-pick "1=Jahmyr Gibbs"
uv run --locked fantasy-draft --offline --undo-pick 1

# Consume a ledger exported by the manual web board.
uv run --locked fantasy-draft --offline --manual-state /path/to/manual_state.json
```

Names must match exactly; MFL player IDs are accepted when names are ambiguous.
The command fails closed when live state is inconsistent. When MFL is down it
uses the complete last-good cache and prints a prominent stale-data warning.

## Manual fallback board

```bash
npm run dev
```

The browser board records the conference in authoritative pick order, removes
selected or X'd-out players immediately, supports undo, and exports the same
`manual_state.json` consumed by the offline model. Browser edits are local to
that device. Its **Recompute EV** button runs a self-contained 50-simulation
version of the draft model from the recorded state. The terminal command uses
100 antithetic simulations and remains the more stable decision board. Results
include paired Monte Carlo error; **top tier** means simulation noise cannot
separate that choice from the observed leader at the 95% level.

**Prize EV** is the expected share of the 70/20/10 prize pool. **Availability**
is the chance a player survives the simulated picks before Levered Leopolds'
next turn. Once Levered Leopolds is on the clock, every undrafted candidate is
literally available, so the board says **available now** instead of displaying
an unhelpful row of identical 100% values.

## Distribution calibration

Veteran means and season uncertainty are calibrated from local, point-in-time
2020–2025 residuals. Means use three prior seasons, a three-game production
prior, a half-season availability prior, and a position-specific age curve.
The age curve improved expanding-year holdout MAE in every tested season from
2022 through 2025. Veterans use position-specific conformal uncertainty scales;
rookies use production bands estimated from the walk-forward rookie backtest.
This prices upside and downside through the payout simulation without adding a
subjective variance or rookie bonus to projected points.

Prior-team rushing efficiency and player opportunity share were tested locally
as offensive-line and timeshare proxies. They did not improve the RB
walk-forward error, so they are not included. Contract status is not available
in the frozen local data and receives no hand-authored adjustment.

```bash
uv run --locked fantasy-calibrate
```

The audit writes coverage and error summaries to `outputs/backtests/`.

## Verification

```bash
uv run --locked pytest -q
npm test
```

Current important limitation: absolute title probabilities from the draft
simulator are not yet empirically calibrated. Treat candidate ordering and
marginal EV—together with the reported top tier—as the decision surface; the
interface labels absolute odds as uncalibrated until the full draft-policy
simulator has a historical draft-and-playoff backtest.
