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
100 simulations and remains the more stable decision board.

## Verification

```bash
uv run --locked pytest -q
npm test
```

Current important limitation: absolute title probabilities from the draft
simulator are not yet empirically calibrated. Treat candidate ordering and
marginal EV as the decision surface; the interface labels absolute odds as
uncalibrated until the full draft-policy simulator has a historical backtest.
