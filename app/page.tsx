"use client";

import { useEffect, useMemo, useState } from "react";
import {
  recomputeDraftEV,
  type BrowserPick,
  type BrowserPlayer,
  type BrowserRecommendation,
} from "./draft-simulator";

type Player = BrowserPlayer;
type Pick = BrowserPick;
type Recommendation = BrowserRecommendation;
type Board = {
  generated_at: string;
  report: { recommendations: Recommendation[] };
  draft_order: Pick[];
  players: Player[];
  xouts: string[];
};

const STORAGE_KEY = "levered-leopolds-manual-draft-v1";
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function Home() {
  const [board, setBoard] = useState<Board | null>(null);
  const [manual, setManual] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("");
  const [xouts, setXouts] = useState<Set<string>>(new Set());
  const [modelRows, setModelRows] = useState<Recommendation[]>([]);
  const [dirty, setDirty] = useState(false);
  const [computing, setComputing] = useState(false);
  const [modelDetail, setModelDetail] = useState("100 saved simulations");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    fetch("/draft-board.json")
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Draft data unavailable")))
      .then((payload: Board) => {
        setBoard(payload);
        setXouts(new Set(payload.xouts));
        setModelRows(payload.report.recommendations);
        try {
          const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
          setManual(saved);
          setDirty(Object.keys(saved).length > 0);
        } catch { setManual({}); }
      })
      .catch((error: Error) => setNotice(error.message));
  }, []);

  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(manual)); }, [manual]);

  const playerById = useMemo(() => new Map(board?.players.map((p) => [p.mfl_player_id, p]) || []), [board]);
  const taken = useMemo(() => {
    const ids = new Set<string>();
    board?.draft_order.forEach((pick) => pick.player_id && ids.add(pick.player_id));
    Object.values(manual).forEach((id) => ids.add(id));
    return ids;
  }, [board, manual]);
  const currentPick = useMemo(() => {
    if (!board) return null;
    const done = board.draft_order.filter((pick) => pick.player_id || manual[String(pick.overall)]).map((pick) => pick.overall);
    return board.draft_order.find((pick) => pick.overall === Math.max(0, ...done) + 1) || null;
  }, [board, manual]);
  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (board?.players || []).filter((p) =>
      !taken.has(p.mfl_player_id) && !xouts.has(p.mfl_player_id) &&
      (!needle || p.name.toLowerCase().includes(needle) || p.team.toLowerCase() === needle || p.position.toLowerCase() === needle)
    ).slice(0, 12);
  }, [board, query, taken, xouts]);
  const recommendations = useMemo(() => modelRows.filter((p) => !taken.has(p.player_id) && !xouts.has(p.player_id)), [modelRows, taken, xouts]);
  const latest = useMemo(() => (board?.draft_order || []).filter((pick) => pick.player_id || manual[String(pick.overall)]).map((pick) => ({ ...pick, resolved: pick.player_id || manual[String(pick.overall)] })).slice(-8).reverse(), [board, manual]);

  function recordPick() {
    if (!currentPick || !selected || taken.has(selected)) return;
    setManual((old) => ({ ...old, [String(currentPick.overall)]: selected }));
    setSelected(""); setQuery("");
    setDirty(true);
    setNotice("Pick recorded. Press Recompute EV to price the new state.");
  }
  function undo() {
    const latestKey = Math.max(...Object.keys(manual).map(Number));
    if (!Number.isFinite(latestKey)) return;
    setManual((old) => { const copy = { ...old }; delete copy[String(latestKey)]; return copy; });
    setDirty(true);
    setNotice("Last manual pick removed. Recompute EV for the restored state.");
  }
  function exportState() {
    const blob = new Blob([JSON.stringify({ league_id: "50473", conference_id: "00", picks: manual }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const link = document.createElement("a");
    link.href = url; link.download = "manual_state.json"; link.click(); URL.revokeObjectURL(url);
    setNotice("Manual ledger downloaded for the offline model.");
  }
  async function recompute() {
    if (!board || computing) return;
    setComputing(true);
    setNotice("Running 50 browser simulations from the current draft state…");
    await new Promise((resolve) => setTimeout(resolve, 30));
    try {
      const result = recomputeDraftEV({
        players: board.players,
        draftOrder: board.draft_order,
        manual,
        xouts,
        simulations: 50,
      });
      setModelRows(result.recommendations);
      setModelDetail(`${result.simulations} browser simulations · target pick ${result.targetOverall}`);
      setDirty(false);
      setNotice(`EV recomputed for pick ${result.targetOverall}. Reference policy: ${result.referencePlayer}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The browser simulation failed");
    } finally {
      setComputing(false);
    }
  }

  if (!board) return <main className="loading"><p className="eyebrow">Levered Leopolds / Draft room</p><h1>{notice || "Loading emergency board…"}</h1></main>;

  return (
    <main className="shell">
      <header className="topbar">
        <div><p className="eyebrow">Old Mission Madness · 2026</p><h1>Levered Leopolds</h1></div>
        <div className="mode"><i /> Manual fallback</div>
      </header>

      <section className="clock">
        <div><p className="label">Now recording</p><b className="pickno">{currentPick ? `${currentPick.round}.${String(currentPick.slot).padStart(2, "0")}` : "Done"}</b></div>
        <div className="owner"><p className="label">On the clock</p><strong>{currentPick?.franchise_id === "0016" ? "Levered Leopolds" : currentPick ? `Franchise ${currentPick.franchise_id}` : "Draft complete"}</strong><span>{currentPick ? `Overall ${currentPick.overall}` : "160 picks recorded"}</span></div>
        <div className="actions"><button onClick={undo} disabled={!Object.keys(manual).length}>Undo</button><button onClick={exportState}>Export state</button></div>
      </section>
      {notice && <div className="notice">{notice}</div>}

      <div className="grid">
        <section className="panel recorder">
          <Heading label="Manual entry" title="Who was selected?" detail="Exact order preserved" />
          <label className="search"><span>⌕</span><input value={query} onChange={(e) => { setQuery(e.target.value); setSelected(""); }} placeholder="Search player, team, or position" /></label>
          <div className="results">
            {results.map((player) => <button key={player.mfl_player_id} className={`player ${selected === player.mfl_player_id ? "selected" : ""}`} onClick={() => setSelected(player.mfl_player_id)}>
              <span className={`pos ${player.position.toLowerCase()}`}>{player.position}</span>
              <Identity name={player.name} detail={`${player.team} · ADP ${player.display_adp.toFixed(1)}`} />
              <small>{player.mean_season_points.toFixed(0)} pts</small>
            </button>)}
          </div>
          <button className="record" disabled={!selected || !currentPick} onClick={recordPick}>Record pick {currentPick?.overall || "—"}</button>
        </section>

        <section className="panel recs">
          <Heading label="Model board" title="Best next policies" detail={dirty ? "Recompute required" : modelDetail} warning={dirty} />
          <p className="modelnote">EV is share of the 70/20/10 pool. Absolute odds remain uncalibrated. Browser results use 50 simulations for speed; the terminal model uses 100 and remains the more stable board.</p>
          <div className={`recompute ${dirty ? "needs-run" : ""}`}>
            <span>{dirty ? "Draft state changed — old probabilities are hidden." : "Probabilities match the recorded draft state."}</span>
            <button onClick={recompute} disabled={computing || !currentPick}>{computing ? "Simulating…" : "Recompute EV"}</button>
          </div>
          {recommendations.map((row, index) => <article className="recommendation" key={row.player_id}>
            <span className="rank">{index + 1}</span><Identity name={row.name} detail={`${row.position} · ${row.team} · ADP ${row.adp.toFixed(1)}`} />
            <div className="ev"><strong>{dirty ? "—" : pct(row.payout_equity)}</strong><small>{dirty ? "stale" : `${row.marginal_ev >= 0 ? "+" : ""}${(row.marginal_ev * 100).toFixed(1)} pp`}</small></div>
            <div className="avail"><i style={{ width: dirty ? "0%" : pct(row.availability_probability) }} /><small>{dirty ? "recompute" : `${pct(row.availability_probability)} avail.`}</small></div>
            <button className="x" onClick={() => { setXouts((old) => new Set(old).add(row.player_id)); setDirty(true); setNotice(`${row.name} X’d out. Recompute EV for the reduced pool.`); }} aria-label={`X out ${row.name}`}>×</button>
          </article>)}
        </section>

        <aside className="panel activity">
          <Heading label="Conference 00" title="Latest picks" detail={`${taken.size}/160`} />
          <div className="latest">{latest.length ? latest.map((pick) => {
            const player = playerById.get(pick.resolved);
            return <div className="latestrow" key={pick.overall}><span>{pick.overall}</span><Identity name={player?.name || pick.resolved} detail={pick.franchise_id === "0016" ? "Levered Leopolds" : `Franchise ${pick.franchise_id}`} /></div>;
          }) : <p className="empty">No picks yet. Record 1.01 to begin.</p>}</div>
          <footer><span>Board generated</span><strong>{new Date(board.generated_at).toLocaleString()}</strong></footer>
        </aside>
      </div>
    </main>
  );
}

function Heading({ label, title, detail, warning = false }: { label: string; title: string; detail: string; warning?: boolean }) {
  return <div className="heading"><div><p className="label">{label}</p><h2>{title}</h2></div><span className={warning ? "warning" : "detail"}>{detail}</span></div>;
}
function Identity({ name, detail }: { name: string; detail: string }) {
  return <span className="identity"><strong>{name}</strong><small>{detail}</small></span>;
}
