"use client";

import { useEffect, useMemo, useState } from "react";

type Player = { mfl_player_id: string; name: string; position: string; team: string; display_adp: number; mean_season_points: number };
type Pick = { overall: number; round: number; slot: number; franchise_id: string; player_id: string | null };
type Recommendation = { rank: number; player_id: string; name: string; position: string; team: string; adp: number; payout_equity: number; marginal_ev: number; availability_probability: number };
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
  const [notice, setNotice] = useState("");

  useEffect(() => {
    fetch("/draft-board.json")
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("Draft data unavailable")))
      .then((payload: Board) => {
        setBoard(payload);
        setXouts(new Set(payload.xouts));
        try { setManual(JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")); } catch { setManual({}); }
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
  const recommendations = useMemo(() => (board?.report.recommendations || []).filter((p) => !taken.has(p.player_id) && !xouts.has(p.player_id)), [board, taken, xouts]);
  const latest = useMemo(() => (board?.draft_order || []).filter((pick) => pick.player_id || manual[String(pick.overall)]).map((pick) => ({ ...pick, resolved: pick.player_id || manual[String(pick.overall)] })).slice(-8).reverse(), [board, manual]);

  function recordPick() {
    if (!currentPick || !selected || taken.has(selected)) return;
    setManual((old) => ({ ...old, [String(currentPick.overall)]: selected }));
    setSelected(""); setQuery("");
    setNotice("Pick recorded. Availability is updated; export and run the offline model to refresh payout EV.");
  }
  function undo() {
    const latestKey = Math.max(...Object.keys(manual).map(Number));
    if (!Number.isFinite(latestKey)) return;
    setManual((old) => { const copy = { ...old }; delete copy[String(latestKey)]; return copy; });
    setNotice("Last manual pick removed.");
  }
  function exportState() {
    const blob = new Blob([JSON.stringify({ league_id: "50473", conference_id: "00", picks: manual }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const link = document.createElement("a");
    link.href = url; link.download = "manual_state.json"; link.click(); URL.revokeObjectURL(url);
    setNotice("Manual ledger downloaded for the offline model.");
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
          <Heading label="Model board" title="Best next policies" detail="Uncalibrated absolute odds" warning />
          <p className="modelnote">EV is share of the 70/20/10 pool. Manual picks update availability immediately; rerun offline for conditional EV.</p>
          {recommendations.map((row, index) => <article className="recommendation" key={row.player_id}>
            <span className="rank">{index + 1}</span><Identity name={row.name} detail={`${row.position} · ${row.team} · ADP ${row.adp.toFixed(1)}`} />
            <div className="ev"><strong>{pct(row.payout_equity)}</strong><small>{row.marginal_ev >= 0 ? "+" : ""}{(row.marginal_ev * 100).toFixed(1)} pp</small></div>
            <div className="avail"><i style={{ width: pct(row.availability_probability) }} /><small>{pct(row.availability_probability)} avail.</small></div>
            <button className="x" onClick={() => setXouts((old) => new Set(old).add(row.player_id))} aria-label={`X out ${row.name}`}>×</button>
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
