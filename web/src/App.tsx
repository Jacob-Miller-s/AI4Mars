import {
  Activity,
  AlertTriangle,
  Archive,
  BarChart3,
  CircleHelp,
  Database,
  Download,
  FlaskConical,
  FolderArchive,
  Gauge,
  LockKeyhole,
  RefreshCw,
  Satellite,
  TableProperties
} from "lucide-react";
import { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { artifactUrl, getComparison, getOverview, getProvenance, getRun, getRuns, getSamples, runStreamUrl } from "./api";
import type { BatchEvent, EpochEvent, Overview, Provenance, RunCard, RunDetail, SampleRecord, SystemEvent } from "./types";

type View = "overview" | "live" | "experiments" | "evaluation" | "workbench" | "provenance" | "artifacts";

const classColors: Record<string, string> = {
  soil: "#b76845",
  bedrock: "#3c7c68",
  sand: "#c89b38",
  big_rock: "#795248"
};

const classRgb: Record<string, [number, number, number]> = {
  soil: [183, 104, 69],
  bedrock: [60, 124, 104],
  sand: [200, 155, 56],
  big_rock: [121, 82, 72]
};

type WorkbenchFilters = {
  bigRockFalseNegative: boolean;
  bigRockToSoil: boolean;
  sortBy: string;
  split: string;
};

const navItems: Array<{ id: View; label: string; icon: typeof Gauge }> = [
  { id: "overview", label: "Overview", icon: Gauge },
  { id: "live", label: "Live training", icon: Activity },
  { id: "experiments", label: "Experiments", icon: TableProperties },
  { id: "evaluation", label: "Evaluation", icon: BarChart3 },
  { id: "workbench", label: "Workbench", icon: FlaskConical },
  { id: "provenance", label: "Provenance", icon: Database },
  { id: "artifacts", label: "Artifacts", icon: FolderArchive }
];

function formatMetric(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || Number.isNaN(value) ? "--" : value.toFixed(digits);
}

function formatBytes(value: number | undefined): string {
  if (value === undefined) return "--";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index ? 1 : 0)} ${units[index]}`;
}

function shortHash(hash?: string | null): string {
  return hash ? `${hash.slice(0, 12)}...` : "--";
}

function formatTimestamp(value?: string | null): string {
  return value ? new Date(value).toLocaleString() : "--";
}

function formatSeeds(seeds?: Record<string, number>): string {
  return seeds && Object.keys(seeds).length ? Object.entries(seeds).map(([name, value]) => `${name}=${value}`).join(", ") : "--";
}

function eventEpochs(detail: RunDetail | null): EpochEvent[] {
  return detail?.metrics.filter((event): event is EpochEvent => event.event_type === "epoch") ?? [];
}

function eventBatches(detail: RunDetail | null): BatchEvent[] {
  return detail?.metrics.filter((event): event is BatchEvent => event.event_type === "batch") ?? [];
}

function latestEpoch(detail: RunDetail | null): EpochEvent | undefined {
  return eventEpochs(detail).at(-1);
}

function downloadText(filename: string, text: string, mime = "application/json"): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function App() {
  const [view, setView] = useState<View>("overview");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [runs, setRuns] = useState<RunCard[]>([]);
  const [provenance, setProvenance] = useState<Provenance | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [streamReadyRunId, setStreamReadyRunId] = useState("");
  const [workbenchFilters, setWorkbenchFilters] = useState<WorkbenchFilters>({
    bigRockFalseNegative: false,
    bigRockToSoil: false,
    sortBy: "image_iou",
    split: ""
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const streamCursorRef = useRef(0);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextOverview, nextRuns, nextProvenance] = await Promise.all([getOverview(), getRuns(), getProvenance()]);
      setOverview(nextOverview);
      setRuns(nextRuns.runs);
      setProvenance(nextProvenance);
      if (!selectedRunId) {
        setSelectedRunId(nextOverview.active_run?.run_id ?? nextRuns.runs[0]?.run_id ?? "");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to reach the local research console API.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      setDetail(null);
      setStreamReadyRunId("");
      streamCursorRef.current = 0;
      return;
    }
    let cancelled = false;
    setDetail(null);
    setStreamReadyRunId("");
    streamCursorRef.current = 0;
    void getRun(selectedRunId)
      .then((nextDetail) => {
        if (!cancelled) {
          streamCursorRef.current = nextDetail.metrics.length + nextDetail.system_metrics.length;
          setDetail(nextDetail);
          setStreamReadyRunId(selectedRunId);
        }
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "Unable to load run detail.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    const selected = runs.find((run) => run.run_id === selectedRunId);
    if (!selected || selected.status !== "running" || streamReadyRunId !== selectedRunId || !("EventSource" in window)) return;
    let cancelled = false;
    let source: EventSource | null = null;
    let reconnectTimer: number | undefined;
    let reconnectAttempt = 0;
    const connect = () => {
      if (cancelled) return;
      const nextSource = new EventSource(runStreamUrl(selectedRunId, streamCursorRef.current));
      source = nextSource;
      nextSource.addEventListener("run", (message) => {
        const payload = JSON.parse((message as MessageEvent<string>).data) as {
          next: number;
        events: Array<{ stream: "metrics" | "system"; event: EpochEvent | BatchEvent | SystemEvent }>;
        };
        streamCursorRef.current = payload.next;
        reconnectAttempt = 0;
        setDetail((current) => {
          if (!current) return current;
          const metrics = [...current.metrics];
          const systemMetrics = [...current.system_metrics];
          for (const item of payload.events) {
            if (item.stream === "metrics") metrics.push(item.event as EpochEvent | BatchEvent);
            else systemMetrics.push(item.event as SystemEvent);
          }
          return { ...current, metrics, system_metrics: systemMetrics };
        });
      });
      nextSource.onerror = () => {
        nextSource.close();
        if (cancelled) return;
        const delay = Math.min(1_000 * 2 ** reconnectAttempt, 15_000);
        reconnectAttempt += 1;
        reconnectTimer = window.setTimeout(connect, delay);
      };
    };
    connect();
    return () => {
      cancelled = true;
      source?.close();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
    };
  }, [runs, selectedRunId, streamReadyRunId]);

  const selectRun = (runId: string) => {
    startTransition(() => setSelectedRunId(runId));
  };

  return (
    <main className="console-shell">
      <header className="topbar">
        <div className="brand-block">
          <Satellite aria-hidden="true" size={28} strokeWidth={1.7} />
          <div>
            <p className="eyebrow">LOCAL RESEARCH INFRASTRUCTURE</p>
            <h1>AI4Mars Research Console</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="local-indicator">127.0.0.1</span>
          <span className="lock-indicator"><LockKeyhole size={15} aria-hidden="true" /> Expert test set locked</span>
          <button className="icon-button" type="button" onClick={() => void refresh()} title="Refresh console data" aria-label="Refresh console data">
            <RefreshCw size={18} aria-hidden="true" />
          </button>
        </div>
      </header>

      <nav className="navigation" aria-label="Research console views">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={view === item.id ? "nav-item active" : "nav-item"}
              type="button"
              onClick={() => startTransition(() => setView(item.id))}
            >
              <Icon size={16} aria-hidden="true" />
              {item.label}
            </button>
          );
        })}
      </nav>

      {error && <div className="notice error"><AlertTriangle size={17} aria-hidden="true" /> {error}</div>}
      {!error && loading && <div className="notice">Loading local run records and manifest evidence...</div>}
      {!error && !loading && (
        <>
          {view === "overview" && <OverviewView overview={overview} runs={runs} provenance={provenance} onSelectRun={selectRun} />}
          {view === "live" && <LiveTrainingView detail={detail} selectedRunId={selectedRunId} runs={runs} onSelectRun={selectRun} />}
          {view === "experiments" && <ExperimentRegistry runs={runs} onSelectRun={selectRun} />}
          {view === "evaluation" && <EvaluationView detail={detail} onOpenWorkbench={(filters) => { setWorkbenchFilters(filters); setView("workbench"); }} />}
          {view === "workbench" && <WorkbenchView runId={selectedRunId} runs={runs} onSelectRun={selectRun} initialFilters={workbenchFilters} />}
          {view === "provenance" && <ProvenanceView provenance={provenance} />}
          {view === "artifacts" && <ArtifactsView detail={detail} />}
        </>
      )}
    </main>
  );
}

function Panel({ title, children, action }: { title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return <section className="panel"><div className="panel-heading"><h2>{title}</h2>{action}</div>{children}</section>;
}

function Metric({ label, value, tone = "default" }: { label: string; value: string; tone?: "default" | "warn" | "good" }) {
  return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function Empty({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><CircleHelp size={20} aria-hidden="true" /><strong>{title}</strong><span>{detail}</span></div>;
}

function OverviewView({ overview, runs, provenance, onSelectRun }: { overview: Overview | null; runs: RunCard[]; provenance: Provenance | null; onSelectRun: (runId: string) => void }) {
  const current = overview?.active_run?.latest_epoch ?? overview?.best_protocol_valid_validation_run?.latest_epoch;
  const bigRock = current?.per_class?.big_rock;
  const health = overview?.system_health;
  return <div className="view-grid">
    {overview?.ranking_warning && <div className="notice warning ranking-warning"><AlertTriangle size={17} aria-hidden="true" />{overview.ranking_warning}</div>}
    <section className="lead-band">
      <div>
        <p className="eyebrow">ITERATIVE RESEARCH STATUS</p>
        <h2>{overview?.active_run ? overview.active_run.experiment_name : "No active training run"}</h2>
        <p>{overview?.active_run ? `Streaming durable events from ${overview.active_run.run_id}` : "Start a protocol-valid training run or the clearly labeled synthetic smoke run to populate this console."}</p>
      </div>
      <div className="lock-panel"><LockKeyhole size={26} aria-hidden="true" /><strong>EXPERT TEST SET LOCKED</strong><span>Iterative ranking accepts crowdsourced validation only.</span></div>
    </section>
    <div className="metric-grid">
      <Metric label="Validation mIoU" value={formatMetric(current?.mean_iou)} tone="good" />
      <Metric label="Big rock IoU" value={formatMetric(bigRock?.iou)} tone={bigRock?.iou != null && bigRock.iou < 0.3 ? "warn" : "default"} />
      <Metric label="Pixel accuracy" value={formatMetric(current?.pixel_accuracy)} />
      <Metric label="Protocol gates" value={provenance?.gates.every((gate) => gate.passed) ? "PASS" : "CHECK"} tone={provenance?.gates.every((gate) => gate.passed) ? "good" : "warn"} />
    </div>
    <Panel title="Best protocol-valid validation run">
      {overview?.best_protocol_valid_validation_run ? <RunSummaryCard run={overview.best_protocol_valid_validation_run} onSelect={onSelectRun} /> : <Empty title="No eligible benchmark" detail="Invalid, legacy, and sealed-test records are excluded from the default best-run ranking." />}
    </Panel>
    <Panel title="Current validation by terrain class">
      {current?.per_class ? <table className="metric-table"><thead><tr><th>Class</th><th>IoU</th><th>Dice/F1</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{Object.entries(current.per_class).map(([name, metrics]) => <tr key={name}><td><span className="class-dot" style={{ background: classColors[name] ?? "#718d9b" }} />{name}</td><td>{formatMetric(metrics.iou)}</td><td>{formatMetric(metrics.dice_f1)}</td><td>{formatMetric(metrics.precision)}</td><td>{formatMetric(metrics.recall)}</td></tr>)}</tbody></table> : <Empty title="No per-class validation yet" detail="Epoch-level metrics will populate once evaluation has completed." />}
    </Panel>
    <Panel title="System health">
      <div className="health-grid">
        <Metric label="CPU" value={health ? `${formatMetric(health.cpu_percent, 1)}%` : "Offline"} />
        <Metric label="RAM" value={health ? `${formatMetric(health.ram_percent, 1)}%` : "Offline"} />
        <Metric label="GPU" value={health?.gpu_available ? `${formatMetric(health.gpu_utilization_percent, 0)}%` : "Unavailable"} />
        <Metric label="VRAM" value={health?.gpu_available ? formatBytes(health.gpu_memory_used_bytes) : "No NVML"} />
        <Metric label="Allocated" value={formatBytes(health?.gpu_memory_allocated_bytes)} />
        <Metric label="Reserved" value={formatBytes(health?.gpu_memory_reserved_bytes)} />
      </div>
    </Panel>
    <Panel title="Recent records">
      {runs.length ? <RunTable runs={runs.slice(0, 8)} onSelect={onSelectRun} compact /> : <Empty title="No run records" detail="Run records are discovered from outputs/runs as they are created." />}
    </Panel>
    <Panel title="Failed and invalid records">
      {overview?.failed_runs.length ? <RunTable runs={overview.failed_runs} onSelect={onSelectRun} compact /> : <Empty title="No failed or invalid records" detail="Terminal failures and protocol-invalid records remain visible here for investigation." />}
    </Panel>
    <Panel title="Protocol watch">
      <div className="gate-list">
        {(provenance?.gates ?? []).map((gate) => <div className={gate.passed ? "gate pass" : "gate fail"} key={gate.name}><span>{gate.passed ? "PASS" : "FAIL"}</span><div><strong>{gate.name.replaceAll("_", " ")}</strong><p>{gate.detail}</p></div></div>)}
      </div>
    </Panel>
  </div>;
}

function RunSummaryCard({ run, onSelect }: { run: RunCard; onSelect: (runId: string) => void }) {
  return <button className="run-summary" type="button" onClick={() => onSelect(run.run_id)}><span className="run-status completed">{run.status}</span><strong>{run.experiment_name}</strong><span>{run.model} {run.encoder ? `/${run.encoder}` : ""}</span><span>mIoU {formatMetric(run.summary?.best_validation_mean_iou ?? run.latest_epoch?.mean_iou)}</span><span>{shortHash(run.dataset_manifest_sha256)}</span></button>;
}

function LiveTrainingView({ detail, selectedRunId, runs, onSelectRun }: { detail: RunDetail | null; selectedRunId: string; runs: RunCard[]; onSelectRun: (runId: string) => void }) {
  const epochs = eventEpochs(detail);
  const batches = eventBatches(detail);
  const systems = detail?.system_metrics ?? [];
  const terrainLines = Object.entries(classColors).map(([name, color]) => ({ key: name, color, name }));
  const perClassTrend = (metric: "iou" | "dice_f1" | "precision" | "recall") => epochs.map((event) => ({
    epoch: event.epoch,
    ...Object.fromEntries(Object.entries(event.per_class ?? {}).map(([name, values]) => [name, values[metric]]))
  })) as Array<Record<string, unknown>>;
  return <div className="view-grid live-grid">
    <RunSelector runs={runs} selectedRunId={selectedRunId} onSelect={onSelectRun} />
    {!detail ? <Empty title="Select a run" detail="Completed and interrupted runs remain inspectable after refresh." /> : <>
      <div className="metric-grid">
        <Metric label="Run state" value={detail.metadata.status.toUpperCase()} tone={detail.metadata.status === "running" ? "good" : "default"} />
        <Metric label="Last epoch" value={String(latestEpoch(detail)?.epoch ?? "--")} />
        <Metric label="Throughput" value={`${formatMetric(batches.at(-1)?.throughput_samples_per_second, 1)} samples/s`} />
        <Metric label="ETA" value={batches.at(-1)?.eta_seconds ? `${Math.ceil(batches.at(-1)!.eta_seconds! / 60)} min` : "--"} />
      </div>
      <LinePanel title="Batch loss" data={batches} x="batch" lines={[{ key: "loss", color: "#c4512f", name: "raw batch loss" }, { key: "smoothed_loss", color: "#5c8c7a", name: "smoothed loss" }]} />
      <LinePanel title="Epoch loss: train and validation" data={epochs} x="epoch" lines={[{ key: "train_loss", color: "#795248", name: "train loss" }, { key: "val_loss", color: "#c4512f", name: "validation loss" }]} />
      <LinePanel title="Validation quality" data={epochs} x="epoch" lines={[{ key: "mean_iou", color: "#3c7c68", name: "mean IoU" }, { key: "pixel_accuracy", color: "#b8872d", name: "pixel accuracy" }]} />
      <LinePanel title="Per-class IoU" data={perClassTrend("iou")} x="epoch" lines={terrainLines} />
      <LinePanel title="Per-class Dice/F1" data={perClassTrend("dice_f1")} x="epoch" lines={terrainLines} />
      <LinePanel title="Per-class precision" data={perClassTrend("precision")} x="epoch" lines={terrainLines} />
      <LinePanel title="Per-class recall" data={perClassTrend("recall")} x="epoch" lines={terrainLines} />
      <LinePanel title="Learning rate and epoch duration" data={epochs} x="epoch" lines={[{ key: "learning_rate", color: "#795248", name: "learning rate" }, { key: "epoch_duration_seconds", color: "#718d9b", name: "epoch seconds" }]} />
      <LinePanel title="Host and GPU telemetry" data={systems} x="timestamp" lines={[{ key: "cpu_percent", color: "#b76845", name: "CPU %" }, { key: "ram_percent", color: "#3c7c68", name: "RAM %" }, { key: "gpu_utilization_percent", color: "#c89b38", name: "GPU %" }]} />
      <div className="metric-grid"><Metric label="Allocator memory" value={formatBytes(systems.at(-1)?.gpu_memory_allocated_bytes)} /><Metric label="Allocator reserved" value={formatBytes(systems.at(-1)?.gpu_memory_reserved_bytes)} /><Metric label="NVML VRAM used" value={formatBytes(systems.at(-1)?.gpu_memory_used_bytes)} /><Metric label="NVML VRAM total" value={formatBytes(systems.at(-1)?.gpu_memory_total_bytes)} /></div>
    </>}
  </div>;
}

function RunSelector({ runs, selectedRunId, onSelect }: { runs: RunCard[]; selectedRunId: string; onSelect: (runId: string) => void }) {
  return <label className="run-selector">Run <select value={selectedRunId} onChange={(event) => onSelect(event.target.value)}><option value="">Select a run</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} - {run.experiment_name}</option>)}</select></label>;
}

function LinePanel({ title, data, x, lines }: { title: string; data: Array<Record<string, unknown>>; x: string; lines: Array<{ key: string; color: string; name: string }> }) {
  return <Panel title={title}>{data.length ? <div className="chart"><ResponsiveContainer width="100%" height={260}><LineChart data={data}><CartesianGrid strokeDasharray="2 4" stroke="#d5ddd5" /><XAxis dataKey={x} stroke="#54625a" /><YAxis stroke="#54625a" width={48} /><Tooltip formatter={(value: number) => formatMetric(value, 5)} /><Legend />{lines.map((line) => <Line type="linear" dataKey={line.key} name={line.name} stroke={line.color} dot={false} strokeWidth={2} connectNulls key={line.key} />)}</LineChart></ResponsiveContainer></div> : <Empty title="No events yet" detail="Durable JSONL events will appear here without a page reload while a run is active." />}</Panel>;
}

function ExperimentRegistry({ runs, onSelectRun }: { runs: RunCard[]; onSelectRun: (runId: string) => void }) {
  const [search, setSearch] = useState("");
  const [validOnly, setValidOnly] = useState(false);
  const [sortBy, setSortBy] = useState<"date" | "miou" | "big_rock">("date");
  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<{ warnings: string[]; config_diff: Record<string, unknown[]>; runs: RunDetail[] } | null>(null);
  const deferredSearch = useDeferredValue(search);
  const filtered = useMemo(() => {
    const matches = runs.filter((run) => (!validOnly || run.protocol_valid) && `${run.run_id} ${run.experiment_name} ${run.hypothesis ?? ""} ${run.researcher_notes ?? ""} ${run.model ?? ""} ${run.tags.join(" ")}`.toLowerCase().includes(deferredSearch.toLowerCase()));
    return [...matches].sort((left, right) => {
      if (sortBy === "miou") return (right.summary?.best_validation_mean_iou ?? right.latest_epoch?.mean_iou ?? -1) - (left.summary?.best_validation_mean_iou ?? left.latest_epoch?.mean_iou ?? -1);
      if (sortBy === "big_rock") return (right.latest_epoch?.per_class?.big_rock?.iou ?? -1) - (left.latest_epoch?.per_class?.big_rock?.iou ?? -1);
      return String(right.started_at ?? "").localeCompare(String(left.started_at ?? ""));
    });
  }, [runs, validOnly, deferredSearch, sortBy]);
  const toggle = (runId: string) => setSelected((current) => current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId].slice(-4));
  const compare = async () => {
    if (selected.length < 2) return;
    const result = await getComparison(selected);
    setComparison({ warnings: result.warnings, config_diff: result.config_diff, runs: result.runs });
  };
  const exportCsv = () => downloadText("ai4mars-run-registry.csv", ["run_id,status,protocol_valid,split_role,hypothesis,tags,notes,model,seeds,started_at,best_epoch,miou,manifest", ...filtered.map((run) => [run.run_id, run.status, run.protocol_valid, run.split_role, run.hypothesis ?? "", run.tags.join("|"), run.researcher_notes ?? "", `${run.model ?? ""}${run.encoder ? `/${run.encoder}` : ""}`, formatSeeds(run.random_seeds), run.started_at ?? "", run.summary?.best_epoch ?? "", run.summary?.best_validation_mean_iou ?? run.latest_epoch?.mean_iou ?? "", run.dataset_manifest_sha256 ?? ""].map((item) => JSON.stringify(item)).join(","))].join("\n"), "text/csv");
  return <div className="view-grid registry-grid">
    <Panel title="Experiment registry" action={<div className="panel-actions"><button className="icon-button" type="button" title="Export registry as CSV" aria-label="Export registry as CSV" onClick={exportCsv}><Download size={17} aria-hidden="true" /></button><button className="icon-button" type="button" title="Export registry as JSON" aria-label="Export registry as JSON" onClick={() => downloadText("ai4mars-run-registry.json", JSON.stringify(filtered, null, 2))}><Archive size={17} aria-hidden="true" /></button></div>}>
      <div className="table-controls"><input aria-label="Search experiments" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter by run, tag, or hypothesis" /><label><input type="checkbox" checked={validOnly} onChange={(event) => setValidOnly(event.target.checked)} /> Protocol valid only</label><label>Sort <select aria-label="Sort experiments" value={sortBy} onChange={(event) => setSortBy(event.target.value as typeof sortBy)}><option value="date">Date</option><option value="miou">Validation mIoU</option><option value="big_rock">Big rock IoU</option></select></label><button type="button" className="command-button" disabled={selected.length < 2} onClick={() => void compare()}>Compare selected</button></div>
      <RunTable runs={filtered} onSelect={onSelectRun} selected={selected} onToggle={toggle} detailed />
    </Panel>
    {comparison && <><Panel title="Selected run mIoU overlay"><ComparisonOverlay details={comparison.runs} /></Panel><Panel title="Configuration diff"><div className="warning-list">{comparison.warnings.map((warning) => <div className="notice warning" key={warning}><AlertTriangle size={16} aria-hidden="true" />{warning}</div>)}</div><table className="diff-table"><thead><tr><th>Field</th><th>Selected values</th></tr></thead><tbody>{Object.entries(comparison.config_diff).map(([key, values]) => <tr key={key}><td>{key}</td><td>{values.map((value) => JSON.stringify(value)).join(" | ")}</td></tr>)}</tbody></table></Panel></>}
  </div>;
}

function ComparisonOverlay({ details }: { details: RunDetail[] }) {
  const colors = ["#c4512f", "#3c7c68", "#c89b38", "#795248"];
  const epochNumbers = Array.from(new Set(details.flatMap((detail) => eventEpochs(detail).map((event) => event.epoch)))).sort((left, right) => left - right);
  const data = epochNumbers.map((epoch) => ({
    epoch,
    ...Object.fromEntries(details.map((detail) => [detail.metadata.run_id, eventEpochs(detail).find((event) => event.epoch === epoch)?.mean_iou]))
  })) as Array<Record<string, unknown>>;
  if (!data.length) return <Empty title="No epoch metrics" detail="Selected records do not contain durable epoch events to overlay." />;
  return <div className="chart"><ResponsiveContainer width="100%" height={280}><LineChart data={data}><CartesianGrid strokeDasharray="2 4" stroke="#d5ddd5" /><XAxis dataKey="epoch" stroke="#54625a" /><YAxis stroke="#54625a" width={48} /><Tooltip formatter={(value: number) => formatMetric(value, 5)} /><Legend />{details.map((detail, index) => <Line key={detail.metadata.run_id} type="linear" dataKey={detail.metadata.run_id} name={detail.metadata.run_id} stroke={colors[index % colors.length]} dot={false} strokeWidth={2} connectNulls />)}</LineChart></ResponsiveContainer></div>;
}

function RunTable({ runs, onSelect, compact = false, detailed = false, selected = [], onToggle }: { runs: RunCard[]; onSelect: (runId: string) => void; compact?: boolean; detailed?: boolean; selected?: string[]; onToggle?: (runId: string) => void }) {
  return <div className="table-wrap"><table className="run-table"><thead><tr>{onToggle && <th aria-label="Compare selection" />}<th>Run</th><th>Status</th><th>Validity</th><th>Split</th><th>mIoU</th><th>Big rock</th>{detailed && <><th>Hypothesis / notes</th><th>Tags</th><th>Model</th><th>Seeds</th><th>Started</th><th>Best epoch</th></>}<th>Manifest</th></tr></thead><tbody>{runs.map((run) => <tr key={run.run_id} onClick={() => onSelect(run.run_id)}>{onToggle && <td onClick={(event) => event.stopPropagation()}><input aria-label={`Select ${run.run_id} for comparison`} type="checkbox" checked={selected.includes(run.run_id)} onChange={() => onToggle(run.run_id)} /></td>}<td><strong>{run.experiment_name}</strong><span className="subtle">{run.run_id}</span></td><td><span className={`run-status ${run.status}`}>{run.status}</span></td><td>{run.protocol_valid ? <span className="validity valid">valid</span> : <span className="validity invalid">{run.legacy ? "legacy" : "invalid"}</span>}</td><td>{run.split_role.replaceAll("_", " ")}</td><td>{formatMetric(run.summary?.best_validation_mean_iou ?? run.latest_epoch?.mean_iou)}</td><td>{formatMetric(run.latest_epoch?.per_class?.big_rock?.iou)}</td>{detailed && <><td><strong>{run.hypothesis ?? "--"}</strong>{run.researcher_notes && <span className="subtle">{run.researcher_notes}</span>}</td><td>{run.tags.join(", ") || "--"}</td><td>{run.model ?? "--"}{run.encoder ? ` / ${run.encoder}` : ""}</td><td>{formatSeeds(run.random_seeds)}</td><td>{formatTimestamp(run.started_at)}</td><td>{run.summary?.best_epoch ?? "--"}</td></>}<td title={run.dataset_manifest_sha256}>{shortHash(run.dataset_manifest_sha256)}</td></tr>)}</tbody></table>{!compact && !runs.length && <Empty title="No matching run records" detail="Adjust the registry filters or create a local run record." />}</div>;
}

function EvaluationView({ detail, onOpenWorkbench }: { detail: RunDetail | null; onOpenWorkbench: (filters: WorkbenchFilters) => void }) {
  const [selectedCell, setSelectedCell] = useState<{ row: number; column: number } | null>(null);
  const epoch = latestEpoch(detail);
  const matrix = epoch?.confusion_matrix ?? [];
  const names = Object.keys(epoch?.per_class ?? classColors);
  const normalized = matrix.map((row) => { const total = row.reduce((sum, value) => sum + value, 0); return row.map((value) => total ? value / total : 0); });
  const exportCsv = () => {
    if (!detail || !epoch) return;
    const header = ["class", "support", "iou", "dice_f1", "precision", "recall"];
    const rows = names.map((name) => {
      const metrics = epoch.per_class?.[name];
      return [name, metrics?.support ?? "", metrics?.iou ?? "", metrics?.dice_f1 ?? "", metrics?.precision ?? "", metrics?.recall ?? ""];
    });
    downloadText(`${detail.metadata.run_id}-evaluation.csv`, [header, ...rows].map((row) => row.map((value) => JSON.stringify(value)).join(",")).join("\n"), "text/csv");
  };
  return <div className="view-grid">
    {!detail || !epoch ? <Empty title="No evaluation metrics" detail="Select a completed or active run with epoch-level evaluation events." /> : <>
      <Panel title="Per-class metrics"><table className="metric-table"><thead><tr><th>Class</th><th>Support</th><th>IoU</th><th>Dice/F1</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{names.map((name) => { const metrics = epoch.per_class?.[name]; return <tr key={name}><td><span className="class-dot" style={{ background: classColors[name] ?? "#718d9b" }} />{name}</td><td>{metrics?.support ?? "--"}</td><td>{formatMetric(metrics?.iou)}</td><td>{formatMetric(metrics?.dice_f1)}</td><td>{formatMetric(metrics?.precision)}</td><td>{formatMetric(metrics?.recall)}</td></tr>; })}</tbody></table></Panel>
      <Panel title="Confusion matrix: rows = ground truth, columns = prediction" action={<div className="panel-actions"><button className="icon-button" type="button" title="Export evaluation metrics as CSV" aria-label="Export evaluation metrics as CSV" onClick={exportCsv}><TableProperties size={17} aria-hidden="true" /></button><button className="icon-button" type="button" title="Export evaluation metrics as JSON" aria-label="Export evaluation metrics as JSON" onClick={() => downloadText(`${detail.metadata.run_id}-evaluation.json`, JSON.stringify(epoch, null, 2))}><Download size={17} aria-hidden="true" /></button></div>}>
        {matrix.length ? <div className="matrix-layout"><div><h3>Raw counts</h3><Matrix matrix={matrix} names={names} selected={selectedCell} onSelect={setSelectedCell} /></div><div><h3>Row-normalized</h3><Matrix matrix={normalized} names={names} selected={selectedCell} onSelect={setSelectedCell} normalized /></div></div> : <Empty title="No confusion counts" detail="Epoch events without raw counts cannot populate the matrix." />}
        {selectedCell && <div className="matrix-selection">Selected: ground truth <strong>{names[selectedCell.row]}</strong> predicted as <strong>{names[selectedCell.column]}</strong><button className="command-button" type="button" onClick={() => onOpenWorkbench({ bigRockFalseNegative: names[selectedCell.row] === "big_rock" && names[selectedCell.column] !== "big_rock", bigRockToSoil: names[selectedCell.row] === "big_rock" && names[selectedCell.column] === "soil", sortBy: "image_iou", split: "" })}>Find representative samples</button></div>}
      </Panel>
    </>}
  </div>;
}

function Matrix({ matrix, names, selected, onSelect, normalized = false }: { matrix: number[][]; names: string[]; selected: { row: number; column: number } | null; onSelect: (cell: { row: number; column: number }) => void; normalized?: boolean }) {
  return <table className="confusion-matrix"><thead><tr><th>GT / Pred</th>{names.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{matrix.map((row, rowIndex) => <tr key={names[rowIndex] ?? rowIndex}><th>{names[rowIndex]}</th>{row.map((value, columnIndex) => <td key={columnIndex}><button type="button" className={selected?.row === rowIndex && selected.column === columnIndex ? "matrix-cell selected" : "matrix-cell"} title={`Ground truth ${names[rowIndex]}, predicted ${names[columnIndex]}: ${normalized ? formatMetric(value, 4) : value}`} onClick={() => onSelect({ row: rowIndex, column: columnIndex })}>{normalized ? formatMetric(value, 3) : value}</button></td>)}</tr>)}</tbody></table>;
}

function WorkbenchView({ runId, runs, onSelectRun, initialFilters }: { runId: string; runs: RunCard[]; onSelectRun: (runId: string) => void; initialFilters: WorkbenchFilters }) {
  const pageSize = 4;
  const [filters, setFilters] = useState<WorkbenchFilters>(initialFilters);
  const [samples, setSamples] = useState<SampleRecord[]>([]);
  const [available, setAvailable] = useState(false);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [availableSplits, setAvailableSplits] = useState<string[]>([]);
  const [comparisonRunId, setComparisonRunId] = useState("");
  const [comparisonSamples, setComparisonSamples] = useState<SampleRecord[]>([]);
  const [comparisonAvailable, setComparisonAvailable] = useState(false);
  const [visibleClasses, setVisibleClasses] = useState<Record<string, boolean>>({ soil: true, bedrock: true, sand: true, big_rock: true });
  const [error, setError] = useState<string | null>(null);
  const updateFilters = (next: Partial<WorkbenchFilters>) => { setOffset(0); setFilters((current) => ({ ...current, ...next })); };
  useEffect(() => { setFilters(initialFilters); setOffset(0); }, [initialFilters]);
  useEffect(() => { setOffset(0); setComparisonRunId(""); }, [runId]);
  useEffect(() => {
    if (!runId) { setSamples([]); setAvailable(false); setTotal(0); setAvailableSplits([]); return; }
    let cancelled = false;
    void getSamples(runId, filters, offset, pageSize).then((result) => {
      if (cancelled) return;
      setSamples(result.samples);
      setAvailable(result.available);
      setTotal(result.total);
      setAvailableSplits(result.available_splits ?? []);
      setError(null);
    }).catch((caught) => { if (!cancelled) setError(caught instanceof Error ? caught.message : "Unable to load prediction index."); });
    return () => { cancelled = true; };
  }, [runId, filters, offset]);
  useEffect(() => {
    if (!comparisonRunId) { setComparisonSamples([]); setComparisonAvailable(false); return; }
    let cancelled = false;
    void getSamples(comparisonRunId, filters, offset, pageSize).then((result) => {
      if (cancelled) return;
      setComparisonSamples(result.samples);
      setComparisonAvailable(result.available);
    }).catch((caught) => { if (!cancelled) setError(caught instanceof Error ? caught.message : "Unable to load comparison prediction index."); });
    return () => { cancelled = true; };
  }, [comparisonRunId, filters, offset]);
  const sampleGrid = (sampleRows: SampleRecord[], targetRunId: string, targetAvailable: boolean) => !targetAvailable ? <Empty title="No indexed prediction assets" detail="This run does not declare an accessible prediction index." /> : sampleRows.length ? <div className="sample-grid">{sampleRows.map((sample) => <SamplePanel key={sample.sample_id} sample={sample} runId={targetRunId} visibleClasses={visibleClasses} />)}</div> : <Empty title="No matching samples" detail="No indexed samples match the selected split and failure filters." />;
  return <div className="view-grid">
    <Panel title="Prediction and failure-analysis workbench"><RunSelector runs={runs} selectedRunId={runId} onSelect={onSelectRun} /><div className="table-controls"><label><input type="checkbox" checked={filters.bigRockFalseNegative} onChange={(event) => updateFilters({ bigRockFalseNegative: event.target.checked })} /> Big rock false negatives</label><label><input type="checkbox" checked={filters.bigRockToSoil} onChange={(event) => updateFilters({ bigRockToSoil: event.target.checked })} /> Big rock to soil</label><label>Split <select aria-label="Filter samples by split" value={filters.split} onChange={(event) => updateFilters({ split: event.target.value })}><option value="">All splits</option>{availableSplits.map((split) => <option key={split} value={split}>{split.replaceAll("_", " ")}</option>)}</select></label><label>Sort <select value={filters.sortBy} onChange={(event) => updateFilters({ sortBy: event.target.value })}><option value="image_iou">Worst image IoU</option><option value="loss">Highest loss</option><option value="uncertainty">Highest uncertainty</option></select></label><label>Compare <select aria-label="Compare prediction run" value={comparisonRunId} onChange={(event) => setComparisonRunId(event.target.value)}><option value="">No comparison</option>{runs.filter((run) => run.run_id !== runId).map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}</select></label></div><div className="class-toggle-row">{Object.keys(classColors).map((className) => <label key={className}><input type="checkbox" checked={visibleClasses[className]} onChange={(event) => setVisibleClasses({ ...visibleClasses, [className]: event.target.checked })} /><span className="class-dot" style={{ background: classColors[className] }} />{className}</label>)}</div></Panel>
    {error && <div className="notice error">{error}</div>}
    {!runId ? <Empty title="Select a run" detail="Prediction assets are indexed per run and loaded lazily." /> : !available ? <Empty title="No indexed prediction assets" detail="Run evaluation can write artifacts/prediction_index.jsonl to enable this workbench." /> : comparisonRunId ? <div className="workbench-compare"><section><h2>{runId}</h2>{sampleGrid(samples, runId, available)}</section><section><h2>{comparisonRunId}</h2>{sampleGrid(comparisonSamples, comparisonRunId, comparisonAvailable)}</section></div> : sampleGrid(samples, runId, available)}
    {runId && available && <div className="pagination-controls"><button type="button" className="command-button" aria-label="Previous samples" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>Previous</button><span>{total ? `${offset + 1}-${Math.min(offset + samples.length, total)} of ${total}` : "No samples"}</span><button type="button" className="command-button" aria-label="Next samples" disabled={offset + pageSize >= total} onClick={() => setOffset(offset + pageSize)}>Next</button></div>}
  </div>;
}

function SamplePanel({ sample, runId, visibleClasses }: { sample: SampleRecord; runId: string; visibleClasses: Record<string, boolean> }) {
  return <section className="sample-panel"><div className="sample-heading"><strong>{sample.sample_id}</strong><span>{sample.synthetic_demo ? "SYNTHETIC DEMO" : sample.split}</span></div><div className="sample-metrics"><span>IoU {formatMetric(sample.image_iou)}</span><span>Loss {formatMetric(sample.loss)}</span><span>Uncertainty {formatMetric(sample.uncertainty)}</span><span>{sample.assets.entropy ? "Entropy map available" : "Entropy map unavailable"}</span></div><div className="sample-images">{["image", "ground_truth", "prediction", "overlay", "error_heatmap"].map((kind) => sample.assets[kind] && <figure key={kind}>{kind === "ground_truth" || kind === "prediction" ? <PaletteMask src={artifactUrl(runId, sample.assets[kind])} alt={`${kind.replaceAll("_", " ")} for ${sample.sample_id}`} visibleClasses={visibleClasses} /> : <img loading="lazy" src={artifactUrl(runId, sample.assets[kind])} alt={`${kind.replaceAll("_", " ")} for ${sample.sample_id}`} />}<figcaption>{kind.replaceAll("_", " ")}</figcaption></figure>)}</div></section>;
}

function PaletteMask({ src, alt, visibleClasses }: { src: string; alt: string; visibleClasses: Record<string, boolean> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (cancelled || !canvasRef.current) return;
      const canvas = canvasRef.current;
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      for (let index = 0; index < pixels.data.length; index += 4) {
        const className = Object.entries(classRgb).find(([, rgb]) => pixels.data[index] === rgb[0] && pixels.data[index + 1] === rgb[1] && pixels.data[index + 2] === rgb[2])?.[0];
        if (className && !visibleClasses[className]) pixels.data[index + 3] = 0;
      }
      context.putImageData(pixels, 0, 0);
    };
    image.src = src;
    return () => { cancelled = true; };
  }, [src, visibleClasses]);
  return <canvas ref={canvasRef} role="img" aria-label={alt} />;
}

function ProvenanceView({ provenance }: { provenance: Provenance | null }) {
  if (!provenance) return <Empty title="Provenance unavailable" detail="The backend could not read current manifest evidence." />;
  return <div className="view-grid">
    <section className="lead-band compact"><div><p className="eyebrow">DATASET AND PROVENANCE</p><h2>{provenance.dataset_name ?? "Manifest unavailable"} {provenance.dataset_version ?? ""}</h2><p>{provenance.source_record ?? "No source record available"}</p></div><div className={provenance.gates.every((gate) => gate.passed) ? "gate-badge pass" : "gate-badge fail"}>{provenance.gates.every((gate) => gate.passed) ? "GATES PASS" : "GATES REQUIRE REVIEW"}</div></section>
    <Panel title="Manifest evidence"><div className="metadata-grid"><span>Manifest</span><code>{provenance.manifest_path}</code><span>SHA-256</span><code title={provenance.manifest_sha256}>{shortHash(provenance.manifest_sha256)}</code><span>Grouping</span><code>{provenance.grouping_keys?.join(", ")}</code></div><div className="health-grid">{Object.entries(provenance.pair_counts ?? {}).map(([key, value]) => <Metric key={key} label={key.replaceAll("_", " ")} value={String(value)} />)}</div></Panel>
    <Panel title="Label roles and schemes"><div className="health-grid">{Object.entries(provenance.label_roles ?? {}).map(([key, value]) => <Metric key={`role-${key}`} label={key.replaceAll("_", " ")} value={String(value)} />)}{Object.entries(provenance.label_schemes ?? {}).map(([key, value]) => <Metric key={`scheme-${key}`} label={`${key} labels`} value={String(value)} />)}</div></Panel>
    <Panel title="Pixel-class distribution"><div className="health-grid">{Object.entries(provenance.class_pixel_counts ?? {}).map(([key, value]) => <Metric key={key} label={key.replaceAll("_", " ")} value={String(value)} />)}</div></Panel>
    <Panel title="Excluded or unmatched records">{Object.keys(provenance.unmatched_or_excluded ?? {}).length ? <div className="health-grid">{Object.entries(provenance.unmatched_or_excluded ?? {}).map(([key, value]) => <Metric key={key} label={key.replaceAll("_", " ")} value={String(value)} tone="warn" />)}</div> : <Empty title="No exclusions recorded" detail="The inspected manifest did not report excluded or unmatched rows." />}</Panel>
    <Panel title="Isolation gates"><div className="gate-list">{provenance.gates.map((gate) => <div className={gate.passed ? "gate pass" : "gate fail"} key={gate.name}><span>{gate.passed ? "PASS" : "FAIL"}</span><div><strong>{gate.name.replaceAll("_", " ")}</strong><p>{gate.detail}</p></div></div>)}</div></Panel>
    <Panel title="Split inventory"><table className="metric-table"><thead><tr><th>Split</th><th>Rows</th><th>Source groups</th><th>Sequence groups</th><th>Hash</th></tr></thead><tbody>{Object.entries(provenance.splits ?? {}).map(([name, split]) => <tr key={name}><td>{name}</td><td>{split.rows}</td><td>{split.source_groups}</td><td>{split.sequence_groups}</td><td title={split.sha256}>{shortHash(split.sha256)}</td></tr>)}</tbody></table></Panel>
  </div>;
}

function ArtifactsView({ detail }: { detail: RunDetail | null }) {
  const artifacts = detail?.metadata.artifact_refs ?? [];
  const environment = detail?.metadata.environment;
  const groupedArtifacts = artifacts.reduce<Record<string, typeof artifacts>>((groups, artifact) => {
    groups[artifact.kind] = [...(groups[artifact.kind] ?? []), artifact];
    return groups;
  }, {});
  return <div className="view-grid">
    <Panel title="Research artifacts">{!detail ? <Empty title="Select a run" detail="Artifact links are contained within the selected run directory." /> : !artifacts.length ? <Empty title="No declared artifacts" detail="Run writers add portable artifact references as checkpoints and figures become available." /> : <div className="artifact-groups">{Object.entries(groupedArtifacts).map(([kind, group]) => <section key={kind}><h3>{kind.replaceAll("_", " ")}</h3><div className="artifact-list">{group.map((artifact) => <a key={artifact.path} href={artifactUrl(detail.metadata.run_id, artifact.path)}><Archive size={18} aria-hidden="true" /><span><strong>{artifact.description ?? artifact.kind}</strong><small>{artifact.path}</small></span></a>)}</div></section>)}</div>}</Panel>
    {detail && <Panel title="Reproducibility record"><div className="metadata-grid"><span>Git commit</span><code>{detail.metadata.provenance.git_commit ?? "not recorded"}</code><span>Branch</span><code>{detail.metadata.provenance.git_branch ?? "not recorded"}</code><span>Manifest</span><code>{shortHash(detail.metadata.provenance.dataset_manifest_sha256)}</code><span>Split role</span><code>{detail.metadata.provenance.split_role}</code><span>Python</span><code>{environment?.python ?? "not recorded"}</code><span>PyTorch</span><code>{environment?.pytorch ?? "not recorded"}</code><span>CUDA / cuDNN</span><code>{[environment?.cuda, environment?.cudnn].filter(Boolean).join(" / ") || "not recorded"}</code><span>GPU</span><code>{environment?.gpu ?? "not recorded"}</code><span>CPU</span><code>{environment?.cpu ?? "not recorded"}</code><span>System memory</span><code>{formatBytes(environment?.memory_total_bytes)}</code></div></Panel>}
  </div>;
}