'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Cpu, Layers, Activity, Search, Play, Pause, SkipBack, SkipForward } from 'lucide-react';

import ControlPanel from '../components/ui/ControlPanel';
import MetricsPanel from '../components/ui/MetricsPanel';
import ComparisonPlot from '../components/ui/ComparisonPlot';
import StepPanel from '../components/ui/StepPanel';
import { useWebSocket } from '../hooks/useWebSocket';
import {
  IndexStats,
  ProjectionMethod,
  QueryParams,
  Vector3D,
  Vectors3DResponse,
} from '../lib/types';
import { formatLatency } from '../lib/math';
import { clampStep, scrubLength, stepBackward, stepForward, stepLabel } from '../lib/beam';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Dynamically import 3D Canvas to avoid SSR hydration mismatches
const EmbeddingSpace = dynamic(
  () => import('../components/canvas/EmbeddingSpace'),
  { ssr: false }
);

export default function Dashboard() {
  const [points, setPoints] = useState<Vector3D[]>([]);
  const [clusters, setClusters] = useState<number[]>([]);
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [isRebuilding, setIsRebuilding] = useState(false);

  const [projection, setProjection] = useState<ProjectionMethod>('pca');
  const [isProjecting, setIsProjecting] = useState(false);
  const isProjectingRef = useRef(false);

  // Scrub position: 0 = before the first iteration/probe, N = after the last.
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [queryParams, setQueryParams] = useState<QueryParams>({
    top_k: 10,
    ef_search: 64,
    nprobe: 8,
    use_ivf: false,
    algorithm: 'hnsw',
    debug: true,
  });

  const { isConnected, latestResponse, latencyHistory, sendQuery } = useWebSocket();

  const length = scrubLength(latestResponse);
  const currentStep = clampStep(step, length);

  // Fetch initial 3D dataset and stats
  const loadData = useCallback(async () => {
    try {
      const [vecRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/vectors/3d`),
        fetch(`${API_BASE}/api/v1/stats`),
      ]);

      if (vecRes.ok && !isProjectingRef.current) {
        const vecData: Vectors3DResponse = await vecRes.json();
        setPoints(vecData.points);
        setClusters(vecData.clusters);
        if (vecData.method) setProjection(vecData.method);
      }

      if (statsRes.ok) {
        const statsData: IndexStats = await statsRes.json();
        setStats(statsData);
      }
    } catch {
      // Backend starting or offline
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Restart the replay scrubber whenever a new traversal arrives
  useEffect(() => {
    setStep(0);
    setIsPlaying(false);
  }, [latestResponse]);

  // Step-by-step playback
  useEffect(() => {
    if (!isPlaying) return;
    if (length === 0 || currentStep >= length) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => {
      setStep((s) => stepForward(s, length));
    }, 350);
    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, length]);

  const handleStepBack = useCallback(() => {
    setIsPlaying(false);
    setStep((s) => stepBackward(s, length));
  }, [length]);

  const handleStepForward = useCallback(() => {
    setIsPlaying(false);
    setStep((s) => stepForward(s, length));
  }, [length]);

  // ArrowLeft / ArrowRight scrub, unless the user is typing/adjusting a control
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      const el = e.target as HTMLElement | null;
      if (
        el &&
        (el.tagName === 'INPUT' ||
          el.tagName === 'TEXTAREA' ||
          el.tagName === 'SELECT' ||
          el.isContentEditable)
      ) {
        return;
      }
      e.preventDefault();
      if (e.key === 'ArrowLeft') handleStepBack();
      else handleStepForward();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleStepBack, handleStepForward]);

  const handleTriggerSearch = () => {
    sendQuery({ ...queryParams, debug: true });
  };

  const handleChangeProjection = async (method: ProjectionMethod) => {
    if (isProjectingRef.current || method === projection) return;
    isProjectingRef.current = true;
    setIsProjecting(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/projection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method }),
      });
      if (res.ok) {
        isProjectingRef.current = false;
        setProjection(method);
        const vecRes = await fetch(`${API_BASE}/api/v1/vectors/3d`);
        if (vecRes.ok) {
          const vecData: Vectors3DResponse = await vecRes.json();
          setPoints(vecData.points);
          setClusters(vecData.clusters);
          setProjection(vecData.method ?? method);
        }
      }
    } catch {
      // Backend offline; keep the current projection
    } finally {
      isProjectingRef.current = false;
      setIsProjecting(false);
    }
  };

  const handleRebuildDataset = async (numVectors: number) => {
    setIsRebuilding(true);
    try {
      await fetch(`${API_BASE}/api/v1/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_vectors: numVectors, dim: 128, num_clusters: 16 }),
      });
      await loadData();
    } finally {
      setIsRebuilding(false);
    }
  };

  return (
    <main className="flex flex-col h-screen w-screen bg-[#08090d] text-slate-100 overflow-hidden select-none">
      {/* Top Navigation Bar */}
      <header className="h-16 px-6 glass-panel border-b border-white/10 flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary via-[#7000ff] to-accent p-0.5 flex items-center justify-center shadow-glow">
            <div className="w-full h-full bg-[#08090d] rounded-[10px] flex items-center justify-center text-primary font-bold font-mono">
              ⚡
            </div>
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-wider bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent flex items-center gap-2">
              FLASHVECTOR<span className="text-primary font-mono font-normal text-xs px-1.5 py-0.5 rounded bg-primary/10 border border-primary/30">GPU</span>
            </h1>
            <p className="text-[11px] text-slate-400 font-mono">Warp-Cooperative SIMT Vector Search</p>
          </div>
        </div>

        {/* Hardware Status Chips */}
        <div className="hidden md:flex items-center gap-3 text-xs font-mono">
          <div className="glass-panel px-3 py-1.5 rounded-xl border border-white/10 flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-300">NVIDIA RTX 3050 (sm_86)</span>
          </div>
          <div className="glass-panel px-3 py-1.5 rounded-xl border border-white/10 flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-primary" />
            <span className="text-slate-300">CUDA 13.5 + Rust FFI</span>
          </div>
          <div className="glass-panel px-3 py-1.5 rounded-xl border border-white/10 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-accent" />
            <span className="text-slate-300">Axum Tokio Gateway</span>
          </div>
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Control Column */}
        <div className="w-80 p-4 flex flex-col gap-4 overflow-y-auto z-10 shrink-0">
          <ControlPanel
            params={queryParams}
            onChangeParams={setQueryParams}
            onTriggerSearch={handleTriggerSearch}
            onRebuildDataset={handleRebuildDataset}
            isRebuilding={isRebuilding}
            isConnected={isConnected}
            projection={projection}
            isProjecting={isProjecting}
            onChangeProjection={handleChangeProjection}
          />
          <ComparisonPlot />
        </div>

        {/* Center 3D Canvas */}
        <div className="flex-1 h-full relative overflow-hidden bg-black">
          <EmbeddingSpace
            points={points}
            clusters={clusters}
            response={latestResponse}
            step={currentStep}
          />

          {isProjecting && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 glass-panel px-4 py-2 rounded-xl text-xs font-mono text-slate-200 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
              Computing projection...
            </div>
          )}

          {/* Traversal Replay Scrubber */}
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20">
            <div className="glass-panel-glow px-6 py-3 rounded-2xl flex items-center gap-3">
              <button
                onClick={handleStepBack}
                disabled={length === 0 || currentStep <= 0}
                aria-label="Step back"
                title="Step back (Left arrow)"
                className="flex items-center px-2 py-1.5 rounded-lg bg-white/5 text-slate-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <SkipBack className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => {
                  if (!isPlaying && currentStep >= length) {
                    setStep(0);
                  }
                  setIsPlaying((playing) => !playing);
                }}
                disabled={length === 0}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary/20 text-primary hover:bg-primary/30 text-xs font-mono font-semibold disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={handleStepForward}
                disabled={length === 0 || currentStep >= length}
                aria-label="Step forward"
                title="Step forward (Right arrow)"
                className="flex items-center px-2 py-1.5 rounded-lg bg-white/5 text-slate-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <SkipForward className="w-3.5 h-3.5" />
              </button>
              <input
                type="range"
                min={0}
                max={length}
                value={currentStep}
                disabled={length === 0}
                onChange={(e) => {
                  setIsPlaying(false);
                  setStep(clampStep(parseInt(e.target.value, 10), length));
                }}
                className="w-56 accent-primary disabled:opacity-40"
              />
              <span className="font-mono text-xs text-primary whitespace-nowrap min-w-[10rem]">
                {stepLabel(latestResponse, currentStep)}
              </span>
              <span className="font-mono text-xs text-slate-400 whitespace-nowrap">
                {formatLatency(latestResponse?.latency_us ?? 0)}
              </span>
            </div>
          </div>
        </div>

        {/* Right Telemetry Column */}
        <div className="w-84 p-4 flex flex-col gap-4 overflow-y-auto z-10 shrink-0">
          <MetricsPanel
            stats={stats}
            latestResponse={latestResponse}
            latencyHistory={latencyHistory}
          />

          <StepPanel response={latestResponse} step={currentStep} />

          {/* Nearest Neighbor Results Table */}
          <div className="glass-panel p-4 rounded-2xl flex flex-col gap-3 text-sm">
            <div className="flex items-center justify-between border-b border-white/10 pb-2">
              <span className="font-mono text-xs text-slate-300 flex items-center gap-1.5">
                <Search className="w-3.5 h-3.5 text-primary" /> TOP-K CANDIDATES
              </span>
              <span className="text-[11px] font-mono text-primary font-semibold">
                {latestResponse?.results?.length ?? 0} MATCHES
              </span>
            </div>

            <div className="max-h-60 overflow-y-auto flex flex-col gap-1 pr-1">
              {!latestResponse || latestResponse.results.length === 0 ? (
                <div className="text-xs text-slate-500 py-6 text-center font-mono">
                  Click Dispatch Query to inspect candidates
                </div>
              ) : (
                latestResponse.results.map((res, rank) => (
                  <div
                    key={rank}
                    className="p-2 rounded-lg bg-black/40 border border-white/5 flex items-center justify-between text-xs font-mono hover:border-primary/40 transition-all"
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-md bg-white/5 flex items-center justify-center text-slate-400 text-[10px]">
                        #{rank + 1}
                      </span>
                      <span className="text-slate-200">ID {res.id}</span>
                    </div>
                    <span className="text-primary font-semibold">
                      dist {res.distance.toFixed(4)}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
