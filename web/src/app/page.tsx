'use client';

import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Cpu, Terminal, Layers, Sparkles, Activity, Search } from 'lucide-react';

import ControlPanel from '../components/ui/ControlPanel';
import MetricsPanel from '../components/ui/MetricsPanel';
import ComparisonPlot from '../components/ui/ComparisonPlot';
import { useWebSocket } from '../hooks/useWebSocket';
import { IndexStats, QueryParams, Vector3D, Vectors3DResponse } from '../lib/types';
import { formatLatency } from '../lib/math';

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

  const [queryParams, setQueryParams] = useState<QueryParams>({
    top_k: 10,
    ef_search: 64,
    nprobe: 8,
    use_ivf: false,
  });

  const { isConnected, latestResponse, latencyHistory, sendQuery } = useWebSocket();

  // Fetch initial 3D dataset and stats
  const loadData = useCallback(async () => {
    try {
      const [vecRes, statsRes] = await Promise.all([
        fetch('http://localhost:8080/api/v1/vectors/3d'),
        fetch('http://localhost:8080/api/v1/stats'),
      ]);

      if (vecRes.ok) {
        const vecData: Vectors3DResponse = await vecRes.json();
        setPoints(vecData.points);
        setClusters(vecData.clusters);
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

  const handleTriggerSearch = () => {
    sendQuery(queryParams);
  };

  const handleRebuildDataset = async (numVectors: number) => {
    setIsRebuilding(true);
    try {
      await fetch('http://localhost:8080/api/v1/index', {
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
            <span className="text-slate-300">CUDA 12.6 + Rust FFI</span>
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
          />
          <ComparisonPlot />
        </div>

        {/* Center 3D Canvas */}
        <div className="flex-1 h-full relative overflow-hidden bg-black">
          <EmbeddingSpace
            points={points}
            clusters={clusters}
            hops={latestResponse?.hops ?? []}
            results={latestResponse?.results ?? []}
          />
        </div>

        {/* Right Telemetry Column */}
        <div className="w-84 p-4 flex flex-col gap-4 overflow-y-auto z-10 shrink-0">
          <MetricsPanel
            stats={stats}
            latestResponse={latestResponse}
            latencyHistory={latencyHistory}
          />

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
