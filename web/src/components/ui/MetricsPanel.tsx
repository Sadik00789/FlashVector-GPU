'use client';

import React from 'react';
import { Activity, Gauge, HardDrive, Zap, Compass } from 'lucide-react';
import { IndexStats, SearchResponse } from '../../lib/types';
import { formatLatency, formatNumber } from '../../lib/math';

interface MetricsPanelProps {
  stats: IndexStats | null;
  latestResponse: SearchResponse | null;
  latencyHistory: number[];
}

export default function MetricsPanel({
  stats,
  latestResponse,
  latencyHistory,
}: MetricsPanelProps) {
  const currentLatency = latestResponse?.latency_us ?? 0;
  const p50 = stats?.stats.p50_us ?? latestResponse?.stats.p50_us ?? 0;
  const p99 = stats?.stats.p99_us ?? latestResponse?.stats.p99_us ?? 0;
  const qps = stats?.stats.qps ?? latestResponse?.stats.qps ?? 0;
  const numVectors = stats?.num_vectors ?? 0;
  const hopsCount = latestResponse?.hops?.length ?? 0;

  const freeVram = stats?.free_vram_mb ?? 3800;
  const totalVram = stats?.total_vram_mb ?? 4096;
  const usedVram = Math.max(0, totalVram - freeVram);
  const vramPercent = Math.min(100, Math.round((usedVram / totalVram) * 100));

  return (
    <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 text-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-accent/10 text-accent border border-accent/30">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-semibold text-white tracking-wide">Telemetry & Latency</h2>
            <p className="text-xs text-slate-400">Microsecond Profiler</p>
          </div>
        </div>
        <div className="text-right">
          <span className="text-xs font-mono text-emerald-400 font-semibold">
            {formatNumber(numVectors)} VECTORS
          </span>
        </div>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-2 gap-3">
        {/* Latency Last Query */}
        <div className="p-3 rounded-xl bg-black/40 border border-white/5 flex flex-col gap-1">
          <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
            <Gauge className="w-3 h-3 text-primary" /> DISPATCH LATENCY
          </span>
          <span className="text-xl font-bold font-mono text-primary">
            {formatLatency(currentLatency)}
          </span>
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>p50: {formatLatency(p50)}</span>
            <span>p99: {formatLatency(p99)}</span>
          </div>
        </div>

        {/* QPS Throughput */}
        <div className="p-3 rounded-xl bg-black/40 border border-white/5 flex flex-col gap-1">
          <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
            <Zap className="w-3 h-3 text-accent" /> THROUGHPUT
          </span>
          <span className="text-xl font-bold font-mono text-accent">
            {qps > 0 ? `${formatNumber(Math.round(qps))} QPS` : 'IDLE'}
          </span>
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>Graph Hops: {hopsCount}</span>
            <span>Dim: {stats?.dim ?? 128}</span>
          </div>
        </div>
      </div>

      {/* Latency Sparkline */}
      <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-black/40 border border-white/5">
        <div className="flex justify-between items-center text-[11px] font-mono text-slate-400">
          <span>REAL-TIME LATENCY TIMELINE (µs)</span>
          <span className="text-xs text-primary">{latencyHistory.length} SAMPLES</span>
        </div>
        <div className="h-16 w-full flex items-end gap-1 pt-2">
          {latencyHistory.length === 0 ? (
            <div className="w-full h-full flex items-center justify-center text-xs text-slate-500 font-mono">
              Awaiting query execution...
            </div>
          ) : (
            latencyHistory.map((lat, idx) => {
              const maxL = Math.max(10, ...latencyHistory);
              const heightPct = Math.min(100, Math.max(10, (lat / maxL) * 100));
              return (
                <div
                  key={idx}
                  style={{ height: `${heightPct}%` }}
                  className="flex-1 bg-gradient-to-t from-primary/30 to-primary rounded-t-sm transition-all duration-300"
                  title={`${lat.toFixed(1)} µs`}
                />
              );
            })
          )}
        </div>
      </div>

      {/* GPU Memory Meter */}
      <div className="p-3 rounded-xl bg-black/40 border border-white/5 flex flex-col gap-2">
        <div className="flex justify-between items-center text-[11px] font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <HardDrive className="w-3 h-3 text-emerald-400" /> RTX 3050 VRAM ALLOCATION
          </span>
          <span className="text-slate-200">
            {usedVram.toFixed(0)} MB / {totalVram.toFixed(0)} MB ({vramPercent}%)
          </span>
        </div>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            style={{ width: `${vramPercent}%` }}
            className="h-full bg-gradient-to-r from-emerald-500 via-primary to-accent rounded-full transition-all duration-500"
          />
        </div>
      </div>
    </div>
  );
}
