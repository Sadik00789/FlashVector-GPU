'use client';

import React from 'react';
import { Play, RefreshCw, Cpu, Layers, Zap, Database } from 'lucide-react';
import { QueryParams } from '../../lib/types';

interface ControlPanelProps {
  params: QueryParams;
  onChangeParams: (params: QueryParams) => void;
  onTriggerSearch: () => void;
  onRebuildDataset: (numVectors: number) => void;
  isRebuilding: boolean;
  isConnected: boolean;
}

export default function ControlPanel({
  params,
  onChangeParams,
  onTriggerSearch,
  onRebuildDataset,
  isRebuilding,
  isConnected,
}: ControlPanelProps) {
  return (
    <div className="glass-panel p-5 rounded-2xl flex flex-col gap-5 text-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-primary/10 text-primary border border-primary/30">
            <Zap className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-semibold text-white tracking-wide">Kernel Controls</h2>
            <p className="text-xs text-slate-400">sm_86 Hardware Dispatcher</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              isConnected ? 'bg-success shadow-[0_0_8px_#00ff66]' : 'bg-rose-500 animate-ping'
            }`}
          />
          <span className="text-xs font-mono text-slate-300">
            {isConnected ? 'LIVE WS' : 'CONNECTING'}
          </span>
        </div>
      </div>

      {/* Algorithm Mode Switcher */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-mono text-slate-400 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-primary" /> ALGORITHM ENGINE
        </label>
        <div className="grid grid-cols-2 gap-2 p-1 bg-black/40 rounded-xl border border-white/5">
          <button
            onClick={() => onChangeParams({ ...params, use_ivf: false })}
            className={`py-2 px-3 rounded-lg text-xs font-medium transition-all ${
              !params.use_ivf
                ? 'bg-primary/20 text-primary border border-primary/40 shadow-glow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            HNSW Warp Beam
          </button>
          <button
            onClick={() => onChangeParams({ ...params, use_ivf: true })}
            className={`py-2 px-3 rounded-lg text-xs font-medium transition-all ${
              params.use_ivf
                ? 'bg-secondary/30 text-purple-300 border border-secondary/50'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            IVF-PQ ADC (Shared)
          </button>
        </div>
      </div>

      {/* Sliders */}
      <div className="flex flex-col gap-4">
        {/* Top-K */}
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-slate-300 font-mono">top_k Nearest Neighbors</span>
            <span className="text-primary font-mono font-semibold">{params.top_k}</span>
          </div>
          <input
            type="range"
            min={1}
            max={50}
            value={params.top_k}
            onChange={(e) => onChangeParams({ ...params, top_k: parseInt(e.target.value, 10) })}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-primary"
          />
        </div>

        {/* efSearch (for HNSW) */}
        {!params.use_ivf && (
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-mono">efSearch (Beam Width)</span>
              <span className="text-primary font-mono font-semibold">{params.ef_search}</span>
            </div>
            <input
              type="range"
              min={16}
              max={256}
              step={16}
              value={params.ef_search}
              onChange={(e) => onChangeParams({ ...params, ef_search: parseInt(e.target.value, 10) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>
        )}

        {/* nprobe (for IVF-PQ) */}
        {params.use_ivf && (
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-mono">nprobe (Centroids Scanned)</span>
              <span className="text-purple-400 font-mono font-semibold">{params.nprobe}</span>
            </div>
            <input
              type="range"
              min={1}
              max={32}
              value={params.nprobe}
              onChange={(e) => onChangeParams({ ...params, nprobe: parseInt(e.target.value, 10) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-secondary"
            />
          </div>
        )}
      </div>

      {/* Trigger Search Button */}
      <button
        onClick={onTriggerSearch}
        disabled={!isConnected}
        className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-primary to-[#00a8ff] text-black font-semibold text-xs tracking-wider uppercase flex items-center justify-center gap-2 hover:brightness-110 active:scale-[0.98] transition-all shadow-glow disabled:opacity-50"
      >
        <Play className="w-4 h-4 fill-current" />
        Dispatch Query Vector
      </button>

      {/* Dataset Scale Presets */}
      <div className="border-t border-white/10 pt-4 flex flex-col gap-2">
        <label className="text-xs font-mono text-slate-400 flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-accent" /> RE-INDEX VECTORS (VRAM)
        </label>
        <div className="grid grid-cols-3 gap-2">
          {[5000, 10000, 25000].map((num) => (
            <button
              key={num}
              onClick={() => onRebuildDataset(num)}
              disabled={isRebuilding}
              className="py-1.5 px-2 rounded-lg bg-white/5 border border-white/10 hover:border-accent text-xs font-mono text-slate-300 hover:text-white transition-all disabled:opacity-50 flex items-center justify-center gap-1"
            >
              {isRebuilding ? (
                <RefreshCw className="w-3 h-3 animate-spin" />
              ) : (
                `${num / 1000}k`
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
