'use client';

import React from 'react';
import { GitBranch } from 'lucide-react';
import { SearchResponse } from '../../lib/types';
import {
  beamStepAt,
  candidatesAtStep,
  clampStep,
  currentProbe,
  scrubLength,
  scrubMode,
  stepLabel,
  visibleProbes,
} from '../../lib/beam';

interface StepPanelProps {
  response: SearchResponse | null;
  step: number;
}

export default function StepPanel({ response, step }: StepPanelProps) {
  const mode = scrubMode(response);
  const length = scrubLength(response);
  const s = clampStep(step, length);

  return (
    <div className="glass-panel p-4 rounded-2xl flex flex-col gap-3 text-sm">
      <div className="flex items-center justify-between border-b border-white/10 pb-2">
        <span className="font-mono text-xs text-slate-300 flex items-center gap-1.5">
          <GitBranch className="w-3.5 h-3.5 text-primary" />
          {mode === 'probe' ? 'IVF PROBE' : mode === 'beam' ? 'BEAM QUEUE' : 'TRAVERSAL'}
        </span>
        <span className="text-[11px] font-mono text-primary font-semibold">
          {response ? stepLabel(response, s) : '-'}
        </span>
      </div>

      {mode === 'beam' && <BeamBody response={response} step={s} />}
      {mode === 'probe' && <ProbeBody response={response} step={s} />}
      {mode === 'hop' && (
        <div className="text-xs text-slate-500 py-4 text-center font-mono">
          {length === 0
            ? 'Dispatch a query to step through the search'
            : 'No per-iteration debug data from this server'}
        </div>
      )}
    </div>
  );
}

function BeamBody({ response, step }: StepPanelProps) {
  const beamSteps = response?.beam_steps;
  const cur = beamStepAt(beamSteps, step);
  const candidates = candidatesAtStep(beamSteps, step);

  if (!cur) {
    return (
      <div className="text-xs text-slate-500 py-4 text-center font-mono">
        Step forward to expand the first node
      </div>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-slate-400">
          Expanded <span className="text-pink-400 font-semibold">ID {cur.expanded_node}</span>
        </span>
        <span className="text-slate-400">
          dist {cur.expanded_distance.toFixed(4)} | {cur.eval_count} evals
        </span>
      </div>
      <div className="max-h-64 overflow-y-auto flex flex-col gap-1 pr-1">
        {candidates.map((c, i) => (
          <div
            key={`${c.id}-${i}`}
            className={`p-2 rounded-lg bg-black/40 border flex items-center justify-between text-xs font-mono ${
              c.id === cur.expanded_node ? 'border-pink-500/60' : 'border-white/5'
            }`}
          >
            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  c.visited ? 'bg-amber-400' : 'bg-cyan-400'
                }`}
              />
              <span className="text-slate-200">ID {c.id}</span>
              <span className="text-[10px] text-slate-500">
                {c.visited ? 'visited' : 'unvisited'}
              </span>
            </div>
            <span className="text-primary font-semibold">{c.distance.toFixed(4)}</span>
          </div>
        ))}
        {candidates.length === 0 && (
          <div className="text-xs text-slate-500 py-4 text-center font-mono">Queue empty</div>
        )}
      </div>
    </>
  );
}

function ProbeBody({ response, step }: StepPanelProps) {
  const probes = response?.probes;
  const cur = currentProbe(probes, step);
  const shown = visibleProbes(probes, step);

  if (!cur) {
    return (
      <div className="text-xs text-slate-500 py-4 text-center font-mono">
        Step forward to probe the nearest centroid
      </div>
    );
  }

  return (
    <>
      <div className="p-2 rounded-lg bg-black/40 border border-pink-500/40 flex flex-col gap-1 text-xs font-mono">
        <div className="flex justify-between">
          <span className="text-pink-400 font-semibold">Cluster {cur.cluster_id}</span>
          <span className="text-slate-400">probe #{shown.length}</span>
        </div>
        <div className="flex justify-between text-slate-300">
          <span>centroid dist</span>
          <span className="text-primary">{cur.centroid_distance.toFixed(4)}</span>
        </div>
        <div className="flex justify-between text-slate-300">
          <span>list length</span>
          <span>{cur.list_len}</span>
        </div>
        <div className="flex justify-between text-slate-300">
          <span>best in list</span>
          <span>{cur.best_distance === null ? '-' : cur.best_distance.toFixed(4)}</span>
        </div>
        <div className="text-slate-400 break-words">
          top-k ids in list:{' '}
          <span className="text-emerald-400">
            {cur.result_ids.length > 0 ? cur.result_ids.join(', ') : 'none'}
          </span>
        </div>
      </div>
      <div className="max-h-40 overflow-y-auto flex flex-col gap-1 pr-1">
        {shown.map((p, i) => (
          <div
            key={p.rank}
            className="px-2 py-1 rounded-md bg-black/30 flex items-center justify-between text-[11px] font-mono text-slate-300"
          >
            <span>
              #{i + 1} cluster {p.cluster_id}
            </span>
            <span className="text-slate-500">{p.result_ids.length} hits</span>
          </div>
        ))}
      </div>
    </>
  );
}
