'use client';

import React from 'react';
import { TrendingUp, Award } from 'lucide-react';

export default function ComparisonPlot() {
  // Pareto frontier data points: [Recall@10, QPS] measured on SIFT1M
  // (1,000,000 x 128-D) against the ann-benchmarks ground truth with 1,000
  // queries, on an NVIDIA RTX 3050 Laptop (sm_86). See the README benchmark
  // section for the full parameter sweep.
  const flashVectorHnswData = [
    { recall: 0.7568, qps: 93686 },
    { recall: 0.8767, qps: 51229 },
    { recall: 0.9511, qps: 38338 },
    { recall: 0.9848, qps: 15719 },
    { recall: 0.9958, qps: 5629 },
  ];

  const flashVectorIvfPqData = [
    { recall: 0.352, qps: 83062 },
    { recall: 0.496, qps: 34657 },
    { recall: 0.5269, qps: 22697 },
    { recall: 0.5382, qps: 14906 },
    { recall: 0.5422, qps: 9522 },
    { recall: 0.542, qps: 5800 },
  ];

  const hnswLibCpuData = [
    { recall: 0.98, qps: 13413 },
    { recall: 0.9948, qps: 7427 },
    { recall: 0.9981, qps: 4342 },
  ];

  // SVG dimensions
  const width = 360;
  const height = 180;
  const pad = { top: 20, right: 20, bottom: 30, left: 45 };

  const minR = 0.30;
  const maxR = 1.00;
  const minQ = 0;
  const maxQ = 100000;

  const toX = (r: number) => pad.left + ((r - minR) / (maxR - minR)) * (width - pad.left - pad.right);
  const toY = (q: number) => height - pad.bottom - ((q - minQ) / (maxQ - minQ)) * (height - pad.top - pad.bottom);

  const makePath = (data: { recall: number; qps: number }[]) => {
    return data
      .map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.recall)} ${toY(d.qps)}`)
      .join(' ');
  };

  return (
    <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 text-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/30">
            <TrendingUp className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-semibold text-white tracking-wide">Pareto Frontier Benchmark</h2>
            <p className="text-xs text-slate-400">SIFT1M (128-D) QPS vs Recall@10</p>
          </div>
        </div>
        <span className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
          <Award className="w-3 h-3" /> Peak Recall@10 0.9958
        </span>
      </div>

      {/* SVG Pareto Frontier Plot */}
      <div className="relative w-full flex justify-center bg-black/40 p-2 rounded-xl border border-white/5">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible font-mono text-[9px]">
          {/* Grid lines */}
          {[0.40, 0.60, 0.80, 1.00].map((r) => (
            <line
              key={r}
              x1={toX(r)}
              y1={pad.top}
              x2={toX(r)}
              y2={height - pad.bottom}
              stroke="rgba(255,255,255,0.06)"
              strokeDasharray="3,3"
            />
          ))}
          {[25000, 50000, 75000, 100000].map((q) => (
            <line
              key={q}
              x1={pad.left}
              y1={toY(q)}
              x2={width - pad.right}
              y2={toY(q)}
              stroke="rgba(255,255,255,0.06)"
              strokeDasharray="3,3"
            />
          ))}

          {/* Axes labels */}
          <text x={toX(0.30)} y={height - 12} fill="#64748b" textAnchor="start">0.30</text>
          <text x={toX(0.65)} y={height - 12} fill="#64748b" textAnchor="middle">0.65</text>
          <text x={toX(1.00)} y={height - 12} fill="#64748b" textAnchor="end">1.00</text>

          <text x={pad.left - 6} y={toY(25000)} fill="#64748b" textAnchor="end" dominantBaseline="middle">25k</text>
          <text x={pad.left - 6} y={toY(50000)} fill="#64748b" textAnchor="end" dominantBaseline="middle">50k</text>
          <text x={pad.left - 6} y={toY(75000)} fill="#64748b" textAnchor="end" dominantBaseline="middle">75k</text>
          <text x={pad.left - 6} y={toY(100000)} fill="#64748b" textAnchor="end" dominantBaseline="middle">100k</text>

          {/* HNSWLib (CPU) Line */}
          <path d={makePath(hnswLibCpuData)} fill="none" stroke="#64748b" strokeWidth="2" />
          {hnswLibCpuData.map((d, i) => (
            <circle key={i} cx={toX(d.recall)} cy={toY(d.qps)} r="3" fill="#64748b" />
          ))}

          {/* FlashVector-GPU IVF-PQ Line */}
          <path d={makePath(flashVectorIvfPqData)} fill="none" stroke="#a855f7" strokeWidth="2" />
          {flashVectorIvfPqData.map((d, i) => (
            <circle key={i} cx={toX(d.recall)} cy={toY(d.qps)} r="3" fill="#a855f7" />
          ))}

          {/* FlashVector-GPU HNSW Line */}
          <path
            d={makePath(flashVectorHnswData)}
            fill="none"
            stroke="#00f0ff"
            strokeWidth="3"
            className="drop-shadow-[0_0_8px_#00f0ff]"
          />
          {flashVectorHnswData.map((d, i) => (
            <circle
              key={i}
              cx={toX(d.recall)}
              cy={toY(d.qps)}
              r="4"
              fill="#00f0ff"
              stroke="#08090d"
              strokeWidth="1.5"
            />
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
        <div className="flex items-center gap-1.5 text-primary">
          <span className="w-2.5 h-2.5 rounded-full bg-primary shadow-glow" />
          <span>FlashVector HNSW</span>
        </div>
        <div className="flex items-center gap-1.5 text-purple-400">
          <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
          <span>FlashVector IVF-PQ</span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
          <span>HNSWLib (CPU)</span>
        </div>
      </div>
    </div>
  );
}
