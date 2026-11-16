/**
 * Pure scrubbing logic for hop-by-hop beam replay.
 *
 * A "step" is an integer in [0, scrubLength]. Step 0 is the state before any
 * iteration/probe happened (nothing revealed). Step k (k >= 1) is the state
 * right AFTER the k-th beam iteration / IVF probe / (legacy) distance hop.
 * Everything is derived from the step index, so scrubbing backwards restores
 * earlier state exactly.
 */
import type {
  Algorithm,
  BeamCandidate,
  BeamStep,
  ProbeStep,
  TraversalHop3D,
  Vector3D,
} from './types';

export type ScrubMode = 'beam' | 'probe' | 'hop';

export interface ScrubSource {
  hops?: TraversalHop3D[] | null;
  beam_steps?: BeamStep[] | null;
  probes?: ProbeStep[] | null;
}

export interface ExpandedNode {
  id: number;
  distance: number;
  pos: Vector3D | null;
}

export function isIvfAlgorithm(algorithm: Algorithm | string | undefined | null): boolean {
  return typeof algorithm === 'string' && algorithm.startsWith('ivf_');
}

export function scrubMode(response: ScrubSource | null | undefined): ScrubMode {
  if (response?.beam_steps && response.beam_steps.length > 0) return 'beam';
  if (response?.probes && response.probes.length > 0) return 'probe';
  return 'hop';
}

/** Number of scrub steps available for a response. */
export function scrubLength(response: ScrubSource | null | undefined): number {
  if (!response) return 0;
  if (response.beam_steps && response.beam_steps.length > 0) return response.beam_steps.length;
  if (response.probes && response.probes.length > 0) return response.probes.length;
  return response.hops?.length ?? 0;
}

/** Clamp a step into [0, length]; tolerates NaN / fractional / negative input. */
export function clampStep(step: number, length: number): number {
  const max = Math.max(0, Math.floor(Number.isFinite(length) ? length : 0));
  if (!Number.isFinite(step)) return 0;
  return Math.min(max, Math.max(0, Math.floor(step)));
}

export function stepForward(step: number, length: number): number {
  return clampStep(clampStep(step, length) + 1, length);
}

export function stepBackward(step: number, length: number): number {
  return clampStep(clampStep(step, length) - 1, length);
}

/** The beam iteration shown at `step` (undefined at step 0 or out of range). */
export function beamStepAt(beamSteps: BeamStep[] | null | undefined, step: number): BeamStep | undefined {
  if (!beamSteps || beamSteps.length === 0) return undefined;
  const s = clampStep(step, beamSteps.length);
  return s === 0 ? undefined : beamSteps[s - 1];
}

function evalEnd(b: BeamStep, hopsLen: number): number {
  return Math.min(hopsLen, Math.max(0, b.eval_start + b.eval_count));
}

/**
 * Cumulative distance evaluations visible up to and including `step`.
 * With beam steps: every hop up to the end of that iteration's evaluations.
 * Legacy (no beam steps): one hop per step.
 */
export function hopsForStep(
  hops: TraversalHop3D[] | null | undefined,
  beamSteps: BeamStep[] | null | undefined,
  step: number,
): TraversalHop3D[] {
  const all = hops ?? [];
  if (all.length === 0) return [];
  if (beamSteps && beamSteps.length > 0) {
    const cur = beamStepAt(beamSteps, step);
    if (!cur) return [];
    return all.slice(0, evalEnd(cur, all.length));
  }
  return all.slice(0, clampStep(step, all.length));
}

/** Distance evaluations made by exactly the current iteration (emphasized). */
export function evalsForStep(
  hops: TraversalHop3D[] | null | undefined,
  beamSteps: BeamStep[] | null | undefined,
  step: number,
): TraversalHop3D[] {
  const all = hops ?? [];
  if (all.length === 0) return [];
  if (beamSteps && beamSteps.length > 0) {
    const cur = beamStepAt(beamSteps, step);
    if (!cur) return [];
    const start = Math.min(all.length, Math.max(0, cur.eval_start));
    return all.slice(start, evalEnd(cur, all.length));
  }
  const s = clampStep(step, all.length);
  return s === 0 ? [] : [all[s - 1]];
}

/** Candidate queue (best first) after the iteration at `step`; empty at step 0. */
export function candidatesAtStep(
  beamSteps: BeamStep[] | null | undefined,
  step: number,
): BeamCandidate[] {
  return beamStepAt(beamSteps, step)?.candidates ?? [];
}

/** The node expanded in the iteration at `step`, if any. */
export function expandedAtStep(
  beamSteps: BeamStep[] | null | undefined,
  step: number,
): ExpandedNode | null {
  const cur = beamStepAt(beamSteps, step);
  if (!cur) return null;
  return { id: cur.expanded_node, distance: cur.expanded_distance, pos: cur.expanded_pos ?? null };
}

/** Probes revealed up to and including `step`, nearest centroid first. */
export function visibleProbes(probes: ProbeStep[] | null | undefined, step: number): ProbeStep[] {
  if (!probes || probes.length === 0) return [];
  const sorted = [...probes].sort((a, b) => a.rank - b.rank);
  return sorted.slice(0, clampStep(step, sorted.length));
}

/** The probe being processed at `step` (last visible one). */
export function currentProbe(probes: ProbeStep[] | null | undefined, step: number): ProbeStep | undefined {
  const vis = visibleProbes(probes, step);
  return vis.length > 0 ? vis[vis.length - 1] : undefined;
}

/** Result ids found in the lists probed up to `step` (deduplicated). */
export function probeResultIds(probes: ProbeStep[] | null | undefined, step: number): number[] {
  const seen = new Set<number>();
  for (const p of visibleProbes(probes, step)) {
    for (const id of p.result_ids ?? []) seen.add(id);
  }
  return Array.from(seen);
}

/** Human readable label for the scrubber, e.g. "Beam iteration 3 / 12". */
export function stepLabel(response: ScrubSource | null | undefined, step: number): string {
  const len = scrubLength(response);
  const s = clampStep(step, len);
  switch (scrubMode(response)) {
    case 'beam':
      return `Beam iteration ${s} / ${len}`;
    case 'probe':
      return `Probe ${s} / ${len}`;
    default:
      return `Hop ${s} / ${len}`;
  }
}
