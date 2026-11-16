import { describe, expect, it } from 'vitest';
import {
  beamStepAt,
  candidatesAtStep,
  clampStep,
  currentProbe,
  evalsForStep,
  expandedAtStep,
  hopsForStep,
  isIvfAlgorithm,
  probeResultIds,
  scrubLength,
  scrubMode,
  stepBackward,
  stepForward,
  stepLabel,
  visibleProbes,
} from './beam';
import type { BeamStep, ProbeStep, TraversalHop3D } from './types';

const p = (n: number) => ({ x: n, y: n, z: n });

function hop(i: number, from: number, to: number): TraversalHop3D {
  return { step: i, from_node: from, to_node: to, distance: i * 0.1, from_pos: p(from), to_pos: p(to) };
}

// 5 hops: iteration 1 evaluates hops[0..2], iteration 2 evaluates hops[2..5]
const hops = [hop(0, 0, 1), hop(1, 0, 2), hop(2, 1, 3), hop(3, 1, 4), hop(4, 1, 5)];
const beamSteps: BeamStep[] = [
  {
    iteration: 0,
    expanded_node: 0,
    expanded_distance: 0.5,
    eval_start: 0,
    eval_count: 2,
    expanded_pos: p(0),
    candidates: [
      { id: 1, distance: 0.1, visited: false, pos: p(1) },
      { id: 2, distance: 0.2, visited: false, pos: null },
    ],
  },
  {
    iteration: 1,
    expanded_node: 1,
    expanded_distance: 0.1,
    eval_start: 2,
    eval_count: 3,
    expanded_pos: null,
    candidates: [
      { id: 1, distance: 0.1, visited: true, pos: p(1) },
      { id: 3, distance: 0.15, visited: false, pos: p(3) },
      { id: 2, distance: 0.2, visited: false, pos: null },
    ],
  },
];

const probes: ProbeStep[] = [
  { rank: 1, cluster_id: 7, centroid_distance: 2, list_len: 10, best_distance: null, result_ids: [], centroid_pos: p(7) },
  { rank: 0, cluster_id: 3, centroid_distance: 1, list_len: 12, best_distance: 0.3, result_ids: [4, 5], centroid_pos: p(3) },
  { rank: 2, cluster_id: 9, centroid_distance: 3, list_len: 8, best_distance: 0.4, result_ids: [5, 6], centroid_pos: null },
];

describe('clampStep / stepping', () => {
  it('clamps into [0, length]', () => {
    expect(clampStep(-3, 5)).toBe(0);
    expect(clampStep(99, 5)).toBe(5);
    expect(clampStep(2.7, 5)).toBe(2);
    expect(clampStep(Number.NaN, 5)).toBe(0);
    expect(clampStep(3, 0)).toBe(0);
  });

  it('steps forward and stops at the end', () => {
    expect(stepForward(0, 3)).toBe(1);
    expect(stepForward(2, 3)).toBe(3);
    expect(stepForward(3, 3)).toBe(3);
  });

  it('steps backward and stops at zero', () => {
    expect(stepBackward(3, 3)).toBe(2);
    expect(stepBackward(1, 3)).toBe(0);
    expect(stepBackward(0, 3)).toBe(0);
  });

  it('forward then backward round-trips', () => {
    for (let s = 0; s < 3; s++) expect(stepBackward(stepForward(s, 3), 3)).toBe(s);
  });

  it('empty length keeps step at zero', () => {
    expect(stepForward(0, 0)).toBe(0);
    expect(stepBackward(0, 0)).toBe(0);
  });
});

describe('scrubLength / scrubMode', () => {
  it('prefers beam_steps, then probes, then hops', () => {
    expect(scrubLength({ hops, beam_steps: beamSteps })).toBe(2);
    expect(scrubLength({ hops: [], probes })).toBe(3);
    expect(scrubLength({ hops })).toBe(5);
    expect(scrubLength({ hops, beam_steps: [], probes: [] })).toBe(5);
    expect(scrubMode({ hops, beam_steps: beamSteps })).toBe('beam');
    expect(scrubMode({ hops: [], probes })).toBe('probe');
    expect(scrubMode({ hops })).toBe('hop');
  });

  it('handles null / empty responses', () => {
    expect(scrubLength(null)).toBe(0);
    expect(scrubLength(undefined)).toBe(0);
    expect(scrubLength({ hops: [] })).toBe(0);
    expect(scrubLength({})).toBe(0);
  });

  it('labels the step by mode', () => {
    expect(stepLabel({ hops, beam_steps: beamSteps }, 1)).toBe('Beam iteration 1 / 2');
    expect(stepLabel({ probes }, 2)).toBe('Probe 2 / 3');
    expect(stepLabel({ hops }, 9)).toBe('Hop 5 / 5');
  });

  it('detects ivf algorithms', () => {
    expect(isIvfAlgorithm('ivf_pq')).toBe(true);
    expect(isIvfAlgorithm('ivf_rabitq')).toBe(true);
    expect(isIvfAlgorithm('hnsw')).toBe(false);
    expect(isIvfAlgorithm(undefined)).toBe(false);
  });
});

describe('hop grouping', () => {
  it('shows nothing at step 0', () => {
    expect(hopsForStep(hops, beamSteps, 0)).toEqual([]);
    expect(evalsForStep(hops, beamSteps, 0)).toEqual([]);
  });

  it('accumulates hops per iteration', () => {
    expect(hopsForStep(hops, beamSteps, 1)).toHaveLength(2);
    expect(hopsForStep(hops, beamSteps, 2)).toHaveLength(5);
    expect(hopsForStep(hops, beamSteps, 50)).toHaveLength(5);
  });

  it('returns only the current iteration evals', () => {
    expect(evalsForStep(hops, beamSteps, 1).map((h) => h.to_node)).toEqual([1, 2]);
    expect(evalsForStep(hops, beamSteps, 2).map((h) => h.to_node)).toEqual([3, 4, 5]);
  });

  it('scrubbing backwards restores the earlier state exactly', () => {
    const first = hopsForStep(hops, beamSteps, 1);
    hopsForStep(hops, beamSteps, 2);
    expect(hopsForStep(hops, beamSteps, 1)).toEqual(first);
  });

  it('is safe with out-of-range eval indices', () => {
    const bad: BeamStep[] = [{ ...beamSteps[0], eval_start: 4, eval_count: 100 }];
    expect(hopsForStep(hops, bad, 1)).toHaveLength(5);
    expect(evalsForStep(hops, bad, 1)).toHaveLength(1);
  });

  it('falls back to one hop per step without beam steps', () => {
    expect(hopsForStep(hops, undefined, 3)).toHaveLength(3);
    expect(hopsForStep(hops, [], 0)).toEqual([]);
    expect(evalsForStep(hops, undefined, 3)).toEqual([hops[2]]);
    expect(evalsForStep(hops, undefined, 0)).toEqual([]);
  });

  it('handles empty hops', () => {
    expect(hopsForStep([], beamSteps, 2)).toEqual([]);
    expect(hopsForStep(undefined, undefined, 2)).toEqual([]);
    expect(evalsForStep(null, beamSteps, 1)).toEqual([]);
  });
});

describe('candidates / expanded', () => {
  it('returns the queue after the iteration', () => {
    expect(candidatesAtStep(beamSteps, 0)).toEqual([]);
    expect(candidatesAtStep(beamSteps, 1)).toHaveLength(2);
    expect(candidatesAtStep(beamSteps, 2).map((c) => c.id)).toEqual([1, 3, 2]);
    expect(candidatesAtStep(beamSteps, 2)[0].visited).toBe(true);
    expect(candidatesAtStep(undefined, 2)).toEqual([]);
  });

  it('returns the expanded node', () => {
    expect(expandedAtStep(beamSteps, 0)).toBeNull();
    expect(expandedAtStep(beamSteps, 2)).toEqual({ id: 1, distance: 0.1, pos: null });
    expect(beamStepAt(beamSteps, 1)?.iteration).toBe(0);
    expect(beamStepAt([], 1)).toBeUndefined();
  });
});

describe('probes', () => {
  it('reveals probes in rank order up to the step', () => {
    expect(visibleProbes(probes, 0)).toEqual([]);
    expect(visibleProbes(probes, 1).map((x) => x.cluster_id)).toEqual([3]);
    expect(visibleProbes(probes, 2).map((x) => x.cluster_id)).toEqual([3, 7]);
    expect(visibleProbes(probes, 99)).toHaveLength(3);
  });

  it('does not mutate the input', () => {
    const copy = probes.map((x) => x.rank);
    visibleProbes(probes, 3);
    expect(probes.map((x) => x.rank)).toEqual(copy);
  });

  it('handles empty / missing probes', () => {
    expect(visibleProbes([], 2)).toEqual([]);
    expect(visibleProbes(undefined, 2)).toEqual([]);
    expect(currentProbe(undefined, 1)).toBeUndefined();
  });

  it('exposes the current probe and the result ids found so far', () => {
    expect(currentProbe(probes, 1)?.cluster_id).toBe(3);
    expect(currentProbe(probes, 3)?.cluster_id).toBe(9);
    expect(currentProbe(probes, 0)).toBeUndefined();
    expect(probeResultIds(probes, 1).sort()).toEqual([4, 5]);
    expect(probeResultIds(probes, 3).sort()).toEqual([4, 5, 6]);
    expect(probeResultIds(probes, 0)).toEqual([]);
  });
});
