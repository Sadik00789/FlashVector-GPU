'use client';

import { useMemo, useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { SearchResponse, TraversalHop3D, Vector3D } from '../../lib/types';
import {
  candidatesAtStep,
  evalsForStep,
  expandedAtStep,
  hopsForStep,
  probeResultIds,
  scrubLength,
  scrubMode,
  visibleProbes,
  currentProbe,
} from '../../lib/beam';

interface TraversalBeamProps {
  response: SearchResponse | null;
  points: Vector3D[];
  step: number;
}

const EMPTY_HOPS: TraversalHop3D[] = [];

function buildSegments(hops: TraversalHop3D[]): THREE.BufferGeometry | null {
  if (hops.length === 0) return null;
  const positions = new Float32Array(hops.length * 6);
  hops.forEach((hop, idx) => {
    const a = hop.from_pos;
    const b = hop.to_pos;
    positions.set([a.x, a.y, a.z, b.x, b.y, b.z], idx * 6);
  });
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  return geom;
}

function useSegments(hops: TraversalHop3D[]): THREE.BufferGeometry | null {
  const geom = useMemo(() => buildSegments(hops), [hops]);
  useEffect(() => {
    return () => {
      geom?.dispose();
    };
  }, [geom]);
  return geom;
}

function PulseMarker({
  pos,
  radius,
  color,
  intensity = 1.2,
  pulse = false,
  opacity = 1,
}: {
  pos: Vector3D;
  radius: number;
  color: string;
  intensity?: number;
  pulse?: boolean;
  opacity?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!pulse || !ref.current) return;
    const s = 1.0 + 0.25 * Math.sin(clock.getElapsedTime() * 6);
    ref.current.scale.set(s, s, s);
  });
  return (
    <mesh ref={ref} position={[pos.x, pos.y, pos.z]}>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={intensity}
        transparent={opacity < 1}
        opacity={opacity}
      />
    </mesh>
  );
}

const PROBE_COLORS = ['#c084fc', '#a78bfa', '#818cf8', '#60a5fa', '#38bdf8', '#22d3ee', '#2dd4bf', '#34d399'];

export default function TraversalBeam({ response, points, step }: TraversalBeamProps) {
  const hops = response?.hops ?? EMPTY_HOPS;
  const beamSteps = response?.beam_steps;
  const probes = response?.probes;
  const results = response?.results;

  const mode = scrubMode(response);
  const length = scrubLength(response);
  const atEnd = step >= length;

  // Everything below is derived purely from `step` so scrubbing is lossless.
  const visibleHops = useMemo(
    () => (mode === 'probe' ? EMPTY_HOPS : hopsForStep(hops, beamSteps, step)),
    [mode, hops, beamSteps, step],
  );
  const currentEvals = useMemo(
    () => (mode === 'probe' ? EMPTY_HOPS : evalsForStep(hops, beamSteps, step)),
    [mode, hops, beamSteps, step],
  );
  const olderHops = useMemo(
    () => visibleHops.filter((h) => !currentEvals.includes(h)),
    [visibleHops, currentEvals],
  );

  const olderGeom = useSegments(olderHops);
  const currentGeom = useSegments(currentEvals);

  const lookup = (id: number, pos: Vector3D | null | undefined): Vector3D | undefined =>
    pos ?? points[id];

  const candidates = useMemo(() => candidatesAtStep(beamSteps, step), [beamSteps, step]);
  const expanded = useMemo(() => expandedAtStep(beamSteps, step), [beamSteps, step]);
  const shownProbes = useMemo(() => visibleProbes(probes, step), [probes, step]);
  const activeProbe = useMemo(() => currentProbe(probes, step), [probes, step]);

  // Final top-k is revealed at the end of the replay (legacy hop mode: always).
  const topKPositions = useMemo(() => {
    if (!results || results.length === 0 || points.length === 0) return [];
    let ids = results.map((r) => r.id);
    if (mode === 'probe') {
      const found = new Set(probeResultIds(probes, step));
      ids = ids.filter((id) => found.has(id));
    } else if (mode === 'beam' && !atEnd) {
      ids = [];
    }
    return ids.map((id) => points[id]).filter((p): p is Vector3D => p !== undefined);
  }, [results, points, mode, probes, step, atEnd]);

  const entry = mode === 'hop' && visibleHops.length > 0 ? visibleHops[0].from_pos : null;

  return (
    <group>
      {/* Earlier distance evaluations (dim) */}
      {olderGeom && (
        <lineSegments geometry={olderGeom}>
          <lineBasicMaterial color="#1fa9d6" transparent opacity={0.35} />
        </lineSegments>
      )}

      {/* Current iteration's distance evaluations (emphasized) */}
      {currentGeom && (
        <lineSegments geometry={currentGeom}>
          <lineBasicMaterial color="#ff2e88" transparent opacity={1} />
        </lineSegments>
      )}

      {/* Legacy entry node marker */}
      {entry && <PulseMarker pos={entry} radius={1.5} color="#ff0055" intensity={1.2} />}

      {/* Beam candidate queue: amber = visited (expanded), cyan = unvisited */}
      {candidates.map((c) => {
        const pos = lookup(c.id, c.pos);
        if (!pos) return null;
        return (
          <PulseMarker
            key={`cand-${c.id}`}
            pos={pos}
            radius={c.visited ? 0.9 : 0.8}
            color={c.visited ? '#ffb020' : '#00e5ff'}
            intensity={1.0}
            opacity={0.9}
          />
        );
      })}

      {/* Expanded node of the current iteration */}
      {expanded &&
        (() => {
          const pos = lookup(expanded.id, expanded.pos);
          return pos ? (
            <PulseMarker pos={pos} radius={1.6} color="#ff0055" intensity={1.6} pulse />
          ) : null;
        })()}

      {/* IVF probes: centroids revealed up to the step */}
      {shownProbes.map((pr) => {
        const pos = pr.centroid_pos;
        if (!pos) return null;
        const isCurrent = activeProbe?.rank === pr.rank;
        return (
          <PulseMarker
            key={`probe-${pr.rank}`}
            pos={pos}
            radius={isCurrent ? 2.2 : 1.4}
            color={isCurrent ? '#ff0055' : PROBE_COLORS[pr.rank % PROBE_COLORS.length]}
            intensity={isCurrent ? 1.6 : 0.9}
            pulse={isCurrent}
            opacity={isCurrent ? 1 : 0.8}
          />
        );
      })}

      {/* Top-K target markers */}
      {topKPositions.map((pos, i) => (
        <PulseMarker key={`topk-${i}`} pos={pos} radius={1.0} color="#00ff66" intensity={1.5} pulse />
      ))}
    </group>
  );
}
