'use client';

import { useMemo, useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { GpuSearchResult, TraversalHop3D, Vector3D } from '../../lib/types';

interface TraversalBeamProps {
  hops: TraversalHop3D[];
  results: GpuSearchResult[];
  points: Vector3D[];
}

export default function TraversalBeam({ hops, results, points }: TraversalBeamProps) {
  const lineRef = useRef<THREE.LineSegments>(null);
  const pulseGroupRef = useRef<THREE.Group>(null);

  // Build line geometry from hops
  const lineGeometry = useMemo(() => {
    if (!hops || hops.length === 0) return null;

    const positions = new Float32Array(hops.length * 6);
    const colors = new Float32Array(hops.length * 6);

    hops.forEach((hop, idx) => {
      const p1 = hop.from_pos;
      const p2 = hop.to_pos;

      // Start pos
      positions[idx * 6 + 0] = p1.x;
      positions[idx * 6 + 1] = p1.y;
      positions[idx * 6 + 2] = p1.z;

      // End pos
      positions[idx * 6 + 3] = p2.x;
      positions[idx * 6 + 4] = p2.y;
      positions[idx * 6 + 5] = p2.z;

      // Color gradient from electric blue to bright hot pink
      const t = idx / Math.max(1, hops.length - 1);
      const r = 0.0 + t * 1.0;
      const g = 0.9 - t * 0.7;
      const b = 1.0 - t * 0.3;

      colors[idx * 6 + 0] = r;
      colors[idx * 6 + 1] = g;
      colors[idx * 6 + 2] = b;

      colors[idx * 6 + 3] = r;
      colors[idx * 6 + 4] = g;
      colors[idx * 6 + 5] = b;
    });

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geom;
  }, [hops]);

  // Clean up geometry to prevent memory leaks
  useEffect(() => {
    return () => {
      if (lineGeometry) lineGeometry.dispose();
    };
  }, [lineGeometry]);

  // Target Top-K point positions
  const topKPositions = useMemo(() => {
    if (!results || results.length === 0 || !points || points.length === 0) return [];
    return results
      .map((r) => points[r.id])
      .filter((p): p is Vector3D => p !== undefined);
  }, [results, points]);

  // Smooth pulse animation
  useFrame(({ clock }) => {
    const elapsed = clock.getElapsedTime();
    if (pulseGroupRef.current) {
      const scale = 1.0 + 0.25 * Math.sin(elapsed * 6);
      pulseGroupRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <group>
      {/* 3D Search Graph Traversal Rays */}
      {lineGeometry && (
        <lineSegments ref={lineRef} geometry={lineGeometry}>
          <lineBasicMaterial
            vertexColors
            transparent
            opacity={0.85}
            linewidth={2}
          />
        </lineSegments>
      )}

      {/* Glowing Entry Node Marker */}
      {hops.length > 0 && hops[0]?.from_pos && (
        <mesh position={[hops[0].from_pos.x, hops[0].from_pos.y, hops[0].from_pos.z]}>
          <sphereGeometry args={[1.5, 16, 16]} />
          <meshStandardMaterial
            color="#ff0055"
            emissive="#ff0055"
            emissiveIntensity={1.2}
          />
        </mesh>
      )}

      {/* Pulsing Top-K Target Results Markers */}
      <group ref={pulseGroupRef}>
        {topKPositions.map((pos, i) => (
          <mesh key={i} position={[pos.x, pos.y, pos.z]}>
            <sphereGeometry args={[1.0, 16, 16]} />
            <meshStandardMaterial
              color="#00ff66"
              emissive="#00ff66"
              emissiveIntensity={1.5}
            />
          </mesh>
        ))}
      </group>
    </group>
  );
}
