'use client';

import { useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

import { GpuSearchResult, TraversalHop3D, Vector3D } from '../../lib/types';
import { getClusterColor, hexToRgb } from '../../lib/math';
import CentroidNodes from './CentroidNodes';
import TraversalBeam from './TraversalBeam';

interface EmbeddingSpaceProps {
  points: Vector3D[];
  clusters: number[];
  hops: TraversalHop3D[];
  results: GpuSearchResult[];
}

function VectorPointCloud({ points, clusters }: { points: Vector3D[]; clusters: number[] }) {
  const pointsRef = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    if (!points || points.length === 0) return null;

    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    points.forEach((p, i) => {
      positions[i * 3 + 0] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;

      const cluster = clusters[i] ?? 0;
      const hex = getClusterColor(cluster);
      const [r, g, b] = hexToRgb(hex);

      colors[i * 3 + 0] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    });

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geom;
  }, [points, clusters]);

  // Clean up geometry to prevent WebGL context leaks
  useEffect(() => {
    return () => {
      if (geometry) geometry.dispose();
    };
  }, [geometry]);

  if (!geometry) return null;

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={0.65}
        vertexColors
        transparent
        opacity={0.7}
        sizeAttenuation
      />
    </points>
  );
}

export default function EmbeddingSpace({
  points,
  clusters,
  hops,
  results,
}: EmbeddingSpaceProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [0, 20, 80], fov: 60 }}
        gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        className="bg-[#08090d]"
      >
        <color attach="background" args={['#08090d']} />
        <ambientLight intensity={0.6} />
        <pointLight position={[50, 50, 50]} intensity={1.2} color="#00f0ff" />
        <pointLight position={[-50, -50, -50]} intensity={0.8} color="#ff007b" />
        <directionalLight position={[0, 40, 20]} intensity={0.8} />

        {/* Ambient starfield background */}
        <Stars radius={150} depth={50} count={3000} factor={4} saturation={1} fade speed={1} />

        {/* Vector Point Cloud */}
        <VectorPointCloud points={points} clusters={clusters} />

        {/* Voronoi / IVF Centroid Nodes */}
        <CentroidNodes points={points} clusters={clusters} />

        {/* Traversal Beam Routing & Top-K Targets */}
        <TraversalBeam hops={hops} results={results} points={points} />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.8}
          zoomSpeed={1.0}
          minDistance={10}
          maxDistance={300}
        />
      </Canvas>

      {/* Viewport Overlay Controls Hint */}
      <div className="absolute bottom-4 left-4 text-xs text-slate-400 glass-panel px-3 py-1.5 rounded-lg pointer-events-none flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        Rotate: Left Click + Drag | Pan: Right Click | Zoom: Scroll
      </div>
    </div>
  );
}
