'use client';

import { useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

import { GpuSearchResult, TraversalHop3D, Vector3D } from '../../lib/types';
import { getClusterColor } from '../../lib/math';
import CentroidNodes from './CentroidNodes';
import TraversalBeam from './TraversalBeam';

interface EmbeddingSpaceProps {
  points: Vector3D[];
  clusters: number[];
  hops: TraversalHop3D[];
  results: GpuSearchResult[];
  currentHop: number;
}

function VectorPointCloud({ points, clusters }: { points: Vector3D[]; clusters: number[] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || points.length === 0) return;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      dummy.position.set(p.x, p.y, p.z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      color.set(getClusterColor(clusters[i] ?? 0));
      mesh.setColorAt(i, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  }, [points, clusters]);

  if (points.length === 0) return null;

  return (
    <instancedMesh
      key={points.length}
      ref={meshRef}
      args={[undefined, undefined, points.length]}
      frustumCulled={false}
    >
      <sphereGeometry args={[0.3, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export default function EmbeddingSpace({
  points,
  clusters,
  hops,
  results,
  currentHop,
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

        {/* Vector Point Cloud (single instanced draw call) */}
        <VectorPointCloud points={points} clusters={clusters} />

        {/* Voronoi / IVF Centroid Nodes */}
        <CentroidNodes points={points} clusters={clusters} />

        {/* Traversal Beam Routing & Top-K Targets */}
        <TraversalBeam
          hops={hops}
          results={results}
          points={points}
          currentHop={currentHop}
        />

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
