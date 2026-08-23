'use client';

import { useMemo } from 'react';
import { Vector3D } from '../../lib/types';
import { getClusterColor, hexToRgb } from '../../lib/math';

interface CentroidNodesProps {
  points: Vector3D[];
  clusters: number[];
}

export default function CentroidNodes({ points, clusters }: CentroidNodesProps) {
  const centroids = useMemo(() => {
    const clusterMap: Record<number, { sumX: number; sumY: number; sumZ: number; count: number }> = {};

    points.forEach((p, i) => {
      const c = clusters[i] ?? 0;
      if (!clusterMap[c]) {
        clusterMap[c] = { sumX: 0, sumY: 0, sumZ: 0, count: 0 };
      }
      clusterMap[c].sumX += p.x;
      clusterMap[c].sumY += p.y;
      clusterMap[c].sumZ += p.z;
      clusterMap[c].count++;
    });

    return Object.entries(clusterMap).map(([clusterStr, val]) => {
      const cId = parseInt(clusterStr, 10);
      return {
        id: cId,
        x: val.sumX / val.count,
        y: val.sumY / val.count,
        z: val.sumZ / val.count,
        count: val.count,
        color: getClusterColor(cId),
      };
    });
  }, [points, clusters]);

  return (
    <group>
      {centroids.map((c) => (
        <group key={c.id} position={[c.x, c.y, c.z]}>
          {/* Centroid sphere */}
          <mesh>
            <sphereGeometry args={[1.2, 16, 16]} />
            <meshStandardMaterial
              color={c.color}
              emissive={c.color}
              emissiveIntensity={0.8}
              roughness={0.2}
              metalness={0.8}
            />
          </mesh>
          {/* Bounding aura */}
          <mesh>
            <sphereGeometry args={[2.0, 16, 16]} />
            <meshBasicMaterial
              color={c.color}
              transparent
              opacity={0.15}
              wireframe
            />
          </mesh>
        </group>
      ))}
    </group>
  );
}
