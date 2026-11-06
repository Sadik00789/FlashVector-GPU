const CLUSTER_PALETTE = [
  '#00f0ff',
  '#ff007b',
  '#7000ff',
  '#00ff66',
  '#ffaa00',
  '#ff4d4d',
  '#4d79ff',
  '#00ffc8',
  '#ff00ff',
  '#a3ff00',
  '#ff6600',
  '#00b3ff',
  '#b300ff',
  '#00ff9d',
  '#ffd500',
  '#ff0055',
];

export function getClusterColor(cluster: number): string {
  if (!Number.isFinite(cluster)) return CLUSTER_PALETTE[0];
  const len = CLUSTER_PALETTE.length;
  const index = ((Math.trunc(cluster) % len) + len) % len;
  return CLUSTER_PALETTE[index];
}

export function hexToRgb(hex: string): [number, number, number] {
  const normalized = hex.replace('#', '');
  const full =
    normalized.length === 3
      ? normalized
          .split('')
          .map((c) => c + c)
          .join('')
      : normalized;
  const value = Number.parseInt(full, 16);
  if (!Number.isFinite(value)) return [1, 1, 1];
  return [((value >> 16) & 0xff) / 255, ((value >> 8) & 0xff) / 255, (value & 0xff) / 255];
}

export function formatLatency(us: number): string {
  if (!Number.isFinite(us) || us <= 0) return '0.0 µs';
  if (us < 1000) return `${us.toFixed(1)} µs`;
  return `${(us / 1000).toFixed(2)} ms`;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '0';
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return `${Math.round(value)}`;
}
