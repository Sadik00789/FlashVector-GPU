/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output symlinks pnpm packages, which requires elevated privileges on Windows.
  ...(process.platform === 'win32' ? {} : { output: 'standalone' }),
  reactStrictMode: true,
  transpilePackages: ['three', '@react-three/fiber', '@react-three/drei'],
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/api/:path*',
      },
    ];
  },
};

export default nextConfig;
