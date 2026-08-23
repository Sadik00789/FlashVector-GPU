import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FlashVector-GPU | Real-Time SIMT Vector Search Visualizer",
  description: "Next-generation GPU vector search engine powered by CUDA sm_86 warp-cooperative beam search, dynamic shared memory ADC, and sub-millisecond 3D trajectory streaming.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-background text-slate-100 antialiased selection:bg-primary selection:text-black">
        {children}
      </body>
    </html>
  );
}
