import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#08090d",
        surface: "#0f1118",
        surfaceBorder: "#1e2230",
        primary: "#00f0ff",
        secondary: "#7000ff",
        accent: "#ff007b",
        success: "#00ff66",
        warning: "#ffaa00",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Outfit", "Inter", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 20px rgba(0, 240, 255, 0.35)",
        glowAccent: "0 0 20px rgba(255, 0, 123, 0.35)",
      },
    },
  },
  plugins: [],
};
export default config;
