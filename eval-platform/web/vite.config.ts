import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发态把 /api 代理到本地 API 服务；SSE 走同一代理，不引入缓冲。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8080", changeOrigin: true },
    },
  },
  build: { sourcemap: true },
});
