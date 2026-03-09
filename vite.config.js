import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    open: true,
    proxy: {
      '/tamarind-api': {
        target: 'https://app.tamarind.bio',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/tamarind-api/, '/api'),
        secure: true,
        followRedirects: true,
      },
    },
  }
})
