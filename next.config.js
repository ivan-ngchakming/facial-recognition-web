/** @type {import('next').NextConfig} */
const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  reactStrictMode: true,
  images: {
    domains: [
      'localhost',
      'facial-recognition-api.ivan0313.tk',
    ],
  },
  typescript: {
    // TODO: Remove
    ignoreBuildErrors: true,
  },
  eslint: {
    // TODO: Remove
    ignoreDuringBuilds: true,
  },
  webpack5: true,
  webpack: (config, {}) => {
    config.resolve.extensions.push(".ts", ".tsx");
    config.resolve.fallback = { fs: false, path:false, "crypto": false };

    config.plugins.push(
      new NodePolyfillPlugin(),
      new CopyPlugin({
        patterns: [
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm.wasm",
            to: "static/chunks/pages",
          },
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm",
            to: "static/chunks/pages",
          },
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm",
            to: "static/chunks/pages/opencv",
          },
        ],
      })
    );

    return config;
  },
};
