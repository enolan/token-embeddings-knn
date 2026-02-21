// In Workers, importing a .wasm file gives you a WebAssembly.Module
// (not the instance exports that brotli-dec-wasm's .d.ts describes)
declare module "brotli-dec-wasm/web/bg.wasm" {
  const module: WebAssembly.Module;
  export default module;
}
