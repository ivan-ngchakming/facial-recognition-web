import type { CV, FS } from 'mirada';

declare module '*.worker.ts' {
  interface WebpackWorker extends Worker {
    constructor();
  }

  export default WebpackWorker;
}
