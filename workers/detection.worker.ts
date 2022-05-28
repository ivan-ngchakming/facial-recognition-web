import axios from 'axios';
import * as ort from "onnxruntime-web";
import { detect } from '../utils';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

const ctx: Worker = self as unknown as Worker;

let session: ort.InferenceSession;

// @ts-ignore
ctx.importScripts('https://docs.opencv.org/4.5.5/opencv.js');

async function init() {
  ort.env.wasm.wasmPaths = 'pages/'

  const res = await axios.get<Uint8Array>(
    `${API_URL}/models/buffalo_s/det_500m.with_runtime_opt.ort`,
    {
      responseType: "arraybuffer",
      withCredentials: false,
    }
  );

  session = await ort.InferenceSession.create(res.data);

  ctx.postMessage({ msg: 'info', info: 'Web worker has started :)' });
}

ctx.addEventListener('message', async (event) => {
  const { msg } = event.data;
  switch (msg) {
    case 'init':
      await init();
      postMessage({ msg: 'ready' });
      break;
    case 'detect':
      const { imageData } = event.data;
      try {
        const picks = await detect(imageData, session, 0.5, 0.3);
        ctx.postMessage({ bboxs: picks, msg: 'detect' });
      } catch (error) {
        ctx.postMessage({ error, msg: 'error' });
      }
      break;
    default:
      ctx.postMessage({ msg: 'error', error: 'Unknown message' });
  }
})
