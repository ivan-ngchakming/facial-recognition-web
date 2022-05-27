import type { Mat } from 'mirada';
import * as ort from "onnxruntime-web";
import axios from 'axios';
import nms from "../utils/nms";
import { detect } from '../utils';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

const ctx: Worker = self as unknown as Worker;

let session: ort.InferenceSession;

// @ts-ignore
ctx.importScripts('https://docs.opencv.org/4.5.5/opencv.js');

ort.env.wasm.wasmPaths = 'pages/'

axios.get<Uint8Array>(
  `${API_URL}/models/buffalo_s/det_500m.with_runtime_opt.ort`,
  {
    responseType: "arraybuffer",
    withCredentials: false,
  }
).then(res => res.data).then(buffer => {
  ort.InferenceSession.create(buffer).then(_session => {
    session = _session;
    ctx.postMessage({ msg: 'info', info: 'Web worker has started :)'});
  });
});

ctx.addEventListener('message', async (event) => {
  const { msg } = event.data;
  if (msg === 'detect') {
    const { imageData } = event.data;
    try {
      const picks = await detect(imageData, session, 0.5, 0.3);
      ctx.postMessage({ bboxs: picks, msg: 'detect' });
    } catch (error) {
      ctx.postMessage({ error, msg: 'error' });
    }
  }
})
