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

function arrayMultiply(array: Float32Array, factor: number) {
  for (let i = 0; i < array.length; i++) {
    array[i] = array[i] * factor;
  }
  return array;
}

async function forward(session: ort.InferenceSession, image: Mat, threshold: number) {
  if (!session) return;

  let scores = [];
  let bboxes = [];
  let distances = [];
  let scoresPred, bboxPreds, bboxPredsDims;
  let height, width;

  // constants to be initialized at constructor
  const fmc = 3;
  const inputMean = 127.5;
  const [inputHeight, inputWeight] = [640, 640];

  // @ts-ignore yes handler does exist on session
  const outNames = session.handler.outputNames;

  const blob = cv.blobFromImage(
    image,
    1 / 128,
    new cv.Size(inputWeight, inputHeight),
    new cv.Scalar(inputMean, inputMean, inputMean),
    true
  );
  
  const tensor = new ort.Tensor(blob.data32F, [1, 3, 640, 640]);
  const netOuts = await session.run({ "input.1": tensor });

  blob.delete();

  const stride = [8, 16, 32];
  for (let idx = 0; idx < 3; idx++) {

    scoresPred = netOuts[outNames[idx]].data;
    bboxPreds = Float32Array.from(netOuts[outNames[idx + fmc]].data);
    bboxPredsDims = netOuts[outNames[idx + fmc]].dims;
    // kpsPreds = netOuts[outNames[idx + fmc * 2]].data;

    bboxPreds = arrayMultiply(bboxPreds, stride[idx]);
    // kpsPreds = arrayMultiply(kpsPreds, stride[idx]);

    height = Math.round(inputHeight / stride[idx]);
    width = Math.round(inputWeight / stride[idx]);

    const getAnchorCenter = (x: number, i: number, height: number) => {
      if (x === 0) {
        return (((i / 2) | 0) % height) * stride[idx];
      } else {
        return ((((i / 2) | 0) / height) | 0) * stride[idx];
      }
    }

    const posIdxs = [];

    for (let i = 0; i < scoresPred.length; i++) {
      if (scoresPred[i] > threshold) {
        posIdxs.push(i);
        scores.push(scoresPred[i]);

        distances.push([
          bboxPreds[i * 4],
          bboxPreds[i * 4 + 1],
          bboxPreds[i * 4 + 2],
          bboxPreds[i * 4 + 3],
        ]);

        bboxes.push([
          getAnchorCenter(0, i, height) - bboxPreds[i * 4 + 0],
          getAnchorCenter(1, i, height) - bboxPreds[i * 4 + 1],
          getAnchorCenter(0, i, height) + bboxPreds[i * 4 + 2],
          getAnchorCenter(1, i, height) + bboxPreds[i * 4 + 3],
        ]);
      }
    }
  }

  return { scores, bboxes, distances };
}

export async function _detect(imageData: ImageData, session: ort.InferenceSession, scoreThreshold: number, nmsThreshold: number) {
  // let mat = cv.imread(imageData);
  const mat = cv.matFromImageData(imageData);
  await cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);

  // resize image to fit the detection model input size (640, 640)
  let newHeight, newWidth;
  const imageRatio = mat.rows / mat.cols; // height / width
  if (imageRatio > 1) {
    newHeight = 640;
    newWidth = Math.round(newHeight / imageRatio);
  } else {
    newWidth = 640;
    newHeight = Math.round(newWidth * imageRatio);
  }

  const detScale = newHeight / mat.rows;

  await cv.resize(mat, mat, new cv.Size(newWidth, newHeight));

  // const detImage = new cv.Mat(640, 640, newMat.type());

  let s = new cv.Scalar(0, 0, 0, 255); // Black
  cv.copyMakeBorder(
    mat,
    mat,
    0,
    640 - newHeight,
    0,
    640 - newWidth,
    cv.BORDER_CONSTANT,
    s
  );

  const { bboxes } = await forward(session, mat, scoreThreshold);

  const bboxRects = [];
  for (let i = 0; i < bboxes.length; i++) {
    for (let j = 0; j < bboxes[i].length; j++) {
      bboxes[i][j] = bboxes[i][j] / detScale;
    }
    bboxRects.push([
      ...bboxes[i],
      bboxes[i][2] - bboxes[i][0],
      bboxes[i][3] - bboxes[i][1],
    ]);
  }

  const picks = nms(bboxRects, nmsThreshold);

  // Somebody tell me why cv.Mat are not garbage collected
  mat.delete();
  // detImage.delete();
  // newMat.delete();

  return picks;
}

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
