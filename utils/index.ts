import * as ort from "onnxruntime-web";
import { Tensor } from "onnxruntime-web";
import nms from "./nms";

export function arrayMultiply(array: Float32Array, factor: number) {
  for (let i = 0; i < array.length; i++) {
    array[i] = array[i] * factor;
  }
  return array;
}

async function forward(session: ort.InferenceSession, image: cv.Mat, threshold: float) {
  if (!session) return;

  let scores = [];
  let bboxes = [];
  let distances = [];
  // let kpss;
  let scoresPred, bboxPreds, bboxPredsDims, kpsPreds;
  let height, width;

  // constants to be initialized at constructor
  const fmc = 3;
  const inputMean = 127.5;
  const [inputHeight, inputWeight] = [640, 640];
  const numAnchors = 2;

  // @ts-ignore yes handler does exist on session
  const outNames = session.handler.outputNames; // ['448', '471', '494', '451', '474', '497', '454', '477', '500']

  const blob = cv.blobFromImage(
    image,
    1 / 128,
    new cv.Size(inputWeight, inputHeight),
    new cv.Scalar(inputMean, inputMean, inputMean),
    true
  );
  const tensor = new Tensor(blob.data32F, [1, 3, 640, 640]);

  const netOuts = await session.run({ "input.1": tensor });

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

    // // TODO: get anchorCenters with key (height, width, stride) from cache if possible
    // let center;
    // const anchorCenters = new Array<Float32Array>(
    //   height * width * numAnchors
    // );
    // for (let i = 0; i < anchorCenters.length; i++) {
    //   center = new Float32Array(2);
    //   center[0] = (((i / 2) | 0) % height) * stride[idx];
    //   center[1] = ((((i / 2) | 0) / height) | 0) * stride[idx];
    //   anchorCenters[i] = center;
    // }
    // // TODO: cache anchor centers if cache is not full

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

export async function detect(image: HTMLImageElement, session: ort.InferenceSession, scoreThreshold: number, nmsThreshold: number) {
  let mat = cv.imread(image);
  await cv.imshow(canvas, mat);
  await cv.cvtColor(mat, mat, cv.COLOR_RGB2BGR);

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
  const newMat = new cv.Mat(newWidth, newHeight, mat.type());

  await cv.resize(mat, newMat, new cv.Size(newWidth, newHeight));

  const detImage = new cv.Mat(640, 640, newMat.type());

  let s = new cv.Scalar(0, 0, 0, 255); // Black
  cv.copyMakeBorder(
    newMat,
    detImage,
    0,
    640 - newHeight,
    0,
    640 - newWidth,
    cv.BORDER_CONSTANT,
    s
  );
  newMat.delete();

  console.time("forward");
  const { bboxes } = await forward(session, detImage, scoreThreshold);
  console.timeEnd("forward");

  detImage.delete()

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

  return picks;
}
