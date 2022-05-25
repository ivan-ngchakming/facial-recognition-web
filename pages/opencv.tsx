import cv, { float } from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import Head from "next/head";
import React, { useRef, useEffect, useState } from "react";
import styles from "../styles/Home.module.css";
import axios from "axios";
import { Tensor } from "onnxruntime-web";
import Navbar from "../components/Navbar";
import nms from "../utils/nms";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>();
  const imgRef = useRef<HTMLImageElement>();
  const detImageRef = useRef<cv.Mat>();
  const [session, setSession] = useState<ort.InferenceSession>();
  const [status, setStatus] = useState("");

  const initModel = async () => {
    // buffalo_s/det_500m.ort
    // buffalo_s/det_500m.with_runtime_opt.ort
    // buffalo_l/det_10g.ort
    try {
      const modelBuffer = await axios.get<Uint8Array>(
        `${API_URL}/models/buffalo_s/det_500m.with_runtime_opt.ort`,
        {
          responseType: "arraybuffer",
          withCredentials: false,
        }
      );

      const _session = await ort.InferenceSession.create(modelBuffer.data);
      setSession(_session);
    } catch (error) {
      console.error(error);
    }
  };

  function handleFileChange(e) {
    const files = e.target.files;
    const image = imgRef.current;
    const canvas = canvasRef.current;
    if (!files || files.length === 0 || !image || !canvas) {
      return;
    }

    image.onload = async () => {
      let mat = cv.imread(image);
      if (mat.rows < mat.cols) {
        image.style.height = 'auto';
        image.style.width = '640px';
        mat = cv.imread(image);
      }
      await cv.imshow(canvas, mat);
    }

    const file = files[0];
    let fr = new FileReader();
    fr.readAsDataURL(file);
    fr.onload = async (event: ProgressEvent<FileReader>) => {
      if (event.target?.readyState === FileReader.DONE) {
        image.src = event.target.result;
      }
    };
  }

  function arrayMultiply(array: Float32Array, factor: number) {
    for (let i = 0; i < array.length; i++) {
      array[i] = array[i] * factor;
    }
    return array;
  }

  async function forward(image: cv.Mat, threshold: float) {
    if (!session) return;

    let scores = [];
    let bboxes = [];
    let distances = [];
    let kpss;
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
      kpsPreds = netOuts[outNames[idx + fmc * 2]].data;

      bboxPreds = arrayMultiply(bboxPreds, stride[idx]);
      kpsPreds = arrayMultiply(kpsPreds, stride[idx]);

      height = Math.round(inputHeight / stride[idx]);
      width = Math.round(inputWeight / stride[idx]);

      // TODO: get anchorCenters with key (height, width, stride) from cache if possible
      let center;
      const anchorCenters = new Array<Float32Array>(
        height * width * numAnchors
      );
      for (let i = 0; i < anchorCenters.length; i++) {
        center = new Float32Array(2);
        center[0] = (((i / 2) | 0) % height) * stride[idx];
        center[1] = ((((i / 2) | 0) / height) | 0) * stride[idx];
        anchorCenters[i] = center;
      }

      // TODO: cache anchor centers if cache is not full

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
            anchorCenters[i][0] - bboxPreds[i * 4 + 0],
            anchorCenters[i][1] - bboxPreds[i * 4 + 1],
            anchorCenters[i][0] + bboxPreds[i * 4 + 2],
            anchorCenters[i][1] + bboxPreds[i * 4 + 3],
          ]);
        }
      }
    }

    return { scores, bboxes, distances, kpss };
  }

  async function runDetection() {
    const scoreThreshold = 0.5;
    const nmsThreshold = 0.4;

    let image = imgRef.current;
    const canvas = canvasRef.current;
    if (!canvas || !image || !session) {
      return;
    }

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
    
    // // TODO: Remove for debugging only
    // await cv.imshow(canvas, detImage);
    // debugger

    // cv.cvtColor(detImage, detImage, cv.COLOR_RGB2BGR);

    detImageRef.current = detImage;

    console.time("forward");
    const { scores, bboxes } = await forward(detImage, scoreThreshold);
    console.timeEnd("forward");

    const ctx = canvas.getContext("2d");

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

    // cv.NMSBoxes(bboxRects, scoreVector, scoreThreshold, nmsThreshold, finalIdxs);

    for (let bbox of picks) {
      ctx.beginPath();
      ctx.rect(bbox.x1, bbox.y1, bbox.width, bbox.height);
      ctx.strokeStyle = "red";
      ctx.stroke();
    }
  }

  useEffect(() => {
    setStatus("Loading model...");
    initModel();
  }, []);

  useEffect(() => {
    if (session) {
      setStatus("Model loaded!");
      console.log(session);
    }
  }, [session]);

  return (
    <div className={styles.container}>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className={styles.main}>
        <Navbar />
        <h1>Test</h1>
        <div>
          <label className="btn">file: </label>
          <input id="file" type="file" onChange={handleFileChange} />
          <button type="button" onClick={runDetection} disabled={!session}>
            Detect Face
          </button>
        </div>
        <div style={{ display: "flex" }}>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <img
              src=""
              alt="testimg"
              ref={imgRef}
              style={{ width: "auto", height: 640, margin: 32 }}
            />
            <p>512 x 640</p>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <canvas
              width={640}
              height={640}
              style={{ margin: 32 }}
              ref={canvasRef}
            ></canvas>
            <p>640 x 640</p>
          </div>
        </div>
        <h2>{status}</h2>
        {session && (
          <div>
            <pre>{JSON.stringify(session, null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  );
}
