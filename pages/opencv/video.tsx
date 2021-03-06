import axios from "axios";
import Head from "next/head";
import React, { useCallback, useEffect, useRef, useState } from "react";
import Navbar from "../../components/Navbar";
import styles from "../../styles/Home.module.css";
import { BBox } from "../../types";


const FPS = 30;
const API_URL = process.env.NEXT_PUBLIC_API_URL;

export default function Home() {
  const workerRef = useRef<Worker>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const bboxsRef = useRef<BBox[]>();
  const imageSentRef = useRef(false);
  const detectResultRef = useRef<any>();
  const [detectResult, setDetectResult] = useState<any>();
  const [identifying, setIdentifying] = useState(false);

  function handleLoadedMetadata() {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) {
      return;
    }
    video.height = video.videoHeight;
    video.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.width = video.videoWidth;
    setTimeout(processVideo, 0);
  }

  async function processVideo() {
    let begin = Date.now();

    const video = videoRef.current;
    const bboxs = bboxsRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.drawImage(video, 0, 0, video.width, video.height);
    if (bboxs && bboxs.length > 0) {
      for (let bbox of bboxs) {
        ctx.beginPath();
        ctx.rect(bbox.x1, bbox.y1, bbox.width, bbox.height);
        ctx.strokeStyle = "red";
        ctx.stroke();

        ctx.font = "16px Arial";
        ctx.fillStyle = "red";
        // name
        if (detectResultRef.current && detectResultRef.current.length > 0) {
          ctx.fillText(
            detectResultRef.current[0].face.profile.name,
            bbox.x1,
            bbox.y1 - 5
          );
        } else {
          ctx.fillText("loading...", bbox.x1, bbox.y1 - 5);
        }
      }
    }

    let delay = 1000 / FPS - (Date.now() - begin);
    setTimeout(processVideo, delay);
  }

  function startVideo() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      return;
    }

    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err) {
        console.log("An error occurred! " + err);
      });
  }

  const handleWork = useCallback(async () => {
    const canvas = canvasRef.current;
    const worker = workerRef.current;
    const video = videoRef.current;
    if (!canvas || !worker || !video) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    worker.postMessage({ msg: "detect", imageData });
  }, []);

  async function sendImage() {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    setIdentifying(true);
    canvas.toBlob(async (blob) => {
      if (!blob) {
        return;
      }
      let file = new File([blob], "fileName.jpg", { type: "image/jpeg" });
      const formData = new FormData();
      formData.append("file", file, file.name);
      const data = await axios.post(API_URL + `/faces/search`, formData);
      setDetectResult(data.data[0]);
      detectResultRef.current = data.data[0];
      setIdentifying(false);
    }, "image/jpeg");
  }

  useEffect(() => {
    workerRef.current = new Worker(
      new URL("../../workers/detection.worker.ts", import.meta.url)
    );
    workerRef.current.postMessage({ msg: "init" });
    workerRef.current.onmessage = (event) => {
      switch (event.data.msg) {
        case "ready":
          startVideo();
          handleWork();
        case "detect":
          const bboxs = event.data.bboxs;
          bboxsRef.current = bboxs;
          if (!imageSentRef.current && bboxs && bboxs.length > 0) {
            imageSentRef.current = true;
            sendImage();
          }
          handleWork();
          break;
        case "info":
          console.log(event.data.info);
          break;
        case "error":
          console.error(event.data.error);
        default:
          console.log("Unknown message received from worker: ", event.data.msg);
      }
    };

    return () => {
      const worker = workerRef.current;
      if (!worker) {
        return;
      }
      worker.terminate();
    };
  }, [handleWork]);

  useEffect(() => {
    console.log(detectResult);
  }, [detectResult]);

  return (
    <div className={styles.container}>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Navbar />
      <main className={styles.main}>
        <h1>Test</h1>
        <div>
          <button onClick={sendImage}>Identify</button>
        </div>
        <div style={{ display: "flex", position: "relative" }}>
          <video
            ref={videoRef}
            width={680}
            height={680}
            onLoadedMetadata={handleLoadedMetadata}
          ></video>
          <canvas
            width={640}
            height={640}
            style={{
              position: "absolute",
              zIndex: 2,
            }}
            ref={canvasRef}
            id="canvasOutput"
          ></canvas>
        </div>
        {identifying && <div>Identifying...</div>}
        {detectResult && detectResult.length > 0 && (
          <div>
            <pre>{JSON.stringify(detectResult[0], null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  );
}
