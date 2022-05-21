import type { NextPage } from "next";
import * as ort from "onnxruntime-web";
import { useEffect, useState } from "react";

import styles from "../styles/Home.module.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

const Page: NextPage = () => {
  const [session, setSession] = useState<any>();
  const [status, setStatus] = useState("");

  const initModel = async () => {
    try {
      const _session = await ort.InferenceSession.create(
        `${API_URL}/models/buffalo_l/w600k_r50.onnx`,
        { logSeverityLevel: 1 }
      );
      setSession(_session);
    } catch (error) {
      console.error(error);
    }
  };

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
      <main className={styles.main}>
        <h2>{status}</h2>
        {session && (
          <div>
            <pre>{JSON.stringify(session, null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  );
};

export default Page;
