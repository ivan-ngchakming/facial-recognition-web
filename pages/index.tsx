import type { NextPage } from "next";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import styles from "../styles/Home.module.css";
import axios, { AxiosError } from "axios";
import Navbar from "../components/Navbar";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

const Home: NextPage = () => {
  const [url, setUrl] = useState(
    "https://image.tmdb.org/t/p/original/wA1ZT3GSWvRjcJP96VRRARs9zEe.jpg"
  );
  const [file, setFile] = useState<any>();
  const [data, setData] = useState<any>();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<AxiosError<any, any> | null>();

  const handleUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    setIsLoading(true);
    setData(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file, file.name);

    try {
      const res: any = await axios.post(API_URL + `/photos/upload`, formData);
      setUrl(API_URL + res.data.url);
      handleSubmit();
    } catch (error) {
      console.error(error);
      if (error instanceof AxiosError) {
        setError(error);
        setIsLoading(false);
      }
    }
  };

  const handleSubmit = async (
    event: React.FormEvent<HTMLFormElement> | null = null
  ) => {
    if (event) event.preventDefault();

    setIsLoading(true);
    setData(null);
    setError(null);

    try {
      const data = await axios.get(
        API_URL + `/faces/search?url=${encodeURIComponent(url)}`
      );
      setData(data.data);
    } catch (error) {
      console.error(error);
      if (error instanceof AxiosError) {
        setError(error);
      }
    }
  };

  useEffect(() => {
    if (data || error) {
      setIsLoading(false);
    }
  }, [data, error]);

  return (
    <div className={styles.container}>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <Navbar />

        <form onSubmit={handleUpload} style={{ margin: "32px 0" }}>
          <div>
            <label htmlFor="upload">Upload:</label>
            <input
              type="file"
              id="upload"
              onChange={(e: any) => setFile(e.target.files[0])}
            />
          </div>
          <button type="submit" disabled={!file || isLoading}>
            Upload
          </button>
        </form>

        <form onSubmit={handleSubmit} style={{ margin: "32px 0" }}>
          <label htmlFor="url">URL: </label>
          <input
            type="text"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button type="submit" disabled={isLoading}>
            Search
          </button>
        </form>

        <div style={{ marginBottom: 32 }}>
          <p>Face to search for: </p>
          <img src={url} alt="" style={{ width: 300, height: "auto" }} />
        </div>
        <p>Search Results: </p>
        {isLoading && <p>Loading...</p>}
        {error && <p>{error.message}</p>}
        {error?.response?.data?.error && <p>{error.response.data.error}</p>}
        {data && data.length === 0 && <p>No face found :(</p>}
        {data && (
          <div style={{ marginBottom: 32 }}>
            {data.map((targetFace: any, index: number) => (
              <div key={targetFace.id}>
                {data.length > 1 && <p>Result for face {index + 1}</p>}
                {targetFace.map(({ face, score }: any) => (
                  <div
                    style={{ display: "flex", margin: 16 }}
                    key={targetFace.id + face.id}
                  >
                    <img
                      style={{ height: "auto", width: 200 }}
                      alt=""
                      src={
                        face.photo.url.startsWith("/static/")
                          ? API_URL + face.photo.url
                          : face.photo.url
                      }
                    />
                    <div style={{ margin: 16 }}>
                      <h3>{face.profile.name}</h3>
                      <table>
                        <tr>
                          <td>
                            <b>id: </b>
                          </td>
                          <td>{face.id}</td>
                        </tr>
                        <tr>
                          <td>
                            <b>score: </b>
                          </td>
                          <td>{score}</td>
                        </tr>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          Powered by{" "}
          <span className={styles.logo}>
            <Image src="/vercel.svg" alt="Vercel Logo" width={72} height={16} />
          </span>
        </a>
      </footer>
    </div>
  );
};

export default Home;
