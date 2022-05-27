import '../styles/globals.css'
import type { AppProps } from 'next/app'

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-sync-scripts */}
      <script src='https://docs.opencv.org/4.5.5/opencv.js' />
      <Component {...pageProps} />
    </>
  )
}

export default MyApp
