import React, { useCallback, useEffect, useRef, useState } from "react";
import { Container, Row, Col } from "react-bootstrap";
import { FaPause, FaPlay } from "react-icons/fa";
import { FiSkipBack, FiSkipForward } from "react-icons/fi";
import Particle from "../Particle";
import AmbientSynth from "./AmbientSynth";

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${remainingSeconds}`;
}

function getTrackUrl(track) {
  if (!track) {
    return "";
  }

  if (/^https?:\/\//.test(track.file)) {
    return track.file;
  }

  return `${process.env.PUBLIC_URL || ""}${track.file}`;
}

function computePeaks(audioBuffer, samples = 900) {
  const channelCount = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const blockSize = Math.max(1, Math.floor(length / samples));
  const peaks = [];

  for (let i = 0; i < samples; i += 1) {
    const start = i * blockSize;
    const end = Math.min(start + blockSize, length);
    let min = 0;
    let max = 0;

    for (let channel = 0; channel < channelCount; channel += 1) {
      const data = audioBuffer.getChannelData(channel);

      for (let sample = start; sample < end; sample += 1) {
        const value = data[sample] || 0;
        if (value < min) {
          min = value;
        }
        if (value > max) {
          max = value;
        }
      }
    }

    peaks.push({
      min: min / channelCount,
      max: max / channelCount,
    });
  }

  return peaks;
}

function getCanvasDimensions(canvas, minimumHeight) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width || canvas.clientWidth || 1));
  const height = Math.max(minimumHeight, Math.floor(rect.height || minimumHeight));
  const resized = canvas.width !== width || canvas.height !== height;

  if (resized) {
    canvas.width = width;
    canvas.height = height;
  }

  return { width, height, resized };
}

function paintSpectrogramBackground(canvas) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext("2d");
  const { width, height } = getCanvasDimensions(canvas, 150);

  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgba(0, 0, 0, 0.24)";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(216, 92, 255, 0.11)";
  context.lineWidth = 1;

  for (let i = 1; i < 4; i += 1) {
    const y = (height / 4) * i;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function getSpectrogramColor(value) {
  const normalized = Math.min(1, Math.max(0, value / 255));
  const heat = Math.pow(normalized, 1.35);
  const flare = Math.pow(normalized, 3);

  return [
    Math.round(14 + 198 * heat + 36 * flare),
    Math.round(10 + 58 * heat + 158 * flare),
    Math.round(24 + 216 * heat),
    255,
  ];
}

const danceGifs = [
  {
    src: "https://nukochannel.neocities.org/NukoImg/Activities/Dance/nukoJumpDance.gif",
    alt: "Nuko jump dance",
  },
  {
    src: "https://nukochannel.neocities.org/NukoImg/Activities/Dance/nukoTapFeet.gif",
    alt: "Nuko tap feet",
  },
  {
    src: "https://nukochannel.neocities.org/NukoImg/Activities/Dance/nukoWiggleDance.gif",
    alt: "Nuko wiggle dance",
  },
  {
    src: "https://nukochannel.neocities.org/NukoImg/Activities/Dance/nukoHappyDance.gif",
    alt: "Nuko happy dance",
  },
  {
    src: "https://nukochannel.neocities.org/NukoImg/Activities/Dance/nukoBreakdance.gif",
    alt: "Nuko breakdance",
  },
  {
    src: "https://nukochannel.neocities.org/NukoImg/Activities/Dance/nukoHeadSwayDance.gif",
    alt: "Nuko head sway dance",
  },
];

function Music() {
  const [tracks, setTracks] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [peaks, setPeaks] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingWaveform, setIsLoadingWaveform] = useState(false);
  const [error, setError] = useState("");
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const spectrogramRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);

  const activeTrack = tracks[activeIndex];
  const activeTrackUrl = getTrackUrl(activeTrack);

  const drawWaveform = useCallback(
    (progress = 0) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const context = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(180, Math.floor(rect.height));
      const centerY = height / 2;

      canvas.width = width * ratio;
      canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      context.fillStyle = "rgba(216, 92, 255, 0.08)";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "rgba(216, 92, 255, 0.18)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(0, centerY);
      context.lineTo(width, centerY);
      context.stroke();

      if (!peaks.length) {
        context.fillStyle = "rgba(255, 250, 241, 0.55)";
        context.font = "600 15px Inter, Segoe UI, sans-serif";
        context.textAlign = "center";
        context.fillText("No waveform loaded.", width / 2, centerY);
        return;
      }

      const barWidth = Math.max(1, width / peaks.length);
      const playedX = width * progress;

      peaks.forEach((peak, index) => {
        const x = index * barWidth;
        const amp = Math.max(Math.abs(peak.min), Math.abs(peak.max));
        const barHeight = Math.max(2, amp * height * 0.9);
        const y = centerY - barHeight / 2;
        context.fillStyle = x <= playedX ? "#d85cff" : "rgba(255, 250, 241, 0.42)";
        context.fillRect(x, y, Math.max(1, barWidth * 0.75), barHeight);
      });

      context.fillStyle = "rgba(216, 92, 255, 0.28)";
      context.fillRect(0, 0, playedX, height);

      context.strokeStyle = "#d85cff";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(playedX, 0);
      context.lineTo(playedX, height);
      context.stroke();
    },
    [peaks]
  );

  const drawSpectrogramFrame = useCallback(() => {
    const analyser = analyserRef.current;
    const canvas = spectrogramRef.current;

    if (!analyser || !canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    const { width, height, resized } = getCanvasDimensions(canvas, 150);

    if (resized) {
      paintSpectrogramBackground(canvas);
    }

    const frequencyData = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(frequencyData);

    if (width > 1) {
      context.drawImage(canvas, 1, 0, width - 1, height, 0, 0, width - 1, height);
    }

    const column = context.createImageData(1, height);

    for (let y = 0; y < height; y += 1) {
      const topToBottom = 1 - y / Math.max(1, height - 1);
      const binIndex = Math.min(
        frequencyData.length - 1,
        Math.floor(Math.pow(topToBottom, 1.7) * frequencyData.length)
      );
      const [red, green, blue, alpha] = getSpectrogramColor(frequencyData[binIndex]);
      const pixel = y * 4;

      column.data[pixel] = red;
      column.data[pixel + 1] = green;
      column.data[pixel + 2] = blue;
      column.data[pixel + 3] = alpha;
    }

    context.putImageData(column, width - 1, 0);
  }, []);

  const setupAudioAnalyzer = useCallback(async () => {
    const audio = audioRef.current;
    const AudioContext = window.AudioContext || window.webkitAudioContext;

    if (!audio || !AudioContext) {
      setError("Spectrogram rendering is not supported in this browser.");
      return null;
    }

    if (!audioContextRef.current || audioContextRef.current.state === "closed") {
      audioContextRef.current = new AudioContext();
    }

    const audioContext = audioContextRef.current;

    if (!sourceRef.current) {
      sourceRef.current = audioContext.createMediaElementSource(audio);
    }

    if (!analyserRef.current) {
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.78;
      sourceRef.current.connect(analyser);
      analyser.connect(audioContext.destination);
      analyserRef.current = analyser;
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    return analyserRef.current;
  }, []);

  useEffect(() => {
    let isMounted = true;

    fetch(`${process.env.PUBLIC_URL || ""}/music/tracks.json`, {
      cache: "no-store",
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Music manifest unavailable.");
        }
        return response.json();
      })
      .then((manifestTracks) => {
        if (!isMounted) {
          return;
        }
        setTracks(Array.isArray(manifestTracks) ? manifestTracks : []);
      })
      .catch(() => {
        if (isMounted) {
          setTracks([]);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!activeTrackUrl) {
      setPeaks([]);
      return;
    }

    let isMounted = true;
    const AudioContext = window.AudioContext || window.webkitAudioContext;

    if (!AudioContext) {
      setError("Waveform rendering is not supported in this browser.");
      return;
    }

    setIsLoadingWaveform(true);
    setError("");

    fetch(activeTrackUrl)
      .then((response) => response.arrayBuffer())
      .then(async (arrayBuffer) => {
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        await audioContext.close();
        return computePeaks(audioBuffer);
      })
      .then((nextPeaks) => {
        if (isMounted) {
          setPeaks(nextPeaks);
          setIsLoadingWaveform(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setPeaks([]);
          setIsLoadingWaveform(false);
          setError("Could not decode this WAV file.");
        }
      });

    return () => {
      isMounted = false;
    };
  }, [activeTrackUrl]);

  useEffect(() => {
    const progress = duration ? currentTime / duration : 0;
    drawWaveform(progress);
  }, [currentTime, drawWaveform, duration, peaks]);

  useEffect(() => {
    function handleResize() {
      const progress = duration ? currentTime / duration : 0;
      drawWaveform(progress);
      paintSpectrogramBackground(spectrogramRef.current);
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [currentTime, drawWaveform, duration]);

  useEffect(() => {
    paintSpectrogramBackground(spectrogramRef.current);
  }, [activeTrackUrl]);

  useEffect(() => {
    paintSpectrogramBackground(spectrogramRef.current);

    return () => {
      window.cancelAnimationFrame(animationFrameRef.current);
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (!isPlaying) {
      window.cancelAnimationFrame(animationFrameRef.current);
      return undefined;
    }

    function tick() {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime || 0);
      }
      drawSpectrogramFrame();
      animationFrameRef.current = window.requestAnimationFrame(tick);
    }

    animationFrameRef.current = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationFrameRef.current);
  }, [drawSpectrogramFrame, isPlaying]);

  async function togglePlayback() {
    const audio = audioRef.current;
    if (!audio || !activeTrack) {
      return;
    }

    if (audio.paused) {
      const analyser = await setupAudioAnalyzer();

      if (!analyser) {
        return;
      }

      audio.play().catch(() => {
        setError("Could not start playback.");
      });
    } else {
      audio.pause();
    }
  }

  function selectTrack(index) {
    setActiveIndex(index);
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    paintSpectrogramBackground(spectrogramRef.current);
  }

  function moveTrack(direction) {
    if (!tracks.length) {
      return;
    }

    const nextIndex = (activeIndex + direction + tracks.length) % tracks.length;
    selectTrack(nextIndex);
  }

  function seekWaveform(event) {
    const audio = audioRef.current;
    const canvas = canvasRef.current;
    if (!audio || !canvas || !duration) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
    const nextTime = (x / rect.width) * duration;
    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  }

  return (
    <Container fluid className="project-section music-section">
      <Particle />
      <Container className="project-shell music-shell">
        <section className="section-intro reveal-up">
          <p className="section-kicker">Music</p>
        </section>

        <AmbientSynth />

        <div className="music-dance-strip reveal-up delay-1" aria-label="Dancing Nuko animations">
          {danceGifs.map((gif) => (
            <span key={gif.src} className="music-dance-gif">
              <img src={gif.src} alt={gif.alt} loading="lazy" />
            </span>
          ))}
        </div>

        <Row className="music-player-grid">
          <Col lg={8}>
            <article className="music-player-panel">
              <div className="music-player-header">
                <div>
                  <p className="section-kicker">Now Playing</p>
                  <h2>{activeTrack ? activeTrack.title : "No tracks yet"}</h2>
                </div>
                <div className="music-time">
                  <span>{formatTime(currentTime)}</span>
                  <span>{formatTime(duration)}</span>
                </div>
              </div>

              <canvas
                ref={canvasRef}
                className="music-waveform"
                onClick={seekWaveform}
                aria-label="Audio waveform"
              />

              <div className="music-visual-label">
                <span>Spectrogram</span>
              </div>
              <canvas
                ref={spectrogramRef}
                className="music-spectrogram"
                aria-label="Live audio spectrogram"
              />

              <div className="music-controls">
                <button type="button" onClick={() => moveTrack(-1)} disabled={!tracks.length}>
                  <FiSkipBack />
                </button>
                <button
                  type="button"
                  className="music-play-button"
                  onClick={togglePlayback}
                  disabled={!activeTrack}
                >
                  {isPlaying ? <FaPause /> : <FaPlay />}
                </button>
                <button type="button" onClick={() => moveTrack(1)} disabled={!tracks.length}>
                  <FiSkipForward />
                </button>
              </div>

              {isLoadingWaveform && <p className="music-status">Loading waveform...</p>}
              {error && <p className="music-status">{error}</p>}

              <audio
                ref={audioRef}
                src={activeTrackUrl}
                preload="metadata"
                onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime || 0)}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => {
                  setIsPlaying(false);
                  moveTrack(1);
                }}
              />
            </article>
          </Col>

          <Col lg={4}>
            <aside className="music-track-list">
              <h2>Track List</h2>
              {tracks.length > 0 ? (
                tracks.map((track, index) => (
                  <button
                    type="button"
                    key={track.file}
                    className={index === activeIndex ? "active" : ""}
                    onClick={() => selectTrack(index)}
                  >
                    <span>{track.title}</span>
                    <small>{track.file.replace("/music/", "")}</small>
                  </button>
                ))
              ) : (
                <p>No WAV tracks available yet.</p>
              )}
            </aside>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default Music;
