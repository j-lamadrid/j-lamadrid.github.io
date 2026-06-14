import React, { useEffect, useRef, useState } from "react";
import { Container, Row, Col } from "react-bootstrap";
import { FaPause, FaPlay } from "react-icons/fa";
import { FiSkipBack, FiSkipForward } from "react-icons/fi";
import Particle from "../Particle";
import AmbientSynth from "./AmbientSynth";
import FavoriteAlbums from "./FavoriteAlbums";

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
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState("");
  const audioRef = useRef(null);

  const activeTrack = tracks[activeIndex];
  const activeTrackUrl = getTrackUrl(activeTrack);
  const activeTrackFile = activeTrack ? activeTrack.file.replace("/music/", "") : "no disk";
  const playbackProgress = duration ? Math.min(100, Math.max(0, (currentTime / duration) * 100)) : 0;

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

  async function togglePlayback() {
    const audio = audioRef.current;
    if (!audio || !activeTrack) {
      return;
    }

    if (audio.paused) {
      setError("");
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
  }

  function moveTrack(direction) {
    if (!tracks.length) {
      return;
    }

    const nextIndex = (activeIndex + direction + tracks.length) % tracks.length;
    selectTrack(nextIndex);
  }

  function seekProgress(event) {
    const audio = audioRef.current;
    if (!audio || !duration) {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
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

        <Row className="music-player-grid music-device-grid">
          <Col lg={7}>
            <article className="music-player-panel music-mp3-player music-device-shell">
              <span className="music-device-side music-device-side-left" aria-hidden="true" />
              <span className="music-device-side music-device-side-right" aria-hidden="true" />
              <div className="music-device-face">
                <div className="music-device-topbar" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>

                <div className="music-device-screen">
                  <div className="music-device-status-row">
                    <span>{isPlaying ? "PLAY" : "STOP"}</span>
                    <span>WAV</span>
                    <span>{tracks.length ? `${activeIndex + 1}/${tracks.length}` : "0/0"}</span>
                  </div>

                  <p className="section-kicker">Now Playing</p>
                  <h2>{activeTrack ? activeTrack.title : "No tracks loaded"}</h2>
                  <p className="music-device-file">{activeTrackFile}</p>

                  <button
                    type="button"
                    className="music-device-progress"
                    onClick={seekProgress}
                    disabled={!duration}
                    aria-label="Seek current song"
                  >
                    <span style={{ width: `${playbackProgress}%` }} />
                  </button>

                  <div className="music-device-timebar">
                    <span>{formatTime(currentTime)}</span>
                    <span>{formatTime(duration)}</span>
                  </div>

                  <div className="music-device-eq" aria-hidden="true">
                    {Array.from({ length: 12 }).map((_, index) => (
                      <span
                        key={`eq-${index}`}
                        style={{ animationDelay: `${index * 0.08}s` }}
                        className={isPlaying ? "active" : ""}
                      />
                    ))}
                  </div>
                </div>

                <div className="music-device-controls">
                  <button
                    type="button"
                    className="music-device-skip"
                    onClick={() => moveTrack(-1)}
                    disabled={!tracks.length}
                    aria-label="Previous song"
                  >
                    <FiSkipBack />
                  </button>
                  <div className="music-device-wheel">
                    <button
                      type="button"
                      className="music-device-center"
                      onClick={togglePlayback}
                      disabled={!activeTrack}
                      aria-label={isPlaying ? "Pause current song" : "Play current song"}
                    >
                      {isPlaying ? <FaPause /> : <FaPlay />}
                    </button>
                  </div>
                  <button
                    type="button"
                    className="music-device-skip"
                    onClick={() => moveTrack(1)}
                    disabled={!tracks.length}
                    aria-label="Next song"
                  >
                    <FiSkipForward />
                  </button>
                </div>

                <div className="music-device-footer">
                  <span>local wav</span>
                </div>
              </div>

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

          <Col lg={5}>
            <aside className="music-track-list music-mp3-playlist music-device-playlist">
              <div className="music-device-playlist-head">
                <span>My Music</span>
                <h2>Loaded Songs</h2>
              </div>
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

        <FavoriteAlbums />
      </Container>
    </Container>
  );
}

export default Music;
