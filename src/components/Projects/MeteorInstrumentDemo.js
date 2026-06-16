import React, { useEffect, useMemo, useState } from "react";
import { Container } from "react-bootstrap";
import { Link } from "react-router-dom";
import Particle from "../Particle";

const orbitDuration = 15000;

const scatterMeteorSpecs = [
  { id: "alpha", delay: 0, startX: 10, endX: 42, angle: -28 },
  { id: "bravo", delay: 0.34, startX: 72, endX: 48, angle: 26 },
  { id: "charlie", delay: 0.68, startX: 30, endX: 62, angle: -18 },
];

const starDots = [
  { x: 6, y: 16, size: 2, drift: 0.4, offset: 0.67 },
  { x: 13, y: 24, size: 4, drift: 0.8, offset: 0.05 },
  { x: 15, y: 44, size: 2, drift: 0.5, offset: 0.82 },
  { x: 22, y: 58, size: 3, drift: 0.5, offset: 0.2 },
  { x: 24, y: 13, size: 2, drift: 0.7, offset: 0.37 },
  { x: 28, y: 36, size: 5, drift: 0.7, offset: 0.42 },
  { x: 31, y: 82, size: 2, drift: 0.6, offset: 0.95 },
  { x: 36, y: 68, size: 3, drift: 0.4, offset: 0.73 },
  { x: 39, y: 18, size: 2, drift: 0.5, offset: 0.27 },
  { x: 45, y: 26, size: 4, drift: 0.9, offset: 0.16 },
  { x: 47, y: 74, size: 2, drift: 0.7, offset: 0.58 },
  { x: 54, y: 49, size: 5, drift: 0.6, offset: 0.55 },
  { x: 57, y: 31, size: 2, drift: 0.8, offset: 0.7 },
  { x: 62, y: 18, size: 3, drift: 0.7, offset: 0.86 },
  { x: 64, y: 82, size: 2, drift: 0.5, offset: 0.04 },
  { x: 70, y: 64, size: 4, drift: 0.5, offset: 0.31 },
  { x: 73, y: 43, size: 2, drift: 0.6, offset: 0.23 },
  { x: 82, y: 34, size: 5, drift: 0.8, offset: 0.64 },
  { x: 84, y: 54, size: 2, drift: 0.5, offset: 0.79 },
  { x: 88, y: 73, size: 3, drift: 0.4, offset: 0.9 },
  { x: 91, y: 21, size: 2, drift: 0.7, offset: 0.33 },
  { x: 10, y: 70, size: 1, drift: 0.4, offset: 0.5 },
  { x: 17, y: 78, size: 2, drift: 0.6, offset: 0.48 },
  { x: 19, y: 32, size: 1, drift: 0.3, offset: 0.11 },
  { x: 33, y: 50, size: 1, drift: 0.4, offset: 0.61 },
  { x: 41, y: 42, size: 2, drift: 0.5, offset: 0.89 },
  { x: 50, y: 15, size: 1, drift: 0.3, offset: 0.44 },
  { x: 59, y: 68, size: 1, drift: 0.4, offset: 0.17 },
  { x: 67, y: 28, size: 1, drift: 0.3, offset: 0.76 },
  { x: 77, y: 14, size: 2, drift: 0.9, offset: 0.12 },
  { x: 79, y: 84, size: 1, drift: 0.5, offset: 0.52 },
  { x: 86, y: 12, size: 1, drift: 0.4, offset: 0.29 },
  { x: 93, y: 63, size: 1, drift: 0.5, offset: 0.98 },
];

const starLinks = [
  { left: 28, top: 37, width: 84, angle: 22 },
  { left: 45, top: 28, width: 78, angle: 42 },
  { left: 54, top: 51, width: 74, angle: -23 },
  { left: 70, top: 65, width: 58, angle: -48 },
];

const imagePalette = {
  ocean: "#211225",
  coast: "#4c2449",
  land: "#86533f",
  cloud: "#fffaf1",
  shadow: "#120a10",
  storm: "#d85cff",
};

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function getSignalStrength(progress) {
  return clamp(1 - Math.abs(progress - 0.5) / 0.2);
}

function getImageLevel(progress) {
  if (progress < 0.42) {
    return 0;
  }

  if (progress < 0.72) {
    return clamp((progress - 0.42) / 0.3);
  }

  if (progress > 0.96) {
    return clamp(1 - (progress - 0.96) / 0.04);
  }

  return 1;
}

function wrap01(value) {
  return ((value % 1) + 1) % 1;
}

function getTelemetry(progress) {
  const wave = Math.PI * 2 * progress;
  const temperature = 66 + 8 * Math.sin(wave + 0.4) + 2 * Math.sin(wave * 3);
  const humidity = 52 + 18 * Math.sin(wave + 2.2);
  const pressure = 1011 + 7 * Math.sin(wave + 4.1);

  return [
    {
      label: "TEMP",
      value: Math.round(temperature),
      unit: "F",
      level: clamp((temperature - 45) / 45) * 100,
    },
    {
      label: "HUM",
      value: Math.round(humidity),
      unit: "%",
      level: clamp(humidity / 100) * 100,
    },
    {
      label: "PRES",
      value: pressure.toFixed(1),
      unit: "hPa",
      level: clamp((pressure - 995) / 32) * 100,
    },
  ];
}

function getScatterFlight(progress, spec) {
  const phase = wrap01(progress * 2.35 + spec.delay);
  const travel = clamp(phase / 0.76);
  const burn = clamp(1 - Math.abs(phase - 0.62) / 0.16);
  const fade = phase > 0.8 ? clamp(1 - (phase - 0.8) / 0.2) : 1;

  return {
    ...spec,
    left: spec.startX + (spec.endX - spec.startX) * travel,
    top: 8 + 64 * travel,
    burn,
    opacity: fade * (phase < 0.9 ? 1 : 0),
  };
}

const meteorPixels = Array.from({ length: 96 }, (_, index) => {
  const row = Math.floor(index / 12);
  const col = index % 12;
  const diagonal = row + col;

  if (row < 2 && col > 6) {
    return imagePalette.cloud;
  }

  if ((row === 2 || row === 3) && col > 4 && col < 10) {
    return index % 2 ? imagePalette.cloud : imagePalette.storm;
  }

  if (diagonal > 7 && diagonal < 14) {
    return imagePalette.land;
  }

  if (row > 4 && col < 5) {
    return imagePalette.coast;
  }

  if (row === 7 && col > 7) {
    return imagePalette.shadow;
  }

  return index % 5 === 0 ? imagePalette.coast : imagePalette.ocean;
});

const scatterCells = Array.from({ length: 48 }, (_, index) => index);

function MeteorInstrumentDemo() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let animationFrame;
    const startTime = window.performance.now();

    function tick(now) {
      setProgress(((now - startTime) % orbitDuration) / orbitDuration);
      animationFrame = window.requestAnimationFrame(tick);
    }

    animationFrame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationFrame);
  }, []);

  const signal = getSignalStrength(progress);
  const imageLevel = getImageLevel(progress);
  const angle = Math.PI * progress;
  const satelliteStyle = useMemo(
    () => ({
      left: `${50 - 44 * Math.cos(angle)}%`,
      top: `${82 - 58 * Math.sin(angle)}%`,
    }),
    [angle]
  );
  const revealedPixels = Math.floor(imageLevel * meteorPixels.length);
  const telemetry = getTelemetry(progress);
  const scatterFlights = scatterMeteorSpecs.map((spec) => getScatterFlight(progress, spec));
  const scatterStrength = Math.max(...scatterFlights.map((meteor) => meteor.burn));
  const scatterCellCount = Math.floor(scatterStrength * scatterCells.length);
  const scatterActive = scatterStrength > 0.18;
  const starScan = wrap01(progress * 1.5);
  const starTargetStyle = {
    left: `${50 + 24 * Math.sin(progress * Math.PI * 2)}%`,
    top: `${47 + 18 * Math.cos(progress * Math.PI * 2)}%`,
  };

  return (
    <Container fluid className="project-section meteor-demo-section">
      <Particle />
      <Container className="project-shell meteor-demo-shell">
        <Link className="learning-back-link meteor-demo-back" to="/projects">
          Back to Projects
        </Link>

        <section className="section-intro reveal-up">
          <p className="section-kicker">Prototype</p>
          <h1 className="project-heading">
            METEOR <strong className="brown">INSTRUMENT DEMO</strong>
          </h1>
          <p className="section-lede">
            Animated demo of a custom embedded instrument unit for the METEOR M2 satellites, 
            showing real-time signal strength and image data as the satellite orbits overhead.
            Included are atmospheric sensor readings, a meteor scatter receiver demo, and an animated star map.
            This instrument also serves as a blueprint for future embedded units designed for
            scientific data collection and education.
          </p>
        </section>

        <section
          className={`meteor-demo-stage ${signal > 0.05 ? "receiving" : ""}`}
          aria-label="Animated Meteor instrument signal flow demo"
          style={{
            "--meteor-signal": signal,
            "--meteor-image-level": imageLevel,
          }}
        >
          <div className="meteor-orbit-arc" aria-hidden="true" />
          <div className="meteor-earth" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>

          <div className="meteor-satellite" style={satelliteStyle}>
            <span>M2</span>
            <i aria-hidden="true" />
          </div>

          <div className="meteor-signal-beam" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>

          <div className="meteor-ground-array" aria-label="Ground antenna">
            <span className="meteor-antenna-dish" />
            <span className="meteor-antenna-mast" />
            <span className="meteor-antenna-base" />
          </div>

          <div className="meteor-data-bus" aria-hidden="true">
            <span style={{ width: `${Math.round(signal * 100)}%` }} />
          </div>

          <div className="meteor-instrument-unit" aria-label="Embedded receiver display unit">
            <div className="meteor-unit-top">
              <span>RX</span>
              <span>{signal > 0.05 ? "LOCK" : "SCAN"}</span>
            </div>
            <div className="meteor-unit-screen">
              {meteorPixels.map((color, index) => (
                <span
                  key={`meteor-pixel-${index}`}
                  style={{
                    backgroundColor: color,
                    opacity: index < revealedPixels ? 1 : 0.12,
                  }}
                />
              ))}
            </div>
            <div className="meteor-unit-readout">
              <span>{String(Math.round(signal * 99)).padStart(2, "0")} db</span>
              <span>{String(Math.round(imageLevel * 100)).padStart(3, "0")}% img</span>
            </div>
          </div>
        </section>

        <section className="meteor-feature-grid" aria-label="Additional instrument functions">
          <article className="meteor-feature-panel meteor-weather-panel">
            <div className="meteor-panel-top">
              <span>ENV SENSOR</span>
              <span>LIVE</span>
            </div>
            <h2>Atmospheric Readings</h2>
            <div className="meteor-telemetry-stack">
              {telemetry.map((metric) => (
                <div className="meteor-telemetry-row" key={metric.label}>
                  <div className="meteor-telemetry-copy">
                    <span>{metric.label}</span>
                    <strong>
                      {metric.value}
                      {metric.unit}
                    </strong>
                  </div>
                  <div className="meteor-telemetry-bar" aria-hidden="true">
                    <span style={{ width: `${metric.level}%` }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="meteor-panel-footer">
              <span>sample {String(Math.floor(progress * 999)).padStart(3, "0")}</span>
              <span>baro sync</span>
            </div>
          </article>

          <article className="meteor-feature-panel meteor-scatter-panel">
            <div className="meteor-panel-top">
              <span>SCATTER RX</span>
              <span>{scatterActive ? "STRIKE" : "SCAN"}</span>
            </div>
            <h2>Meteor Scatter Receiver</h2>
            <div className="meteor-scatter-sky" aria-hidden="true">
              <div className="meteor-scatter-atmosphere" />
              {scatterFlights.map((meteor) => (
                <span
                  className={`scatter-meteor ${meteor.burn > 0.18 ? "burning" : ""}`}
                  key={meteor.id}
                  style={{
                    "--burn": meteor.burn,
                    left: `${meteor.left}%`,
                    opacity: meteor.opacity,
                    top: `${meteor.top}%`,
                    transform: `rotate(${meteor.angle}deg) scale(${0.75 + meteor.burn * 0.5})`,
                  }}
                >
                  <i />
                </span>
              ))}
            </div>
            <div className="meteor-scatter-screen">
              <div className="meteor-scatter-grid" aria-hidden="true">
                {scatterCells.map((cell) => (
                  <span
                    key={`scatter-cell-${cell}`}
                    style={{ opacity: cell < scatterCellCount ? 1 : 0.12 }}
                  />
                ))}
              </div>
              <div className="meteor-panel-footer">
                <span>{scatterActive ? "strike detected" : "listening"}</span>
                <span>{String(Math.round(scatterStrength * 99)).padStart(2, "0")} db</span>
              </div>
            </div>
          </article>

          <article className="meteor-feature-panel meteor-star-panel">
            <div className="meteor-panel-top">
              <span>STAR MAP</span>
              <span>TRACK</span>
            </div>
            <h2>Animated Star Map</h2>
            <div className="meteor-star-map" aria-hidden="true">
              <span className="meteor-star-sweep" style={{ left: `${starScan * 100}%` }} />
              {starLinks.map((link, index) => (
                <span
                  className="meteor-star-link"
                  key={`star-link-${index}`}
                  style={{
                    left: `${link.left}%`,
                    top: `${link.top}%`,
                    transform: `rotate(${link.angle}deg)`,
                    width: `${link.width}px`,
                  }}
                />
              ))}
              {starDots.map((star, index) => {
                const pulse = 0.45 + 0.55 * clamp(Math.sin((progress + star.offset) * Math.PI * 2) * 0.5 + 0.5);

                return (
                  <span
                    className="meteor-star-dot"
                    key={`star-dot-${index}`}
                    style={{
                      height: `${star.size}px`,
                      left: `${star.x + Math.sin(progress * Math.PI * 2 + star.offset) * star.drift}%`,
                      opacity: pulse,
                      top: `${star.y + Math.cos(progress * Math.PI * 2 + star.offset) * star.drift}%`,
                      width: `${star.size}px`,
                    }}
                  />
                );
              })}
              <span className="meteor-star-target" style={starTargetStyle} />
            </div>
            <div className="meteor-panel-footer">
              <span>az {String(Math.round(starScan * 359)).padStart(3, "0")}</span>
              <span>sky lock</span>
            </div>
          </article>
        </section>
      </Container>
    </Container>
  );
}

export default MeteorInstrumentDemo;
