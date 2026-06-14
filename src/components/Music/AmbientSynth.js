import React, { useEffect, useRef, useState } from "react";
import { FaPause, FaPlay } from "react-icons/fa";
import { FiChevronDown } from "react-icons/fi";

const defaultSettings = {
  rootNote: 40,
  spread: 12,
  brightness: 760,
  resonance: 4.2,
  drift: 0.08,
  motion: 320,
  delayTime: 0.62,
  feedback: 0.46,
  space: 0.52,
  noise: 0.14,
  subLevel: 0.24,
  drive: 0.28,
  chorus: 0.34,
  arpRate: 4,
  arpGate: 0.58,
  arpOctaves: 2,
  arpLevel: 0.34,
  tempo: 76,
  drumVolume: 0.28,
  kickPitch: 1,
  kickDecay: 0.46,
  kickLevel: 0.9,
  snareTone: 1850,
  snareDecay: 0.2,
  snareLevel: 0.42,
  hatTone: 6400,
  hatDecay: 0.07,
  hatLevel: 0.22,
  openHatTone: 5200,
  openHatDecay: 0.38,
  openHatLevel: 0.2,
  volume: 0.24,
};

const stepIndexes = Array.from({ length: 16 }, (_, index) => index);
const noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const pianoRollNotes = Array.from({ length: 24 }, (_, index) => 36 + index);

const defaultDrumPattern = {
  kick: stepIndexes.map(() => false),
  snare: stepIndexes.map(() => false),
  hat: stepIndexes.map(() => false),
  openHat: stepIndexes.map(() => false),
};

const drumTracks = [
  { key: "kick", label: "Kick" },
  { key: "snare", label: "Snare" },
  { key: "hat", label: "Hat" },
  { key: "openHat", label: "Open Hat" },
];

const arpPatterns = [
  { key: "up", label: "Up", intervals: [0, 4, 7, 12, 7, 4] },
  { key: "down", label: "Down", intervals: [12, 7, 4, 0, 4, 7] },
  { key: "rise", label: "Rise", intervals: [0, 7, 12, 16, 19, 16, 12, 7] },
  { key: "minor", label: "Minor", intervals: [0, 3, 7, 10, 12, 10, 7, 3] },
];

const drumShapeModules = [
  {
    title: "Kick",
    controls: [
      { key: "kickPitch", label: "Pitch", min: 0.72, max: 1.38, step: 0.01, unit: "" },
      { key: "kickDecay", label: "Decay", min: 0.16, max: 0.82, step: 0.01, unit: "s" },
      { key: "kickLevel", label: "Level", min: 0, max: 1.2, step: 0.01, unit: "" },
    ],
  },
  {
    title: "Snare",
    controls: [
      { key: "snareTone", label: "Tone", min: 900, max: 4200, step: 25, unit: "Hz" },
      { key: "snareDecay", label: "Decay", min: 0.08, max: 0.46, step: 0.01, unit: "s" },
      { key: "snareLevel", label: "Level", min: 0, max: 0.9, step: 0.01, unit: "" },
    ],
  },
  {
    title: "Closed Hat",
    controls: [
      { key: "hatTone", label: "Tone", min: 3600, max: 9200, step: 50, unit: "Hz" },
      { key: "hatDecay", label: "Decay", min: 0.03, max: 0.18, step: 0.01, unit: "s" },
      { key: "hatLevel", label: "Level", min: 0, max: 0.6, step: 0.01, unit: "" },
    ],
  },
  {
    title: "Open Hat",
    controls: [
      { key: "openHatTone", label: "Tone", min: 3000, max: 8600, step: 50, unit: "Hz" },
      { key: "openHatDecay", label: "Decay", min: 0.12, max: 0.8, step: 0.01, unit: "s" },
      { key: "openHatLevel", label: "Level", min: 0, max: 0.6, step: 0.01, unit: "" },
    ],
  },
];

const synthModules = [
  {
    title: "Voice",
    controls: [
      { key: "spread", label: "Spread", min: 0, max: 28, step: 1, unit: "ct" },
      { key: "subLevel", label: "Sub", min: 0, max: 0.5, step: 0.01, unit: "" },
      { key: "noise", label: "Noise", min: 0, max: 0.34, step: 0.01, unit: "" },
    ],
  },
  {
    title: "Arp",
    controls: [
      { key: "arpRate", label: "Rate", min: 1, max: 8, step: 1, unit: "x" },
      { key: "arpGate", label: "Gate", min: 0.18, max: 0.9, step: 0.01, unit: "" },
      { key: "arpOctaves", label: "Octaves", min: 1, max: 3, step: 1, unit: "oct" },
      { key: "arpLevel", label: "Level", min: 0, max: 0.55, step: 0.01, unit: "" },
    ],
  },
  {
    title: "Motion",
    controls: [
      { key: "drift", label: "Drift", min: 0.02, max: 0.22, step: 0.01, unit: "Hz" },
      { key: "motion", label: "Sweep", min: 0, max: 620, step: 10, unit: "Hz" },
    ],
  },
  {
    title: "Tone",
    controls: [
      { key: "brightness", label: "Cutoff", min: 320, max: 2200, step: 10, unit: "Hz" },
      { key: "resonance", label: "Resonance", min: 0.4, max: 10, step: 0.1, unit: "Q" },
      { key: "drive", label: "Drive", min: 0, max: 0.72, step: 0.01, unit: "" },
      { key: "chorus", label: "Chorus", min: 0, max: 0.72, step: 0.01, unit: "" },
    ],
  },
  {
    title: "Space",
    controls: [
      { key: "delayTime", label: "Echo", min: 0.12, max: 0.9, step: 0.01, unit: "s" },
      { key: "feedback", label: "Repeat", min: 0.08, max: 0.68, step: 0.01, unit: "" },
      { key: "space", label: "Wash", min: 0, max: 0.75, step: 0.01, unit: "" },
    ],
  },
];

function formatControlValue(value, unit) {
  if (unit === "Hz" && value >= 1) {
    return `${Math.round(value)} Hz`;
  }

  if (unit === "ct") {
    return `${Math.round(value)} ct`;
  }

  if (unit === "s") {
    return `${value.toFixed(2)} s`;
  }

  if (unit === "Q") {
    return value.toFixed(1);
  }

  if (unit === "bpm") {
    return `${Math.round(value)} bpm`;
  }

  if (unit === "x") {
    return `${Math.round(value)}x`;
  }

  if (unit === "oct") {
    return `${Math.round(value)} oct`;
  }

  return value.toFixed(2);
}

function midiToFrequency(note) {
  return 440 * Math.pow(2, (Number(note) - 69) / 12);
}

function midiToNoteName(note) {
  const midi = Number(note);
  const name = noteNames[((midi % 12) + 12) % 12];
  const octave = Math.floor(midi / 12) - 1;
  return `${name}${octave}`;
}

function isSharpNote(note) {
  return midiToNoteName(note).includes("#");
}

function getRootFrequency(settings) {
  return midiToFrequency(settings.rootNote);
}

function setParam(param, value, context, timeConstant = 0.08) {
  param.cancelScheduledValues(context.currentTime);
  param.setTargetAtTime(value, context.currentTime, timeConstant);
}

function createDriveCurve(amount) {
  const samples = 2048;
  const curve = new Float32Array(samples);
  const drive = 1 + amount * 36;

  for (let index = 0; index < samples; index += 1) {
    const x = (index * 2) / samples - 1;
    curve[index] = Math.tanh(x * drive) / Math.tanh(drive);
  }

  return curve;
}

function createReverbImpulse(context) {
  const seconds = 5;
  const decay = 3.2;
  const length = context.sampleRate * seconds;
  const impulse = context.createBuffer(2, length, context.sampleRate);

  for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
    const data = impulse.getChannelData(channel);

    for (let index = 0; index < length; index += 1) {
      const envelope = Math.pow(1 - index / length, decay);
      data[index] = (Math.random() * 2 - 1) * envelope;
    }
  }

  return impulse;
}

function createNoiseBuffer(context) {
  const seconds = 3;
  const length = context.sampleRate * seconds;
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const data = buffer.getChannelData(0);
  let last = 0;

  for (let index = 0; index < length; index += 1) {
    const white = Math.random() * 2 - 1;
    last = (last + 0.02 * white) / 1.02;
    data[index] = last * 3.2;
  }

  return buffer;
}

function createWhiteNoiseBuffer(context, seconds) {
  const length = Math.max(1, Math.floor(context.sampleRate * seconds));
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const data = buffer.getChannelData(0);

  for (let index = 0; index < length; index += 1) {
    data[index] = Math.random() * 2 - 1;
  }

  return buffer;
}

function triggerArpNote(graph, time, settings, pattern, step) {
  const { context, arpGain } = graph;
  const rootNote = Number(settings.rootNote);
  const baseFrequency = getRootFrequency(settings);
  const octaveSpan = Number(settings.arpOctaves);
  const patternStep = pattern.intervals[step % pattern.intervals.length];
  const octaveOffset = Math.floor(step / pattern.intervals.length) % octaveSpan;
  const interval = patternStep + octaveOffset * 12;
  const midiNote = rootNote + interval;
  const frequency = Math.max(24, baseFrequency * Math.pow(2, interval / 12));
  const stepDuration = getArpInterval(settings);
  const duration = Math.max(0.035, stepDuration * Number(settings.arpGate));
  const accent = step % 4 === 0 ? 1.18 : 0.88;
  const level = Math.max(0.0001, Number(settings.arpLevel) * accent);
  const saw = context.createOscillator();
  const sawWide = context.createOscillator();
  const pulse = context.createOscillator();
  const sub = context.createOscillator();
  const subGain = context.createGain();
  const gain = context.createGain();
  const tone = context.createBiquadFilter();
  const panner = context.createStereoPanner ? context.createStereoPanner() : null;

  saw.type = "sawtooth";
  saw.frequency.setValueAtTime(frequency, time);
  saw.detune.setValueAtTime(-Number(settings.spread) * 0.65, time);

  sawWide.type = "sawtooth";
  sawWide.frequency.setValueAtTime(frequency, time);
  sawWide.detune.setValueAtTime(Number(settings.spread) * 0.68, time);

  pulse.type = "square";
  pulse.frequency.setValueAtTime(frequency, time);
  pulse.detune.setValueAtTime(Number(settings.spread) * 0.22, time);

  sub.type = "triangle";
  sub.frequency.setValueAtTime(Math.max(20, frequency / 2), time);
  subGain.gain.setValueAtTime(Math.max(0.0001, Number(settings.subLevel) * 0.32), time);

  tone.type = "lowpass";
  tone.frequency.setValueAtTime(Math.max(320, Number(settings.brightness) * 1.45), time);
  tone.Q.setValueAtTime(Math.max(0.5, Number(settings.resonance) * 0.65), time);

  gain.gain.setValueAtTime(0.0001, time);
  gain.gain.linearRampToValueAtTime(level, time + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + duration);

  saw.connect(tone);
  sawWide.connect(tone);
  pulse.connect(tone);
  sub.connect(subGain);
  subGain.connect(tone);
  tone.connect(gain);

  if (panner) {
    panner.pan.setValueAtTime(step % 2 === 0 ? -0.22 : 0.22, time);
    gain.connect(panner);
    panner.connect(arpGain);
  } else {
    gain.connect(arpGain);
  }

  saw.start(time);
  sawWide.start(time);
  pulse.start(time);
  sub.start(time);
  saw.stop(time + duration + 0.04);
  sawWide.stop(time + duration + 0.04);
  pulse.stop(time + duration + 0.04);
  sub.stop(time + duration + 0.04);

  return midiNote;
}

function triggerKick(graph, time, settings) {
  const { context, drumGain } = graph;
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  const tone = context.createBiquadFilter();
  const pitch = Number(settings.kickPitch);
  const decay = Number(settings.kickDecay);
  const level = Number(settings.kickLevel);

  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(148 * pitch, time);
  oscillator.frequency.exponentialRampToValueAtTime(46 * pitch, time + Math.min(0.22, decay * 0.48));

  tone.type = "lowpass";
  tone.frequency.setValueAtTime(520 * pitch, time);
  tone.frequency.exponentialRampToValueAtTime(140 * pitch, time + Math.min(0.28, decay * 0.56));

  gain.gain.setValueAtTime(0.0001, time);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, level), time + 0.006);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + decay);

  oscillator.connect(tone);
  tone.connect(gain);
  gain.connect(drumGain);
  oscillator.start(time);
  oscillator.stop(time + decay + 0.06);
}

function triggerSnare(graph, time, settings) {
  const { context, drumGain } = graph;
  const noise = context.createBufferSource();
  const noiseFilter = context.createBiquadFilter();
  const noiseGain = context.createGain();
  const body = context.createOscillator();
  const bodyGain = context.createGain();
  const tone = Number(settings.snareTone);
  const decay = Number(settings.snareDecay);
  const level = Number(settings.snareLevel);

  noise.buffer = createWhiteNoiseBuffer(context, decay + 0.04);
  noiseFilter.type = "bandpass";
  noiseFilter.frequency.value = tone;
  noiseFilter.Q.value = 0.7;
  noiseGain.gain.setValueAtTime(0.0001, time);
  noiseGain.gain.exponentialRampToValueAtTime(Math.max(0.0001, level), time + 0.006);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, time + decay);

  body.type = "triangle";
  body.frequency.setValueAtTime(Math.max(120, tone * 0.095), time);
  body.frequency.exponentialRampToValueAtTime(Math.max(92, tone * 0.072), time + Math.min(0.12, decay));
  bodyGain.gain.setValueAtTime(0.0001, time);
  bodyGain.gain.exponentialRampToValueAtTime(Math.max(0.0001, level * 0.54), time + 0.004);
  bodyGain.gain.exponentialRampToValueAtTime(0.0001, time + Math.max(0.1, decay * 0.78));

  noise.connect(noiseFilter);
  noiseFilter.connect(noiseGain);
  noiseGain.connect(drumGain);
  body.connect(bodyGain);
  bodyGain.connect(drumGain);

  noise.start(time);
  noise.stop(time + decay + 0.04);
  body.start(time);
  body.stop(time + Math.max(0.14, decay));
}

function triggerHat(graph, time, settings) {
  const { context, drumGain } = graph;
  const noise = context.createBufferSource();
  const highpass = context.createBiquadFilter();
  const gain = context.createGain();
  const tone = Number(settings.hatTone);
  const decay = Number(settings.hatDecay);
  const level = Number(settings.hatLevel);

  noise.buffer = createWhiteNoiseBuffer(context, decay + 0.02);
  highpass.type = "highpass";
  highpass.frequency.value = tone;
  highpass.Q.value = 0.8;

  gain.gain.setValueAtTime(0.0001, time);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, level), time + 0.003);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + decay);

  noise.connect(highpass);
  highpass.connect(gain);
  gain.connect(drumGain);
  noise.start(time);
  noise.stop(time + decay + 0.02);
}

function triggerOpenHat(graph, time, settings) {
  const { context, drumGain } = graph;
  const noise = context.createBufferSource();
  const highpass = context.createBiquadFilter();
  const gain = context.createGain();
  const tone = Number(settings.openHatTone);
  const decay = Number(settings.openHatDecay);
  const level = Number(settings.openHatLevel);

  noise.buffer = createWhiteNoiseBuffer(context, decay + 0.08);
  highpass.type = "highpass";
  highpass.frequency.value = tone;
  highpass.Q.value = 0.62;

  gain.gain.setValueAtTime(0.0001, time);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, level), time + 0.004);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + decay);

  noise.connect(highpass);
  highpass.connect(gain);
  gain.connect(drumGain);
  noise.start(time);
  noise.stop(time + decay + 0.08);
}

function scheduleDrumStep(graph, pattern, step, time, settings) {
  if (pattern.kick[step]) {
    triggerKick(graph, time, settings);
  }

  if (pattern.snare[step]) {
    triggerSnare(graph, time, settings);
  }

  if (pattern.hat[step]) {
    triggerHat(graph, time, settings);
  }

  if (pattern.openHat[step]) {
    triggerOpenHat(graph, time, settings);
  }
}

function getStepDuration(settings) {
  return 60 / Number(settings.tempo) / 4;
}

function getArpInterval(settings) {
  return 60 / Number(settings.tempo) / Number(settings.arpRate);
}

function stopAudioNode(node) {
  try {
    node.stop();
  } catch (error) {
    void error;
  }
}

function teardownGraph(graph) {
  if (!graph) {
    return;
  }

  graph.voices.forEach((voice) => stopAudioNode(voice.oscillator));
  stopAudioNode(graph.lfo);
  stopAudioNode(graph.chorusLfo);
  stopAudioNode(graph.noiseSource);

  if (graph.context.state !== "closed") {
    graph.context.close().catch((error) => {
      void error;
    });
  }
}

function applySettingsToGraph(graph, settings, isActive) {
  const { context } = graph;
  const baseFrequency = getRootFrequency(settings);
  const spread = Number(settings.spread);

  graph.voices.forEach((voice) => {
    setParam(voice.oscillator.frequency, baseFrequency * voice.ratio, context);
    setParam(voice.oscillator.detune, voice.detuneBase * spread, context);
    setParam(
      voice.gain.gain,
      voice.role === "sub" ? Number(settings.subLevel) : voice.baseGain,
      context
    );
  });

  graph.drive.curve = createDriveCurve(Number(settings.drive));
  setParam(graph.filter.frequency, Number(settings.brightness), context);
  setParam(graph.filter.Q, Number(settings.resonance), context);
  setParam(graph.lfo.frequency, Number(settings.drift), context);
  setParam(graph.lfoDepth.gain, Number(settings.motion), context);
  setParam(graph.chorusGain.gain, Number(settings.chorus), context);
  setParam(graph.delay.delayTime, Number(settings.delayTime), context);
  setParam(graph.feedback.gain, Number(settings.feedback), context);
  setParam(graph.reverbSend.gain, Number(settings.space), context);
  setParam(graph.noiseGain.gain, Number(settings.noise) * 0.22, context);
  setParam(graph.arpGain.gain, isActive ? 1 : 0, context, 0.06);
  setParam(graph.drumGain.gain, isActive ? Number(settings.drumVolume) : 0, context, 0.08);
  setParam(graph.masterGain.gain, isActive ? Number(settings.volume) : 0, context, 0.18);
}

function createSynthGraph(context) {
  const masterGain = context.createGain();
  const voiceBus = context.createGain();
  const drive = context.createWaveShaper();
  const filter = context.createBiquadFilter();
  const dryGain = context.createGain();
  const chorusDelay = context.createDelay(0.05);
  const chorusGain = context.createGain();
  const chorusLfo = context.createOscillator();
  const chorusDepth = context.createGain();
  const delay = context.createDelay(1.5);
  const feedback = context.createGain();
  const delayGain = context.createGain();
  const reverbSend = context.createGain();
  const convolver = context.createConvolver();
  const reverbGain = context.createGain();
  const noiseGain = context.createGain();
  const arpGain = context.createGain();
  const drumGain = context.createGain();
  const drumCompressor = context.createDynamicsCompressor();
  const lfo = context.createOscillator();
  const lfoDepth = context.createGain();
  const noiseSource = context.createBufferSource();

  masterGain.gain.value = 0;
  voiceBus.gain.value = 0.92;
  drive.curve = createDriveCurve(defaultSettings.drive);
  drive.oversample = "4x";
  arpGain.gain.value = 0;
  drumGain.gain.value = 0;
  filter.type = "lowpass";
  dryGain.gain.value = 0.72;
  chorusDelay.delayTime.value = 0.018;
  chorusGain.gain.value = defaultSettings.chorus;
  chorusDepth.gain.value = 0.006;
  delayGain.gain.value = 0.34;
  reverbGain.gain.value = 0.48;
  drumCompressor.threshold.value = -18;
  drumCompressor.knee.value = 18;
  drumCompressor.ratio.value = 3;
  drumCompressor.attack.value = 0.006;
  drumCompressor.release.value = 0.14;
  convolver.buffer = createReverbImpulse(context);

  const voiceSettings = [
    { type: "sawtooth", ratio: 1, gain: 0.13, pan: -0.42, detuneBase: -0.72 },
    { type: "sawtooth", ratio: 1, gain: 0.13, pan: 0.42, detuneBase: 0.72 },
    { type: "square", ratio: 1, gain: 0.07, pan: 0, detuneBase: 0.18 },
    { type: "triangle", ratio: 0.5, gain: defaultSettings.subLevel, pan: 0, detuneBase: 0, role: "sub" },
    { type: "sine", ratio: 1.5, gain: 0.05, pan: -0.12, detuneBase: -0.22 },
    { type: "sine", ratio: 2.003, gain: 0.035, pan: 0.12, detuneBase: 0.22 },
  ];

  const voices = voiceSettings.map((voiceSetting) => {
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const panner = context.createStereoPanner ? context.createStereoPanner() : null;

    oscillator.type = voiceSetting.type;
    gain.gain.value = voiceSetting.gain;

    oscillator.connect(gain);

    if (panner) {
      panner.pan.value = voiceSetting.pan;
      gain.connect(panner);
      panner.connect(voiceBus);
    } else {
      gain.connect(voiceBus);
    }

    oscillator.start();
    return {
      oscillator,
      gain,
      panner,
      ratio: voiceSetting.ratio,
      detuneBase: voiceSetting.detuneBase,
      baseGain: voiceSetting.gain,
      role: voiceSetting.role || "voice",
    };
  });

  noiseSource.buffer = createNoiseBuffer(context);
  noiseSource.loop = true;
  noiseSource.connect(noiseGain);
  noiseGain.connect(voiceBus);
  noiseSource.start();
  arpGain.connect(voiceBus);
  voiceBus.connect(drive);
  drive.connect(filter);

  lfo.type = "sine";
  lfo.connect(lfoDepth);
  lfoDepth.connect(filter.frequency);
  lfo.start();

  chorusLfo.type = "sine";
  chorusLfo.frequency.value = 0.33;
  chorusLfo.connect(chorusDepth);
  chorusDepth.connect(chorusDelay.delayTime);
  chorusLfo.start();

  filter.connect(dryGain);
  dryGain.connect(masterGain);
  filter.connect(chorusDelay);
  chorusDelay.connect(chorusGain);
  chorusGain.connect(masterGain);

  filter.connect(delay);
  delay.connect(delayGain);
  delayGain.connect(masterGain);
  delay.connect(feedback);
  feedback.connect(delay);

  filter.connect(reverbSend);
  reverbSend.connect(convolver);
  convolver.connect(reverbGain);
  reverbGain.connect(masterGain);
  masterGain.connect(context.destination);
  drumGain.connect(drumCompressor);
  drumCompressor.connect(context.destination);

  return {
    context,
    voices,
    voiceBus,
    drive,
    filter,
    chorusGain,
    chorusLfo,
    delay,
    feedback,
    reverbSend,
    noiseGain,
    arpGain,
    drumGain,
    masterGain,
    lfo,
    lfoDepth,
    noiseSource,
  };
}

function AmbientSynth() {
  const [settings, setSettings] = useState(defaultSettings);
  const [pattern, setPattern] = useState(defaultDrumPattern);
  const [currentStep, setCurrentStep] = useState(0);
  const [activeArpNote, setActiveArpNote] = useState(defaultSettings.rootNote);
  const [isArpEnabled, setIsArpEnabled] = useState(true);
  const [arpPatternKey, setArpPatternKey] = useState("up");
  const [isExpanded, setIsExpanded] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const graphRef = useRef(null);
  const stopTimeoutRef = useRef(null);
  const schedulerRef = useRef(null);
  const arpSchedulerRef = useRef(null);
  const nextStepTimeRef = useRef(0);
  const nextArpTimeRef = useRef(0);
  const sequencerStepRef = useRef(0);
  const arpStepRef = useRef(0);
  const stepTimeoutsRef = useRef([]);
  const arpTimeoutsRef = useRef([]);
  const settingsRef = useRef(defaultSettings);
  const patternRef = useRef(defaultDrumPattern);
  const arpEnabledRef = useRef(true);
  const arpPatternRef = useRef(arpPatterns[0]);

  useEffect(() => {
    settingsRef.current = settings;

    if (graphRef.current) {
      applySettingsToGraph(graphRef.current, settings, isRunning);
    }
  }, [settings, isRunning]);

  useEffect(() => {
    patternRef.current = pattern;
  }, [pattern]);

  useEffect(() => {
    arpEnabledRef.current = isArpEnabled;
  }, [isArpEnabled]);

  useEffect(() => {
    arpPatternRef.current =
      arpPatterns.find((patternOption) => patternOption.key === arpPatternKey) || arpPatterns[0];
  }, [arpPatternKey]);

  useEffect(() => {
    return () => {
      window.clearInterval(schedulerRef.current);
      window.clearInterval(arpSchedulerRef.current);
      stepTimeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
      arpTimeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
      stepTimeoutsRef.current = [];
      arpTimeoutsRef.current = [];
      window.clearTimeout(stopTimeoutRef.current);
      teardownGraph(graphRef.current);
      graphRef.current = null;
    };
  }, []);

  function updateSetting(key, value) {
    setSettings((currentSettings) => ({
      ...currentSettings,
      [key]: Number(value),
    }));
  }

  function clearStepTimeouts() {
    stepTimeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
    stepTimeoutsRef.current = [];
  }

  function clearArpTimeouts() {
    arpTimeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
    arpTimeoutsRef.current = [];
  }

  function stopSequencer() {
    window.clearInterval(schedulerRef.current);
    schedulerRef.current = null;
    clearStepTimeouts();
    sequencerStepRef.current = 0;
    nextStepTimeRef.current = 0;
    setCurrentStep(0);
  }

  function stopArpScheduler() {
    window.clearInterval(arpSchedulerRef.current);
    arpSchedulerRef.current = null;
    clearArpTimeouts();
    nextArpTimeRef.current = 0;
    arpStepRef.current = 0;
    setActiveArpNote(settingsRef.current.rootNote);
  }

  function runArpSchedulerTick() {
    const graph = graphRef.current;

    if (!graph || graph.context.state === "closed") {
      return;
    }

    const lookahead = 0.14;

    if (!arpEnabledRef.current) {
      nextArpTimeRef.current = graph.context.currentTime + getArpInterval(settingsRef.current);
      return;
    }

    while (nextArpTimeRef.current < graph.context.currentTime + lookahead) {
      const midiNote = triggerArpNote(
        graph,
        nextArpTimeRef.current,
        settingsRef.current,
        arpPatternRef.current,
        arpStepRef.current
      );

      const timeoutId = window.setTimeout(() => {
        setActiveArpNote(midiNote);
        arpTimeoutsRef.current = arpTimeoutsRef.current.filter((id) => id !== timeoutId);
      }, Math.max(0, (nextArpTimeRef.current - graph.context.currentTime) * 1000));
      arpTimeoutsRef.current.push(timeoutId);

      nextArpTimeRef.current += getArpInterval(settingsRef.current);
      arpStepRef.current += 1;
    }
  }

  function startArpScheduler() {
    const graph = graphRef.current;

    if (!graph) {
      return;
    }

    stopArpScheduler();
    nextArpTimeRef.current = graph.context.currentTime + 0.04;
    arpStepRef.current = 0;
    runArpSchedulerTick();
    arpSchedulerRef.current = window.setInterval(runArpSchedulerTick, 25);
  }

  function runSchedulerTick() {
    const graph = graphRef.current;

    if (!graph || graph.context.state === "closed") {
      return;
    }

    const lookahead = 0.12;

    while (nextStepTimeRef.current < graph.context.currentTime + lookahead) {
      const step = sequencerStepRef.current;
      const stepTime = nextStepTimeRef.current;

      scheduleDrumStep(graph, patternRef.current, step, stepTime, settingsRef.current);

      const timeoutId = window.setTimeout(() => {
        setCurrentStep(step);
        stepTimeoutsRef.current = stepTimeoutsRef.current.filter((id) => id !== timeoutId);
      }, Math.max(0, (stepTime - graph.context.currentTime) * 1000));
      stepTimeoutsRef.current.push(timeoutId);

      nextStepTimeRef.current += getStepDuration(settingsRef.current);
      sequencerStepRef.current = (sequencerStepRef.current + 1) % stepIndexes.length;
    }
  }

  function startSequencer() {
    const graph = graphRef.current;

    if (!graph) {
      return;
    }

    stopSequencer();
    nextStepTimeRef.current = graph.context.currentTime + 0.05;
    sequencerStepRef.current = 0;
    runSchedulerTick();
    schedulerRef.current = window.setInterval(runSchedulerTick, 25);
  }

  function toggleStep(trackKey, step) {
    setPattern((currentPattern) => ({
      ...currentPattern,
      [trackKey]: currentPattern[trackKey].map((isActive, index) =>
        index === step ? !isActive : isActive
      ),
    }));
  }

  function selectRootNote(note) {
    setSettings((currentSettings) => ({
      ...currentSettings,
      rootNote: note,
    }));

    if (!isRunning) {
      setActiveArpNote(note);
    }
  }

  async function startSynth() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;

    if (!AudioContext) {
      setError("This browser does not support the Web Audio synth.");
      return;
    }

    window.clearTimeout(stopTimeoutRef.current);

    if (!graphRef.current) {
      const context = new AudioContext();
      graphRef.current = createSynthGraph(context);
    }

    if (graphRef.current.context.state === "suspended") {
      await graphRef.current.context.resume();
    }

    setError("");
    applySettingsToGraph(graphRef.current, settings, true);
    startArpScheduler();
    startSequencer();
    setIsRunning(true);
  }

  function stopSynth() {
    const graph = graphRef.current;

    if (!graph) {
      setIsRunning(false);
      return;
    }

    applySettingsToGraph(graph, settings, false);
    stopArpScheduler();
    stopSequencer();
    setIsRunning(false);

    window.clearTimeout(stopTimeoutRef.current);
    stopTimeoutRef.current = window.setTimeout(() => {
      if (graphRef.current === graph) {
        teardownGraph(graph);
        graphRef.current = null;
      }
    }, 900);
  }

  function toggleSynth() {
    if (isRunning) {
      stopSynth();
    } else {
      startSynth();
    }
  }

  return (
    <article className={`ambient-synth-panel reveal-up ${isExpanded ? "expanded" : "collapsed"}`}>
      <div className="ambient-synth-head">
        <button
          type="button"
          className="ambient-synth-toggle"
          aria-expanded={isExpanded}
          aria-controls="ambient-synth-controls"
          onClick={() => setIsExpanded((expanded) => !expanded)}
        >
          <span>
            <p className="section-kicker">Synth + Sequencer</p>
            <strong>{isExpanded ? "Hide modular controls" : "Open modular synth"}</strong>
          </span>
          <FiChevronDown className="ambient-synth-chevron" aria-hidden="true" />
        </button>
        <button
          type="button"
          className={`ambient-synth-power ${isRunning ? "active" : ""}`}
          onClick={toggleSynth}
        >
          {isRunning ? <FaPause /> : <FaPlay />}
          <span>{isRunning ? "Stop" : "Start"}</span>
        </button>
      </div>

      {isExpanded && (
        <div id="ambient-synth-controls" className="ambient-synth-body">
      <section className="ambient-piano-panel" aria-label="Root note piano roll">
        <div className="ambient-piano-head">
          <h3>Root Note</h3>
          <span>{midiToNoteName(settings.rootNote)}</span>
        </div>
        <div className="ambient-piano-roll">
          {pianoRollNotes.map((note) => {
            const isRoot = Number(settings.rootNote) === note;
            const isActive = Number(activeArpNote) % 12 === note % 12;

            return (
              <button
                type="button"
                key={note}
                className={`${isSharpNote(note) ? "sharp" : "natural"} ${isRoot ? "root" : ""} ${
                  isActive && isRunning && isArpEnabled ? "playing" : ""
                }`}
                onClick={() => selectRootNote(note)}
                aria-label={`Set root note to ${midiToNoteName(note)}`}
                aria-pressed={isRoot}
              >
                <span>{midiToNoteName(note)}</span>
              </button>
            );
          })}
        </div>
      </section>

      <div className="ambient-synth-grid">
        {synthModules.map((module) => (
          <section className="ambient-synth-module" key={module.title}>
            <h3>{module.title}</h3>
            {module.controls.map((control) => (
              <label className="ambient-synth-control" key={control.key}>
                <span>
                  {control.label}
                  <em>{formatControlValue(settings[control.key], control.unit)}</em>
                </span>
                <input
                  type="range"
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  value={settings[control.key]}
                  onChange={(event) => updateSetting(control.key, event.target.value)}
                />
              </label>
            ))}
          </section>
        ))}

        <section className="ambient-synth-module ambient-synth-output">
          <h3>Output</h3>
          <label className="ambient-synth-control">
            <span>
              Volume
              <em>{formatControlValue(settings.volume, "")}</em>
            </span>
            <input
              type="range"
              min="0.04"
              max="0.42"
              step="0.01"
              value={settings.volume}
              onChange={(event) => updateSetting("volume", event.target.value)}
            />
          </label>
          <label className="ambient-synth-control">
            <span>
              Tempo
              <em>{formatControlValue(settings.tempo, "bpm")}</em>
            </span>
            <input
              type="range"
              min="48"
              max="132"
              step="1"
              value={settings.tempo}
              onChange={(event) => updateSetting("tempo", event.target.value)}
            />
          </label>
          <label className="ambient-synth-control">
            <span>
              Drums
              <em>{formatControlValue(settings.drumVolume, "")}</em>
            </span>
            <input
              type="range"
              min="0"
              max="0.55"
              step="0.01"
              value={settings.drumVolume}
              onChange={(event) => updateSetting("drumVolume", event.target.value)}
            />
          </label>
          <div className="ambient-patch-row" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
        </section>
      </div>

      <section className="ambient-arp-panel" aria-label="Arpeggiator controls">
        <div className="ambient-arp-head">
          <h3>Arpeggiator</h3>
          <button
            type="button"
            className={isArpEnabled ? "active" : ""}
            onClick={() => setIsArpEnabled((enabled) => !enabled)}
          >
            {isArpEnabled ? "On" : "Off"}
          </button>
        </div>
        <div className="ambient-arp-patterns">
          {arpPatterns.map((patternOption) => (
            <button
              type="button"
              key={patternOption.key}
              className={arpPatternKey === patternOption.key ? "active" : ""}
              onClick={() => setArpPatternKey(patternOption.key)}
            >
              {patternOption.label}
            </button>
          ))}
        </div>
      </section>

      <section className="ambient-sequencer" aria-label="Analog drum sequencer">
        <div className="ambient-sequencer-head">
          <h3>Analog Drums</h3>
          <span>{String(currentStep + 1).padStart(2, "0")}</span>
        </div>

        <div className="ambient-step-index" aria-hidden="true">
          {stepIndexes.map((step) => (
            <span key={step} className={step === currentStep && isRunning ? "current" : ""}>
              {step + 1}
            </span>
          ))}
        </div>

        <div className="ambient-step-grid">
          {drumTracks.map((track) => (
            <div className="ambient-step-row" key={track.key}>
              <span>{track.label}</span>
              <div>
                {stepIndexes.map((step) => {
                  const isActive = pattern[track.key][step];
                  const isCurrent = step === currentStep && isRunning;

                  return (
                    <button
                      type="button"
                      key={`${track.key}-${step}`}
                      className={`${isActive ? "active" : ""} ${isCurrent ? "current" : ""}`}
                      aria-label={`${track.label} step ${step + 1}`}
                      aria-pressed={isActive}
                      onClick={() => toggleStep(track.key, step)}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="ambient-drum-shape" aria-label="Drum shaping controls">
        {drumShapeModules.map((module) => (
          <section className="ambient-drum-module" key={module.title}>
            <h3>{module.title}</h3>
            {module.controls.map((control) => (
              <label className="ambient-synth-control" key={control.key}>
                <span>
                  {control.label}
                  <em>{formatControlValue(settings[control.key], control.unit)}</em>
                </span>
                <input
                  type="range"
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  value={settings[control.key]}
                  onChange={(event) => updateSetting(control.key, event.target.value)}
                />
              </label>
            ))}
          </section>
        ))}
      </section>
        </div>
      )}

      {error && <p className="music-status">{error}</p>}
    </article>
  );
}

export default AmbientSynth;
