import eegReport from "../../Assets/Projects/Fellowship Presentation.pptx.pdf";
import audioTextureReport from "../../Assets/Projects/Project_Report___ECE_176 (2).pdf";

const projects = [
  {
    title: "Bayesian Temporal Signal Extraction & Uncertainty Quantification",
    domain: "Bayesian Signal Processing",
    type: "R/Shiny",
    summary:
      "Probabilistic peak detection and uncertainty quantification for sparse LC-MS sensor data with an interactive Shiny app.",
    impact:
      "Uses BIC model selection, Stan variational inference, temporal aggregation, and 2D/3D visualization to separate latent signal from stochastic noise and artifacts.",
    stack: ["R", "Stan", "RShiny", "Plotly", "Variational Bayes", "LC-MS"],
    links: {
      github: "https://github.com/j-lamadrid/probabilistic-sensor-uq",
      demo: "https://j-lamadrid.shinyapps.io/probabilistic-sensor-uq/",
    },
    featured: true,
  },
  {
    title: "ACE Clinical Data Desk",
    domain: "Clinical Data Engineering",
    type: "R/Shiny",
    summary:
      "A deployed Shiny data-management app packaging clinical workflows for the UCSD Autism Center of Excellence.",
    impact:
      "Supports eye-tracking merges, visit flagging, diagnostic grouping, treatment-hour summaries, and MacArthur percentile population for research data workflows.",
    stack: ["Python", "Shiny", "Pandas", "Clinical Data", "Excel Workflows"],
    links: {
      github: "https://github.com/j-lamadrid/clinical-dm-shiny",
      demo: "https://j-lamadrid.shinyapps.io/ace_clinical_data_desk/",
    },
    featured: true,
  },
  {
    title: "EEG2Audio",
    domain: "NeuroAI",
    type: "Research",
    summary:
      "Audio reconstruction research from EEG signals using spectrogram representations and deep generative modeling workflows.",
    impact:
      "Connects EEG preprocessing, audio feature extraction, CNNs, latent diffusion, and DCGAN experiments into reproducible neuroimaging pipelines.",
    stack: ["Python", "PyTorch", "EEG", "Spectrograms", "Deep Learning"],
    links: {
      report: eegReport,
    },
  },
  {
    title: "Audio Texture Synthesis",
    domain: "Signal Processing",
    type: "Applied ML",
    summary:
      "Audio texture generation using PyTorch and TorchAudio by adapting image texture synthesis and style-transfer methods to spectrograms.",
    impact:
      "Builds a CNN-based generative approach around Gram matrix optimization while documenting network design, limitations, and audio-quality tradeoffs.",
    stack: ["Python", "PyTorch", "TorchAudio", "CNNs", "Signal Processing"],
    links: {
      github: "https://github.com/j-lamadrid/audio-texture-synthesis",
      report: audioTextureReport,
    },
  },
  {
    title: "TrackiT!",
    domain: "Health Application",
    type: "Software",
    summary:
      "Cross-platform clinical tracking app for longitudinal developmental trends across neurodevelopmental studies.",
    impact:
      "Combines Flutter, Dart, and Firestore with clinical workflow design for treatment engagement, provider contact, and progress tracking.",
    stack: ["Flutter", "Dart", "Firestore", "Mobile UX", "Health Tech"],
    links: {
      github: "https://github.com/ACE-UCSD/Treatment_Engagement_Parent_App",
    },
  },
  {
    title: "WaveMAP",
    domain: "Computational Neuroscience",
    type: "Applied ML",
    summary:
      "Neural signal processing and analysis project applying WaveMAP-style waveform clustering to neocortical waveforms.",
    impact:
      "Highlights reproducible analysis, waveform feature extraction, dimensionality reduction, and clustering for high-dimensional neural recordings.",
    stack: ["Python", "Jupyter", "Clustering", "Waveforms", "Dimensionality Reduction"],
    links: {
      github: "https://github.com/j-lamadrid/wavemap-project",
    },
  },
  {
    title: "MRI Tumor Classification",
    domain: "Computer Vision",
    type: "Applied ML",
    summary:
      "Applied machine learning and deep learning methods to MRI data for brain tumor detection and classification.",
    impact:
      "Demonstrates medical-imaging preprocessing, classifier comparison, model evaluation, and applied computer vision for healthcare data.",
    stack: ["Python", "Medical Imaging", "Classification", "Computer Vision"],
    links: {
      github: "https://github.com/j-lamadrid/mri-tumor-classification",
    },
  },
   {
    title: "Meteor Instrument (WIP)",
    domain: "Radio Astronomy & Meteorology",
    type: "Engineering",
    summary:
      "Developing a standalone embedded instrument on Raspberry Pi and Arduino display hardware for real-time radio astronomy and meteorology data capture, processing, and visualization.",
    impact:
      "Integrates embedded systems design, signal processing, and data visualization for a singular scientific instrument.",
    stack: ["Python", "C++", "Raspberry Pi", "Arduino", "Embedded Systems", "Signal Processing", "RTL-SDR"],
    links: {
      demo: "/projects/meteor-instrument-demo",
    },
    demoLabel: "Prototype",
  },
];

export const projectFilters = ["All", "R/Shiny", "Research", "Applied ML", "Software", "Engineering"];

export default projects;
