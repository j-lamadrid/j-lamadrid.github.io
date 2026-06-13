const resumeData = {
  name: "Jacob Lamadrid",
  headline: "Machine Learning | Data Science | Systems Analytics | Research",
  contact: [
    {
      label: "Phone",
      value: "(760) 622-7023",
      href: "tel:17606227023",
    },
    {
      label: "Email",
      value: "jlamadrid3@gatech.edu",
      href: "mailto:jlamadrid3@gatech.edu",
    },
    {
      label: "GitHub",
      value: "github.com/j-lamadrid",
      href: "https://github.com/j-lamadrid",
    },
    {
      label: "LinkedIn",
      value: "linkedin.com/in/jlamadrid",
      href: "https://linkedin.com/in/jlamadrid",
    },
  ],
  education: [
    {
      degree: "Master of Science in Analytics",
      school: "Georgia Institute of Technology",
      location: "Atlanta, GA",
      date: "Expected: Dec 2027",
      details: [
        "Focus: Modeling, Simulation, and Optimization",
        "Coursework: Bayesian Statistics, High Dimensional Data Analytics, Data & Visual Analytics",
      ],
    },
    {
      degree: "Bachelor of Science in Cognitive Science - Machine Learning and Neural Computation",
      school: "University of California, San Diego",
      location: "San Diego, CA",
      date: "Jun 2024",
      details: [
        "Minor: Data Science",
        "Provost Honors | 2024 Cognitive Science Summer Scholar ($6,000 Award)",
        "Instructional Assistant: Introduction to Python, Introduction to Data Science",
      ],
    },
  ],
  experience: [
    {
      title: "Graduate Data Science Intern",
      organization: "Centene Corporation",
      location: "St. Louis, MO",
      date: "May 2026 - Aug 2026",
      bullets: [
        "Developed end-to-end data engineering pipelines unifying Internal Audit and claims warehouse sources into a common schema of 127K+ claims for anomaly detection on large-scale health insurance claims data.",
        "Independently designed and implemented an AutoEncoder-based weakly-supervised anomaly detection method in PyTorch, achieving an 10x increase in precision of identifying erroneous claims over the prior baseline on a 50:1 class-imbalanced dataset.",
      ],
    },
    {
      title: "Research Fellow",
      organization: "UC San Diego Natural Computation Lab",
      location: "San Diego, CA",
      date: "Jul 2024 - Sep 2024",
      bullets: [
        "Conducted independent research on audio reconstruction from EEG signals using CNNs, Latent Diffusion Models, and DCGANs; managed full ML lifecycle from multimodal data ingestion to model evaluation.",
        "Designed signal processing pipelines for EEG and audio data; translated theoretical research into reproducible PyTorch experiments on custom neuroimaging datasets.",
      ],
    },
    {
      title: "Application Developer & Research Data Associate",
      organization: "UC San Diego Health - Autism Center of Excellence",
      location: "San Diego, CA",
      date: "Jun 2022 - Sep 2024",
      bullets: [
        "Built TrackiT!, a cross-platform clinical app (Flutter/Dart/Firestore) for longitudinal tracking of developmental trends across neurodevelopmental studies.",
        "Designed GUI tools and algorithmic pipelines for clinical data entry, eye-tracking QA, and behavioral assessments, supporting downstream statistical modeling workflows.",
      ],
    },
    {
      title: "Software Development Intern",
      organization: "NASA Langley Research Center",
      location: "Hampton, VA",
      date: "Jun 2023 - Sep 2023",
      bullets: [
        "Developed end-to-end data pipelines in Python to standardize and migrate legacy atmospheric science datasets to the NASA TOLNet repository, ensuring format compliance across multiple research instruments and sites.",
        "Built automated data quality analytics to flag anomalies and inconsistencies in historical lidar datasets; delivered summary reports and documentation to cross-functional research teams.",
      ],
    },
  ],
  projects: [
    {
      title: "Bayesian Temporal Signal Extraction & Uncertainty Quantification",
      date: "2025",
      stack: ["R", "RStan", "RShiny", "Variational Inference", "LC-MS Mass Spectrometry"],
      bullets: [
        "Engineered a Bayesian deconvolution pipeline for sparse high-dimensional sensor arrays; designed automated QC using posterior uncertainty estimates to separate latent signals from stochastic noise and artifacts.",
      ],
    },
    {
      title: "Audio Texture Synthesis via Convolutional Neural Networks",
      date: "2023",
      stack: ["Python", "PyTorch", "Generative AI", "Signal Processing"],
      bullets: [
        "Adapted 2D computer vision style-transfer algorithms for 1D temporal audio data; built a CNN-based generative model from scratch using PyTorch to synthesize realistic audio textures via Gram matrix optimization.",
      ],
    },
    {
      title: "Meteor Instrument - Embedded Radio Astronomy & Meteorology System",
      date: "Ongoing",
      stack: ["Python", "Raspberry Pi", "C++", "Arduino", "Embedded Systems", "Signal Processing"],
      bullets: [
        "Developing a standalone embedded instrument on Raspberry Pi and Arduino display hardware for real-time radio astronomy and meteorology data capture, processing, and visualization.",
      ],
    },
  ],
  skills: [
    {
      label: "Languages",
      values: ["Python", "R", "MATLAB", "Julia", "SQL", "C++", "JavaScript", "Dart"],
    },
    {
      label: "Frameworks & Tools",
      values: [
        "PyTorch",
        "TensorFlow",
        "Scikit-learn",
        "Pandas",
        "NumPy",
        "RStan",
        "RShiny",
        "Spark",
        "Firestore",
        "Git",
      ],
    },
  ],
};

export default resumeData;
