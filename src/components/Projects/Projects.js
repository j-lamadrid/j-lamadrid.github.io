import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ProjectCard from "./ProjectCards";
import Particle from "../Particle";
import brain from "../../Assets/Projects/Te-glTr_0000.jpg";
import waves from "../../Assets/Projects/waves.png";
import spec from "../../Assets/Projects/bird_generated.png";
import model from "../../Assets/Projects/figure.PNG";
import flutter from "../../Assets/Projects/IMG_4649.png";
import eeg from "../../Assets/Projects/eeg_viz.png";
import eeg2audio from "../../Assets/Projects/Fellowship Presentation.pptx.pdf"
import ats from "../../Assets/Projects/Project_Report___ECE_176 (2).pdf"

function Projects() {
  return (
    <Container fluid className="project-section">
      <Particle />
      <Container>
        <h1 className="project-heading">
          <strong className="brown">Projects </strong>
        </h1>
        <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={spec}
              isBlog={false}
              title="Audio Texture Synthesis"
              description="Method of generation of audio textures using conventional techniques found in image texture synthesis, image in-painting and style transfer"
              ghLink="https://github.com/j-lamadrid/audio-texture-synthesis"
              demoLink={ats}
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={eeg}
              isBlog={false}
              title="EEG2Audio"
              description="Seeking to extract/reconstruct auditory stimuli via deep learning models with some input neural signal, possible via time-frequency domain images (Spectrograms)"
              demoLink={eeg2audio}
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={flutter}
              isBlog={false}
              title="TrackiT!"
              description="Flutter app originally created to help track the amount of time a parent spends in active treatment engagement with their child, but has included features to help parents contact providers and track their child’s progress"
              ghLink="https://github.com/ACE-UCSD/Treatment_Engagement_Parent_App"          
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={waves}
              isBlog={false}
              title="WaveMAP"
              description="Applying the WaveMAP method from Lee et al. (2021) to a new dataset from Horvath et al. (2021) to validate and extend the clustering of neocortical waveforms"
              ghLink="https://github.com/j-lamadrid/wavemap-project"
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={brain}
              isBlog={false}
              title="MRI Tumor Classification"
              description="The application of machine and deep learning techniques to Magnetic Resonance Imaging (MRI) data in order to address the task of detecting and classifying brain tumors"
              ghLink="https://github.com/j-lamadrid/mri-tumor-classification"
            />
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default Projects;
