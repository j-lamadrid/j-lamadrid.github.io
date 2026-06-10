import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Particle from "../Particle";
import Github from "./Github";
import Techstack from "./Techstack";
import Aboutcard from "./AboutCard";
import Toolstack from "./Toolstack";
import Workstack from "./Workstack";

const aboutGif =
  "https://nukochannel.neocities.org/NukoImg/Stickers/nukoStickerThrone.gif";

function About() {
  return (
    <Container fluid className="about-section">
      <Particle />
      <Container className="about-container">
        <Row className="about-lead-row align-items-center">
          <Col xs={12} lg={7} className="reveal-up">
            <p className="section-kicker">About</p>
            <h1 className="project-heading">
              Building useful systems from{" "}
              <strong className="brown">scientific data</strong>.
            </h1>
            <Aboutcard />
          </Col>
          <Col xs={12} lg={5} className="about-img reveal-up delay-1">
            <div className="about-art-frame">
              <img src={aboutGif} alt="Nuko pixel cat sitting on a throne" />
            </div>
          </Col>
        </Row>

        <section className="stack-section reveal-up">
          <p className="section-kicker">Capabilities</p>
          <h1 className="project-heading">
            Professional <strong className="brown">Skillset</strong>
          </h1>
          <Techstack />
        </section>

        <section className="stack-section reveal-up">
          <p className="section-kicker">Workflow</p>
          <h1 className="project-heading">
            <strong className="brown">Tools</strong> I use
          </h1>
          <Toolstack />
        </section>

        <section className="stack-section reveal-up">
          <p className="section-kicker">Experience</p>
          <h1 className="project-heading">
            <strong className="brown">Work</strong> Experience
          </h1>
          <Workstack />
        </section>

        <Github />
      </Container>
    </Container>
  );
}

export default About;
