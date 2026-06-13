import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Link } from "react-router-dom";
import { AiOutlineFund } from "react-icons/ai";
import { CgFileDocument } from "react-icons/cg";
import { FiArrowUpRight } from "react-icons/fi";
import Particle from "../Particle";
import About from "./About";
import Type from "./Type";
import pdf from "../../Assets/Jacob_Lamadrid_Resume.pdf";

const homeGif =
  "https://nukochannel.neocities.org/NukoImg/Stickers/nukoStickerIRLCat.gif";

function Home() {
  return (
    <section>
      <Container fluid className="home-section" id="home">
        <Particle />
        <Container className="home-content">
          <Row className="align-items-center">
            <Col xs={12} lg={7} className="home-header reveal-up">
              <p className="home-eyebrow">
                Data Science | Machine Learning | Scientific Software | Research
              </p>
              <h1 className="heading-name">
                Jacob <strong className="main-name">Lamadrid</strong>
              </h1>

              <div className="type-wrap" aria-label="Current focus areas">
                <Type />
              </div>

              <div className="home-actions">
                <Link to="/projects" className="primary-link-button">
                  <AiOutlineFund />
                  <span>View Projects</span>
                </Link>
                <a
                  href={pdf}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="secondary-link-button"
                >
                  <CgFileDocument />
                  <span>Download Resume</span>
                </a>
              </div>

              <div className="home-snapshot" aria-label="Professional snapshot">
                <div>
                  <span>Georgia Tech (MS)</span>
                  <small>Analytics - Modeling, Simulation, and Optimization</small>
                </div>
                <div>
                  <span>UC San Diego (BS)</span>
                  <small>Cognitive Science - Machine Learning and Neural Computation</small>
                </div>
              </div>
            </Col>

            <Col xs={12} lg={5} className="home-visual-col reveal-up delay-1">
              <div className="home-visual-panel">
                <img
                  src={homeGif}
                  alt="Nuko pixel cat at a computer"
                  className="home-avatar"
                />
                <div className="home-signal-strip">
                  <span>Research</span>
                  <span>Modeling</span>
                  <span>Engineering</span>
                  <FiArrowUpRight aria-hidden="true" />
                </div>
              </div>
            </Col>
          </Row>
        </Container>
      </Container>
      <About />
    </section>
  );
}

export default Home;
