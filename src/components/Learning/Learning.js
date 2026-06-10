import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Particle from "../Particle";
import LearningCards from "./LearningCards";
import learningTopics from "./learningData";

function Learning() {
  return (
    <Container fluid className="project-section learning-hub-section">
      <Particle />
      <Container className="project-shell">
        <section className="section-intro reveal-up">
          <p className="section-kicker">Learning Lab</p>
          <h1 className="project-heading">
            Machine learning and AI <strong className="brown">models</strong>
          </h1>
          <p className="section-lede">
            Topic guides for the models, workflows, and implementation details
            I revisit most often while building applied ML systems.
          </p>
        </section>

        <Row className="project-grid">
          {learningTopics.map((topic) => (
            <Col lg={4} md={6} className="project-card" key={topic.title}>
              <LearningCards {...topic} />
            </Col>
          ))}
        </Row>
      </Container>
    </Container>
  );
}

export default Learning;
