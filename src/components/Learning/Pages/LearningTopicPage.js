import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { Link, Navigate, useParams } from "react-router-dom";
import { FiArrowLeft } from "react-icons/fi";
import Particle from "../../Particle";
import learningTopics from "../learningData";

function LearningTopicPage() {
  const { topicSlug } = useParams();
  const topic = learningTopics.find((item) => item.slug === topicSlug);

  if (!topic) {
    return <Navigate to="/learning" />;
  }

  return (
    <Container fluid className="project-section learning-topic-section">
      <Particle />
      <Container className="project-shell learning-topic-shell">
        <Link to="/learning" className="learning-back-link">
          <FiArrowLeft />
          <span>Learning Lab</span>
        </Link>

        <section className="section-intro reveal-up">
          <p className="section-kicker">Learning Guide</p>
          <h1 className="project-heading">
            <strong className="brown">{topic.title}</strong>
          </h1>
          <p className="section-lede">{topic.overview}</p>
        </section>

        <Row className="learning-topic-grid">
          <Col lg={4}>
            <article className="learning-topic-panel">
              <h2>Core Methods</h2>
              <ul>
                {topic.methods.map((method) => (
                  <li key={method.name}>
                    <strong>{method.name}</strong>
                    <span>{method.detail}</span>
                  </li>
                ))}
              </ul>
            </article>
          </Col>

          <Col lg={4}>
            <article className="learning-topic-panel">
              <h2>Key Ideas</h2>
              <ul>
                {topic.keyIdeas.map((idea) => (
                  <li key={idea}>{idea}</li>
                ))}
              </ul>
            </article>
          </Col>

          <Col lg={4}>
            <article className="learning-topic-panel learning-formula-panel">
              <h2>Mathematical Formulations</h2>
              <div className="formula-list">
                {topic.methods.map((method) => (
                  <div className="formula-card" key={`${method.name}-formula`}>
                    <span>{method.name}</span>
                    <code>{method.formula}</code>
                  </div>
                ))}
              </div>
            </article>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default LearningTopicPage;
