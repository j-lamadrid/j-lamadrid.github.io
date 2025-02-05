import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ProjectCard from "./LearningCards";
import Particle from "../Particle";
import fox from "../../Assets/Learning/fox.jpg";
import silverFox from "../../Assets/Learning/silver fox.jpg";
import hedge from "../../Assets/Learning/hedge.jpg";
import bird from "../../Assets/Learning/bird.jpg";
import dove from "../../Assets/Learning/dove.jpg";
import toucan from "../../Assets/Learning/toucan.jpg";
import crab from "../../Assets/Learning/crab.jpg";
import squid from "../../Assets/Learning/squid.jpg";

function Learning() {
  return (
    <Container fluid className="project-section">
      <Particle />
      <Container>
        <h1 className="project-heading">
          Machine Learning and Artificial Intelligence <strong className="brown">Models </strong>
        </h1>
        <p style={{ color: "white" }}>
          Learn the fundamentals
        </p>
        <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={fox}
              isBlog={false}
              title="Supervised Learning"
              description="Regression, Classifiers, KNN, SVM, Naive Bayes, Decision Trees, Ensembles"
              ghLink="https://github.com/soumyajit4419/Chatify"
              demoLink="https://chatify-49.web.app/"
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={silverFox}
              isBlog={false}
              title="Unsupervised Learning"
              description="Clustering, PCA, LDA, ICA, SVD, UMAP, t-SNE"
              ghLink="https://github.com/soumyajit4419/Bits-0f-C0de"
              demoLink="https://blogs.soumya-jit.tech/"
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={hedge}
              isBlog={false}
              title="Deep Learning"
              description="Neural Networks, RNN, LSTM, GANs, Autoencoders, Transformers"
              ghLink="https://github.com/soumyajit4419/Editor.io"
              demoLink="https://editor.soumya-jit.tech/"              
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={dove}
              isBlog={false}
              title="Reinforcement Learning"
              description="Q-Learning, Actor-Critic, Genetic Algorithms, Markov Decision Processes"
              ghLink="https://github.com/soumyajit4419/Plant_AI"
              demoLink="https://plant49-ai.herokuapp.com/"
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={bird}
              isBlog={false}
              title="Computer Vision"
              description="Convolutional Neural Networks, Image Segmentation, Object Detection"
              ghLink="https://github.com/soumyajit4419/AI_For_Social_Good"
              // demoLink="https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley" <--------Please include a demo link here
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={toucan}
              isBlog={false}
              title="Natural Language Processing"
              description="Tokenization, Semantic Analysis, Speech Tagging, Speech-to-Text"
              ghLink="https://github.com/soumyajit4419/Face_And_Emotion_Detection"
              // demoLink="https://blogs.soumya-jit.tech/"      <--------Please include a demo link here 
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={crab}
              isBlog={false}
              title="Optimization Methods"
              description="SGD, Adam, RMSProp, Adagrad, Adadelta"
              ghLink="https://github.com/soumyajit4419/Face_And_Emotion_Detection"
              // demoLink="https://blogs.soumya-jit.tech/"      <--------Please include a demo link here 
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={squid}
              isBlog={false}
              title="Lite ML"
              description="TensorFlow Lite, ONNX, CoreML, Embedded Systems"
              ghLink="https://github.com/soumyajit4419/Face_And_Emotion_Detection"
              // demoLink="https://blogs.soumya-jit.tech/"      <--------Please include a demo link here 
            />
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default Learning;
