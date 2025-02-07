import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Particle from "../Particle";
import fox from "../../Assets/Learning/fox.jpg";
import silverFox from "../../Assets/Learning/silver fox.jpg";
import hedge from "../../Assets/Learning/hedge.jpg";
import bird from "../../Assets/Learning/bird.jpg";
import dove from "../../Assets/Learning/dove.jpg";
import toucan from "../../Assets/Learning/toucan.jpg";
import crab from "../../Assets/Learning/crab.jpg";
import squid from "../../Assets/Learning/squid.jpg";
import LearningCards from "./LearningCards";
import SupervisedLearning from "./Pages/SupervisedLearning";

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
            <LearningCards
              imgPath={fox}
              isBlog={false}
              title="Supervised Learning"
              description="Regression, Classifiers, KNN, SVM, Naive Bayes, Decision Trees, Ensembles"
              link='/SupervisedLearning'
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={silverFox}
              isBlog={false}
              title="Unsupervised Learning"
              description="Clustering, PCA, LDA, ICA, SVD, UMAP, t-SNE"
              link="https://github.com/soumyajit4419/Bits-0f-C0de"
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={hedge}
              isBlog={false}
              title="Deep Learning"
              description="Neural Networks, RNN, LSTM, GANs, Autoencoders, Transformers"
              link="https://github.com/soumyajit4419/Editor.io"         
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={bird}
              isBlog={false}
              title="Computer Vision"
              description="Convolutional Neural Networks, Image Segmentation, Object Detection"
              link="https://github.com/soumyajit4419/AI_For_Social_Good"
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={dove}
              isBlog={false}
              title="Reinforcement Learning"
              description="Q-Learning, Actor-Critic, Genetic Algorithms, Markov Decision Processes"
              link="https://github.com/soumyajit4419/Plant_AI"
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={toucan}
              isBlog={false}
              title="Natural Language Processing"
              description="Tokenization, Semantic Analysis, Speech Tagging, Speech-to-Text"
              link="https://github.com/soumyajit4419/Face_And_Emotion_Detection"
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={crab}
              isBlog={false}
              title="Optimization Methods"
              description="SGD, Adam, RMSProp, Adagrad, Adadelta"
              link="https://github.com/soumyajit4419/Face_And_Emotion_Detection"
            />
          </Col>

          <Col md={4} className="project-card">
            <LearningCards
              imgPath={squid}
              isBlog={false}
              title="Lite ML"
              description="TensorFlow Lite, ONNX, CoreML, Embedded Systems"
              link="https://github.com/soumyajit4419/Face_And_Emotion_Detection"
            />
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default Learning;
