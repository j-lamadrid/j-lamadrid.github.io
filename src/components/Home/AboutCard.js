import React from "react";
import Card from "react-bootstrap/Card";
import { ImPointRight } from "react-icons/im";

function AboutCard() {
  return (
    <Card className="quote-card-view">
      <Card.Body>
        <blockquote className="blockquote mb-0">
          <p style={{ textAlign: "justify" }}>
            I am from <span className="brown"> San Diego, California</span> and
            I am currently pursuing my Masters. I have completed my Bachelors of Science 
            in Cognitive Science specializing in Machine Learning with a minor in 
            Data Science at the University of California, San Diego.
            <br />
            <br />
            My previous accomplishments include a NASA internship at the Langley
            Research Center at the Atmospheric Science Data Center as a Software
            Developer involved in the Quality Analysis and Migration of Scientific
            Data on the <a href="https://tolnet.larc.nasa.gov/" target="_blank" 
            rel="noopener noreferrer" style={{ color: "white" }}>TOLNet</a> service.
            I was also awarded a 2024 Cognitive Science Summer Scholar Research
            Grant for my project on Artificial Intelligence methods for audio
            reconstruction from EEG signals.
            <br />
            <br />
            Some of my general research interests include:
            <ul>
            <li className="about-activity">
              + Machine Learning in Signal Processing for Audio/Neural Signals
            </li>
            <li className="about-activity">
              + Embedded Artificial Intelligence and Lite Machine Learning
            </li>
            <li className="about-activity">
              + Computational Neuroscience
            </li>
          </ul>
            <br />
            Outside of academics, my hobbies include:
          </p>
          <ul>
            <li className="about-activity">
              + Making Music
            </li>
            <li className="about-activity">
              + Going to the Gym
            </li>
            <li className="about-activity">
              + Surfing
            </li>
          </ul>
        </blockquote>
      </Card.Body>
    </Card>
  );
}

export default AboutCard;
