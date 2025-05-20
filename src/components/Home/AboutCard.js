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
            I am currently pursuing my Masters in Analytics at 
            <span className="brown"> Georgia Tech</span> with a 
            focus on analytical tools and statistical learning. I have completed my 
            Bachelors of Science in Cognitive Science specializing in Machine Learning 
            with a minor in Data Science at the 
            <span className="brown"> University of California, San Diego</span>.
            <br />
            <br />
            My previous experiences include a <span className="brown"> NASA </span> 
            internship at the Langley
            Research Center with the Atmospheric Science Data Center as a Software
            Developer involved in the Quality Analysis and Migration of Scientific
            Data on the <a href="https://tolnet.larc.nasa.gov/" target="_blank" 
            rel="noopener noreferrer" style={{ color: "white" }}>TOLNet</a> service.
            I was also awarded the <span className="brown">2024 Cognitive Science 
              Summer Scholar</span> Research Grant for my project on Artificial 
              Intelligence methods for audio reconstruction from EEG signals.
            <br />
            <br />
            Some of my general research interests include:
            <ul>
            <li className="about-activity">
              + Machine Learning in Signal Processing for Audio/Neural Signals
            </li>
            <li className="about-activity">
              + Probabilistic Modeling and High Dimensional Data Analysis
            </li>
            <li className="about-activity">
              + Embedded Artificial Intelligence and Lite Machine Learning
            </li>
          </ul>
            <br />
            Outside of academics, my hobbies include:
          </p>
          <ul>
            <li className="about-activity">
              + Making Music
            </li>
            <li className="about-activity" style={{ display: "flex", alignItems: "center" }}>
              + Going to the Gym&nbsp;
              <span style={{ display: "flex", alignItems: "center" }}>
                (245 Bench, 365 Squat, 420 Deadlift)
                <img
                  src='https://nukocities.neocities.org/nuko/act/cat527.gif'
                  alt="flex"
                  style={{ width: "25px", marginLeft: "8px" }}
                />
              </span>
            </li>
            <li className="about-activity">
              + Surfing (North Ponto, Swamis, La Jolla Shores)
            </li>
          </ul>
        </blockquote>
      </Card.Body>
    </Card>
  );
}

export default AboutCard;
