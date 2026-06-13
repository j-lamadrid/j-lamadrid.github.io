import React from "react";
import Card from "react-bootstrap/Card";

function AboutCard() {
  return (
    <Card className="quote-card-view about-copy-card">
      <Card.Body>
        <p>
          I am from <span className="brown">San Diego, California</span> and I
          am pursuing a Masters in Analytics at{" "}
          <span className="brown">Georgia Tech</span>, focused on analytical
          tools, probabilistic modeling, and machine learning and am a 
          <span className="brown"> Graduate Data Science Intern</span> with Centene. 
          I have also completed
          my B.S. in Cognitive Science specializing in Machine Learning with a
          minor in Data Science at{" "}
          <span className="brown">UC San Diego</span>.
        </p>

        <p>
          My previous experience includes a{" "}
          <span className="brown">NASA Langley Research Center</span> internship
          with the Atmospheric Science Data Center, where I worked on quality
          analysis and migration of scientific data for the{" "}
          <a
            href="https://tolnet.larc.nasa.gov/"
            target="_blank"
            rel="noopener noreferrer"
          >
            TOLNet
          </a>{" "}
          service. I was also awarded the{" "}
          <span className="brown">2024 Cognitive Science Summer Scholar</span>{" "}
          Research Grant for work on AI methods for audio reconstruction from
          EEG signals. Among these experiences, I have also worked at a clinical
          research lab at UCSD involved in developing data management tools and
          a mobile/web app for a research grant proposal on methods for tracking
          parent's interaction with infants at risk for autism spectrum disorder
          under the NIH.
        </p>

        <div className="about-focus-grid">
          <div>
            <span>Research Interests</span>
            <ul>
              <li>Machine Learning for Scientific Data</li>
              <li>Bayesian Probabilistic Modeling in Signal Processing</li>
              <li>EEG Signal Processing and Analysis</li>
              <li>Computational Neuroscience and Brain-Computer Interfaces</li>
            </ul>
          </div>
          <div>
            <span>Relevant Coursework</span>
            <ul>
              <li>Deep Learning</li>
              <li>Advanced Machine Learning Methods</li>
              <li>Bayesian Statistics</li>
              <li>High Dimensional Data Analytics</li>
              <li>Systems for Scalable Analytics</li>
              <li>Data and Visual Analytics</li>
              <li>Neural Data Science</li>
              <li>Cognitive/Systems Neuroscience</li>
            </ul>
          </div>
        </div>

        <p className="about-personal-note">
          Outside of academics, I make music, hike, train consistently, and surf
          around North County and La Jolla. Currently reading Hard-Boiled Wonderland 
          and the End of the World by Haruki Murakami.
        </p>
      </Card.Body>
    </Card>
  );
}

export default AboutCard;
