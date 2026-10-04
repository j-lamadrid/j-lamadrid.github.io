import React from "react";
import Card from "react-bootstrap/Card";

function AboutCard() {
  return (
    <Card className="quote-card-view about-copy-card">
      <Card.Body>
        <p>
        I'm from <span className="brown">San Diego, California</span> and am
        currently pursuing a Master's in Analytics at{" "}
        <span className="brown">Georgia Tech</span>, focused on probabilistic
        modeling, machine learning, and analytical tools. I earned my B.S. in
        Cognitive Science (Machine Learning and Neural Computation) with a minor
        in Data Science at <span className="brown">UC San Diego</span>.
      </p>

      <p>
        I currently work as a{" "}
        <span className="brown">Production Test Technician</span> at{" "}
        <span className="brown">Frontgrade Technologies</span>, testing RF
        components and building hands-on experience with hardware performance and
        reliability. This past summer, I was a{" "}
        <span className="brown">Graduate Data Science Intern</span> at{" "}
        <span className="brown">Centene Corporation</span>,
        where I developed a machine learning model for risk assessment and anomaly 
        detection on claims data.
      </p>

      <p>
        Before that, I interned at{" "}
        <span className="brown">NASA Langley Research Center</span> in the
        Atmospheric Science Data Center, where I worked on quality analysis and
        migration of scientific data for the{" "}
        <a
          href="https://tolnet.larc.nasa.gov/"
          target="_blank"
          rel="noopener noreferrer"
        >
          TOLNet
        </a>{" "}
        service. At UCSD, I was awarded the{" "}
        <span className="brown">2024 Cognitive Science Summer Scholar</span>{" "}
        Research Grant for work on AI methods for audio reconstruction from EEG
        signals. I also worked at a clinical research lab at UCSD, developing data
        management tools and a mobile/web app to support an NIH-funded study on
        tracking parent interaction with infants at risk for autism spectrum
        disorder.
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
              <li>Simulation Methods</li>
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
