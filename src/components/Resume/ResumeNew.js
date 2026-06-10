import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Button from "react-bootstrap/Button";
import Particle from "../Particle";
import pdf from "../../Assets/Jacob_Lamadrid_Resume.pdf";
import resumeData from "./resumeData";
import { AiOutlineDownload, AiOutlineMail, AiOutlinePhone } from "react-icons/ai";
import { BsGithub, BsLinkedin } from "react-icons/bs";

const contactIcons = {
  Phone: <AiOutlinePhone />,
  Email: <AiOutlineMail />,
  GitHub: <BsGithub />,
  LinkedIn: <BsLinkedin />,
};

function ResumeItem({ item }) {
  return (
    <article className="resume-native-item">
      <div className="resume-native-item-head">
        <div>
          <h3>{item.title || item.degree}</h3>
          <p>{item.organization || item.school}</p>
        </div>
        <div className="resume-native-meta">
          <span>{item.date}</span>
          <small>{item.location}</small>
        </div>
      </div>

      {item.stack && (
        <div className="project-stack resume-stack">
          {item.stack.map((tool) => (
            <span key={tool}>{tool}</span>
          ))}
        </div>
      )}

      <ul>
        {(item.bullets || item.details).map((detail) => (
          <li key={detail}>{detail}</li>
        ))}
      </ul>
    </article>
  );
}

function ResumeSection({ title, children }) {
  return (
    <section className="resume-native-section reveal-up">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function ResumeNew() {
  return (
    <Container fluid className="resume-section resume-native-section-wrap">
      <Particle />
      <Container className="resume-native-shell">
        <section className="section-intro resume-native-hero reveal-up">
          <p className="section-kicker">Resume</p>
          <h1 className="project-heading">
            {resumeData.name}
            <br />
            <strong className="brown">{resumeData.headline}</strong>
          </h1>

          <div className="resume-contact-grid">
            {resumeData.contact.map((contact) => (
              <a
                key={contact.label}
                href={contact.href}
                target={contact.href.startsWith("http") ? "_blank" : undefined}
                rel="noreferrer"
              >
                {contactIcons[contact.label]}
                <span>{contact.value}</span>
              </a>
            ))}
          </div>

          <Button
            variant="primary"
            href={pdf}
            target="_blank"
            rel="noopener noreferrer"
            className="resume-download-button"
          >
            <AiOutlineDownload />
            <span>Download PDF</span>
          </Button>
        </section>

        <Row>
          <Col lg={5}>
            <ResumeSection title="Education">
              {resumeData.education.map((item) => (
                <ResumeItem item={item} key={`${item.school}-${item.degree}`} />
              ))}
            </ResumeSection>

            <ResumeSection title="Technical Skills">
              <div className="resume-skills-list">
                {resumeData.skills.map((group) => (
                  <div key={group.label}>
                    <h3>{group.label}</h3>
                    <div className="project-stack resume-stack">
                      {group.values.map((value) => (
                        <span key={value}>{value}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </ResumeSection>
          </Col>

          <Col lg={7}>
            <ResumeSection title="Experience">
              {resumeData.experience.map((item) => (
                <ResumeItem item={item} key={`${item.organization}-${item.title}`} />
              ))}
            </ResumeSection>

            <ResumeSection title="Projects">
              {resumeData.projects.map((item) => (
                <ResumeItem item={item} key={item.title} />
              ))}
            </ResumeSection>
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default ResumeNew;
