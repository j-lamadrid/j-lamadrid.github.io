import React from "react";
import { Container } from "react-bootstrap";
import { AiFillGithub } from "react-icons/ai";

function Footer() {
  const year = new Date().getFullYear();

  return (
    <Container fluid className="footer">
      <div className="footer-inner">
        <p>Jacob Lamadrid | Data science, ML research, and scientific software</p>
        <div className="footer-links">
          <a
            href="https://github.com/j-lamadrid"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Jacob Lamadrid on GitHub"
          >
            <AiFillGithub />
          </a>
        </div>
        <p className="footer-year">&copy; {year}</p>
      </div>
    </Container>
  );
}

export default Footer;
