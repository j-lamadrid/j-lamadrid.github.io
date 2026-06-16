import React from "react";
import Card from "react-bootstrap/Card";
import Button from "react-bootstrap/Button";
import { AiOutlineFileText } from "react-icons/ai";
import { BsGithub } from "react-icons/bs";
import { FiExternalLink } from "react-icons/fi";
import { Link } from "react-router-dom";

function isInternalLink(url) {
  return typeof url === "string" && url.startsWith("/") && !url.startsWith("//");
}

function ProjectCards(props) {
  const links = props.links || {
    github: props.ghLink,
    report: props.demoLink,
  };
  const stack = props.stack || [];
  const hasImage = Boolean(props.image || props.imgPath);

  return (
    <Card
      className={`project-card-view project-card-text-only ${
        props.featured ? "project-card-featured" : ""
      }`}
    >
      {hasImage && (
        <div className="project-image-wrap">
          <Card.Img
            variant="top"
            src={props.image || props.imgPath}
            alt={`${props.title} project preview`}
          />
        </div>
      )}
      <Card.Body className="project-card-body">
        <div className="project-card-meta">
          {props.type && <span>{props.type}</span>}
          {props.domain && <span>{props.domain}</span>}
        </div>
        <Card.Title>{props.title}</Card.Title>
        <Card.Text>{props.summary || props.description}</Card.Text>

        {stack.length > 0 && (
          <div className="project-stack" aria-label={`${props.title} tools`}>
            {stack.map((tool) => (
              <span key={tool}>{tool}</span>
            ))}
          </div>
        )}

        {props.impact && <p className="project-impact">{props.impact}</p>}

        <div className="project-actions">
          {links.github && (
            <Button
              variant="primary"
              href={links.github}
              target="_blank"
              rel="noopener noreferrer"
            >
              <BsGithub />
              <span>Code</span>
            </Button>
          )}

          {links.report && (
            <Button
              variant="primary"
              href={links.report}
              target="_blank"
              rel="noopener noreferrer"
            >
              <AiOutlineFileText />
              <span>Report</span>
            </Button>
          )}

          {links.demo && (
            <Button
              variant="primary"
              {...(isInternalLink(links.demo)
                ? { as: Link, to: links.demo }
                : { href: links.demo, target: "_blank", rel: "noopener noreferrer" })}
            >
              <FiExternalLink />
              <span>{props.demoLabel || "Demo"}</span>
            </Button>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}

export default ProjectCards;
