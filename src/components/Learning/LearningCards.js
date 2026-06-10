import React from "react";
import Card from "react-bootstrap/Card";
import { AiOutlineBook } from "react-icons/ai";
import { Link } from "react-router-dom";

function LearningCards(props) {
  const hasExternalLink = props.link && /^https?:\/\//.test(props.link);
  const linkLabel = props.cta || props.status || "Open guide";

  return (
    <Card className="project-card-view project-card-text-only learning-card-view">
      <Card.Body className="project-card-body">
        <div className="project-card-meta">
          <span>{props.status || "Guide"}</span>
        </div>
        <Card.Title>{props.title}</Card.Title>
        <Card.Text>{props.description}</Card.Text>

        {props.tags && (
          <div className="project-stack" aria-label={`${props.title} concepts`}>
            {props.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        )}

        <div className="project-actions">
          {props.link && hasExternalLink && (
            <a
              href={props.link}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
            >
              <AiOutlineBook />
              <span>{linkLabel}</span>
            </a>
          )}

          {props.link && !hasExternalLink && (
            <Link to={props.link} className="btn btn-primary">
              <AiOutlineBook />
              <span>{linkLabel}</span>
            </Link>
          )}

          {!props.link && (
            <span className="btn btn-primary disabled learning-card-status">
              <AiOutlineBook />
              <span>{linkLabel}</span>
            </span>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}

export default LearningCards;
