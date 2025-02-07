import React from "react";
import Card from "react-bootstrap/Card";
import Button from "react-bootstrap/Button";
import { AiOutlineBook } from "react-icons/ai";
import { useNavigate } from "react-router-dom";
import { Link } from "react-router-dom";

function LearningCards(props) {
  const navigate = useNavigate();

  return (
    <Card className="project-card-view">
      <Card.Img variant="top" src={props.imgPath} alt="card-img" />
      <Card.Body>
        <Card.Title>{props.title}</Card.Title>
        <Card.Text style={{ textAlign: "justify" }}>
          {props.description}
        </Card.Text>
        <Link to={props.link} className="btn btn-primary">
          <AiOutlineBook /> &nbsp;
          {props.isBlog ? "Blog" : "Notebooks"}
        </Link>
      </Card.Body>
    </Card>
  );
}

export default LearningCards;
