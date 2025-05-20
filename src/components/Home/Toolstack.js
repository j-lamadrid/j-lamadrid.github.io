import React from "react";
import { Col, Row } from "react-bootstrap";
import {
  SiVisualstudiocode,
  SiPostman,
  SiSlack,
  SiVercel,
  SiMacos,
  SiPycharm,
  SiIntellijidea,
  SiAbletonlive,
  SiWindows,
  SiJupyter,
  SiMicrosoftazure
} from "react-icons/si";
import {
  DiAws,
  DiGoogleCloudPlatform,
  DiMicrosoftazure,
} from "react-icons/di";

function Toolstack() {
  return (
    <Row
      className="justify-content-center"
      style={{ paddingBottom: "50px" }}
    >
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiWindows />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiVisualstudiocode />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiPycharm />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiIntellijidea />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <DiAws />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiMicrosoftazure />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiAbletonlive />
      </Col>
    </Row>
  );
}

export default Toolstack;
