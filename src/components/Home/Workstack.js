import React from "react";
import { Col, Row } from "react-bootstrap";
import {
  SiNasa
} from "react-icons/si";
import {
  DiAws,
  DiGoogleCloudPlatform,
  DiMicrosoftazure,
} from "react-icons/di";
import ucsdHealthLogo from "../../Assets/image.png";
import brainLogo from "../../Assets/image (1).png";


function Workstack() {
  return (
    <Row style={{ justifyContent: "center", paddingBottom: "50px" }}>
      <Col xs={4} md={2} className="tech-icons">
        <SiNasa />
      </Col>
      <Col xs={4} md={2} className="tech-icons">
        <img src={ucsdHealthLogo} alt="UCSD Health Logo" style={{ width: "100%" }} />
      </Col>
      <Col xs={4} md={2} className="tech-icons">
        <img src={brainLogo} alt="Brain Logo" style={{ width: "100%" }} />
      </Col>
    </Row>
  );
}

export default Workstack;
