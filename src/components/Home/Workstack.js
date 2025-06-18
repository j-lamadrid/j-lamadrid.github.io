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
  const iconBoxStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "140px", // fixed height for all boxes
    width: "100%",   // take full width of Col
  };

  const imgStyle = {
    width: "100%",
    height: "auto",
  };

  const iconStyle = {
    fontSize: "60px",
  };

  return (
    <Row style={{ justifyContent: "center", paddingBottom: "50px" }}>
      <Col xs={4} md={2} className="tech-icons">
        <div style={iconBoxStyle}>
          <SiNasa style={iconStyle} />
        </div>
      </Col>
      <Col xs={4} md={2} className="tech-icons">
        <div style={iconBoxStyle}>
          <img src={ucsdHealthLogo} alt="UCSD Health Logo" style={imgStyle} />
        </div>
      </Col>
      <Col xs={4} md={2} className="tech-icons">
        <div style={iconBoxStyle}>
          <img src={brainLogo} alt="Brain Logo" style={imgStyle} />
        </div>
      </Col>
    </Row>
  );
}

export default Workstack;
