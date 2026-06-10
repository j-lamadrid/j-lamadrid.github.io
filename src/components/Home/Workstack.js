import React from "react";
import { Col, Row } from "react-bootstrap";
import {
  SiNasa
} from "react-icons/si";
import centeneLogo from "../../Assets/CNC_BIG.D-ae819181.png";
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

  const wideImgStyle = {
    width: "100%",
    height: "auto",
    maxHeight: "90px",
    objectFit: "contain",
  };

  const iconStyle = {
    fontSize: "60px",
  };

  return (
    <Row style={{ justifyContent: "center", paddingBottom: "50px" }}>
      <Col xs={6} md={3} lg={2} className="tech-icons work-logo-tile">
        <div style={iconBoxStyle}>
          <img src={centeneLogo} alt="Centene Corporation Logo" style={wideImgStyle} />
        </div>
      </Col>
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
