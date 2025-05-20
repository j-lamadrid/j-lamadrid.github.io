import React from "react";
import { Col, Row } from "react-bootstrap";
import { CgCPlusPlus } from "react-icons/cg";
import {
  DiReact,
  DiNodejs,
  DiMongodb,
  DiPython,
  DiGit,
  DiJava,
  DiAnaconda,
  DiAws,
  DiIntellij,
  DiMysql,
  DiRStudio,
  DiPostgresql,
  DiDart,
} from "react-icons/di";
import {
  SiRedis,
  SiFirebase,
  SiNextdotjs,
  SiSolidity,
  SiPostgresql,
  SiRstudio,
  SiFlutter,
  SiPandas,
  SiPytorch,
  SiScikitlearn,
  SiTensorflow,
  SiArduino,
  SiApachespark,
  SiNasa,
} from "react-icons/si";
import { TbBrandGolang } from "react-icons/tb";

function Techstack() {
  return (
    <Row className="justify-content-center" style={{ paddingBottom: "50px" }}>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <DiPython />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiPandas />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiPytorch />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiTensorflow />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiScikitlearn />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiApachespark />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiRstudio />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <DiPostgresql />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <DiDart />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiFlutter />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <DiReact />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <CgCPlusPlus />
      </Col>
      <Col xs={4} md={2} className="d-flex justify-content-center tech-icons">
        <SiArduino />
      </Col>
    </Row>
  );
}

export default Techstack;
