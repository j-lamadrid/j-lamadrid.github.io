import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Particle from "../Particle";
import About from "./About";
import Type from "./Type";

function Home() {
  return (
    <section>
      <Container fluid className="home-section" id="home">
        <Particle />
        <Container className="home-content">
          <Row>
            <Col md={7} className="home-header">
              <h1 style={{ paddingBottom: 15 }} className="heading">
                Hello!{" "}
                <span className="cat" role="img" aria-labelledby="cat">
                  <img
                    src="https://nukochannel.neocities.org/NukoImg/Banners/Activities/nukoBreadcrumbs.gif"
                    alt="cat"
                    style={{ width: "80px" }}
                  />
                </span>
              </h1>
              <h1 className="heading-name">
                I'm<strong className="main-name"> Jacob Lamadrid</strong>
              </h1>

              <div style={{ padding: 50, textAlign: "left" }}>
                <Type />
              </div>
            </Col>

            <Col md={5} style={{ paddingBottom: 20 }}>
              <img
                src='https://nukochannel.neocities.org/NukoImg/Stickers/nukoStickerIRLCat.gif'
                alt="home pic"
                style={{ width: "400px", paddingTop: 50 }}
              />
            </Col>
          </Row>
        </Container>
      </Container>
      <About />
    </section>
  );
}

export default Home;
