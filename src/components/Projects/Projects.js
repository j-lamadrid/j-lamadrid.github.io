import React, { useState } from "react";
import { Container, Row, Col } from "react-bootstrap";
import ProjectCard from "./ProjectCards";
import Particle from "../Particle";
import projects, { projectFilters } from "./projectsData";

function Projects() {
  const [activeFilter, setActiveFilter] = useState("All");
  const filteredProjects =
    activeFilter === "All"
      ? projects
      : projects.filter((project) => project.type === activeFilter);
  const featuredProjects =
    activeFilter === "All"
      ? filteredProjects.filter((project) => project.featured)
      : [];
  const projectLibrary =
    activeFilter === "All"
      ? filteredProjects.filter((project) => !project.featured)
      : filteredProjects;

  return (
    <Container fluid className="project-section">
      <Particle />
      <Container className="project-shell">
        <section className="section-intro reveal-up">
          <p className="section-kicker">Selected Work</p>
          <h1 className="project-heading">
            Projects for <strong className="brown">data science</strong>, signal
            processing, and engineering.
          </h1>
          <p className="section-lede">
            A focused collection of research, applied ML, and software projects
            with tools, technical value, and source links surfaced up front.
          </p>

          <div className="project-filter" role="group" aria-label="Project filters">
            {projectFilters.map((filter) => (
              <button
                key={filter}
                type="button"
                className={filter === activeFilter ? "active" : ""}
                aria-pressed={filter === activeFilter}
                onClick={() => setActiveFilter(filter)}
              >
                {filter}
              </button>
            ))}
          </div>
        </section>

        {featuredProjects.length > 0 && (
          <Row className="featured-project-grid">
            {featuredProjects.map((project) => (
              <Col lg={6} className="project-card" key={project.title}>
                <ProjectCard {...project} />
              </Col>
            ))}
          </Row>
        )}

        <h2 className="project-subheading">
          {activeFilter === "All" ? "Project Library" : `${activeFilter} Projects`}
        </h2>
        <Row className="project-grid">
          {projectLibrary.map((project) => (
            <Col lg={4} md={6} className="project-card" key={project.title}>
              <ProjectCard {...project} />
            </Col>
          ))}
        </Row>
      </Container>
    </Container>
  );
}

export default Projects;
