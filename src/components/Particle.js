import React from "react";
import Particles from "react-tsparticles";

function Particle() {
  return (
    <Particles
      id="tsparticles"
      params={{
        particles: {
          number: {
            value: 400,
            density: {
              enable: true,
              value_area: 800,
            },
          },
          shape: {
            type: "square",
          },
          line_linked: {
            enable: true,
            opacity: 0.025,
          },
          move: {
            direction: "right",
            speed: 0.05,
          },
          size: {
            value: 3,
          },
          opacity: {
            value: 0.5,
            anim: {
              enable: true,
              speed: 1,
              opacity_min: 0.05,
            },  
          },
        },
        interactivity: {
          events: {
            onhover: {
              enable: true,
              mode: "push",
            },
          },
          modes: {
            push: {
              quantity: 50,
            },
            repulse: {
              distance: 100,
              duration: 2,
            },
          },
        },
        retina_detect: true,
      }}
    />
  );
}

export default Particle;
