import React from "react";
import Typewriter from "typewriter-effect";

function Type() {
  return (
    <Typewriter
      options={{
        strings: [
          "Analytics Graduate Student",
          "Machine Learning Researcher",
          "Scientific Software Developer",
          "Computational Neuroscientist",
          "Super Cool!",
        ],
        autoStart: true,
        loop: true,
        deleteSpeed: 30,
      }}
    />
  );
}

export default Type;
