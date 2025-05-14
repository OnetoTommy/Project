import React from "react";
import AboutImg1 from './img/about-1.png';
import AboutImg2 from './img/about-2.png';
import "../style/about.css";

const About = () => {

  return(
    <div className="About">
      <div className="About_section"> 
        <img src={AboutImg1} alt="About-1" />
      </div>
      
      <div className="About_section"> 
        <img src={AboutImg2} alt="About-2" />
      </div>
    </div>
  )

}

export default About;