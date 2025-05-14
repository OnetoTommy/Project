import React from "react";
import homeImg from './img/home.png';
import Products from "./Products";
import "../style/Home.css";

const Home = () => {

  return (
    <div className="hero">
      <div className="card text-bg-dark border-0">
        <img src={homeImg} className="card-img" alt="Backgroud" />
          <div className="card-img-overlay d-flex flex-column justify-content-around">
            <div className="container">
              <h1 className="home-title">NEW SEASON ARRIVALS</h1>
              <p className="home-text">CHECK OUT ALL THE TRENDS</p>
{/* 
              <h5 className="card-title fw-bolder mb-0 display-6 display-md-4 display-lg-3">
                NEW SEASON ARRIVALS
              </h5>
              <p className="card-text lead fs-6 fs-md-5 fs-lg-2">
                CHECK OUT ALL THE TRENDS
              </p> */}
            </div>
          </div>
      </div>
      <Products />
    </div>
  );
}

export default Home;