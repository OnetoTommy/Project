import React from "react";
import { NavLink } from "react-router-dom"
import { useSelector } from "react-redux";
import "../style/Navbar.css"


const Navbar = () => {
  // const state = useSelector((state) => state.handleCart);
  const cart = useSelector((state) => state.handleCart);
  const totalItems = cart.reduce((acc, item) => acc + item.qty, 0);

  return (
    <div>
      {/* <nav class="navbar navbar-expand-lg bg-white py-3 shadow-sm">
        <div class="container">
          <a class="navbar-brand fw-bold fs-4" href="/Navbar">
            LA COLLECTION
          </a>

          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarSupportedContent">
            <ul class="navbar-nav mx-auto mb-2 mb-lg-0">
              <li class="nav-item">
                <a class="nav-link active" aria-current="page" href="/">Home</a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="/products">Products</a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="/About">About</a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="/AI">AI</a>
              </li>
            </ul>
            <div className="buttons">
              <a href="/login" className="btn btn-outline-dark"><i className="fa fa-sign-in me-1"></i>Login</a>
              <a href="/register" className="btn btn-outline-dark ms-2"><i className="fa fa-user-plus me-1"></i>Register</a>
              <a href="/cart" className="btn btn-outline-dark ms-2"><i className="fa fa-shopping-cart me-1"></i>Cart({totalItems})</a>
            </div>
          </div>
        </div>
      </nav> */}

      <nav className="navbar navbar-expand-lg bg-white py-3 shadow-sm">
        <div className="container">
          {/* LOGO */}
          <a className="navbar-brand fw-bold fs-4" href="/">LA COLLECTION</a>

          {/* Hamburger Toggle */}
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>

          {/* Collapsible Content */}
          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav mx-auto mb-2 mb-lg-0">
              <li className="nav-item">
                <a className="nav-link active" aria-current="page" href="/">Home</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="/products">Products</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="/About">About</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="/AI">AI</a>
              </li>
            </ul>

            {/* Button Group */}
            <div className="d-flex gap-2">
              <a href="/login" className="btn btn-outline-dark">
                <i className="fa fa-sign-in me-1"></i>Login
              </a>
              <a href="/register" className="btn btn-outline-dark">
                <i className="fa fa-user-plus me-1"></i>Register
              </a>
              <a href="/cart" className="btn btn-outline-dark">
                <i className="fa fa-shopping-cart me-1"></i>Cart ({totalItems})
              </a>
            </div>
          </div>
        </div>
      </nav>




    </div>

  )
}

export default Navbar;