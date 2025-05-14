import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../style/signup.css";

const Login = () => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };
  const navigate = useNavigate();
  const handleLogin = async (e) => {
    e.preventDefault();
    const data = new FormData();
    data.append("action", "signIn");
    data.append("email", formData.email);
    data.append("password", formData.password);

    try {
      const res = await fetch(
        "https://backend-1-7ie7.onrender.com/register.php",
        {
          method: "POST",
          body: data,
          credentials: "include",
        }
      );
      const result = await res.json();
      setMessage(result.message);

      if (result.status === "success") {
        // window.location.href = "http://localhost/shopping_ai/backend/homepage.php";
        navigate("/");
      }
    } catch (error) {
      setMessage("Login failed.");
    }
  };

  return (
    <>
      <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css"
      />
      <div className="auth-wrapper">
        <div className="sign-container">
          <h1 className="form-title">Sign In</h1>
          <form onSubmit={handleLogin}>
            <div className="input-group">
              <i className="fas fa-envelope"></i>
              <input
                type="email"
                name="email"
                placeholder="Email"
                required
                value={formData.email}
                onChange={handleChange}
              />
              <label for="email">Email</label>
            </div>
            <div className="input-group">
              <i className="fas fa-lock"></i>
              <input
                type="password"
                name="password"
                placeholder="Password"
                required
                value={formData.password}
                onChange={handleChange}
              />
              <label for="password">Password</label>
            </div>
            <input
              type="submit"
              name="signIn"
              id="signIn"
              class="sign_btn"
              value="Sign In"
            />
          </form>
          <p className="or">----------or-----------</p>
          <div class="icons">
            <i class="fab fa-google"></i>
            <i class="fab fa-facebook"></i>
          </div>
          <div className="links">
            <p>Don't Have Account ?</p>
            <a href="/register" >Sign Up</a>
          </div>
          {message && <div className="message-box">{message}</div>}
        </div>
      </div>
    </>
  );
};

export default Login;
