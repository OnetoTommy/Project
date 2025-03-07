import React from 'react';
import './scene.css';
import { loginWithEmailAndPassword } from './firebase.js';


const Signin = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
  
    const handleSubmit = async (e) => {
      e.preventDefault();
      try {
        const user = await loginWithEmailAndPassword(email, username, password);
        alert('Logging in...');
        // Redirect to another page
        window.location.href = '/Today.jsx'; // Update to a React route if using React Router
      } catch (error) {
        setErrorMessage(error.message);
      }
    };

  return (
    <div>
      <h2>Briefly</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="email">Username:</label>
          <input
            type="email"
            placeholder="Email or Phone"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <br />
        <div>
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
          />
        </div>
        <br />
        {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
        <div className="FP">
          <a href="#">Forgot password?</a>
        </div>
        <br />
        <div>
          <label>
            By signing in, you agree to our{' '}
            <a href="#">Terms of Service</a> and{' '}
            <a href="#">Privacy Policy</a>.
          </label>
        </div>
        <div>
          <button id="submit" type="submit">
            Log In
          </button>
        </div>
        <div>
          <label>
            <a href="./Register.html" target="_blank">
              Create a new account
            </a>
          </label>
        </div>
      </form>
    </div>
  );
};

export default Signin;
