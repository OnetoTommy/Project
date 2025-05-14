import React from 'react';
import './App.css';  
import Navbar from './component/Navbar';
import Home from './component/Home';
import Products from './component/Products';
import Product from './component/Product';
import Cart from './component/Cart'; 
import AI from './component/AI'
import Login from './component/Login';
import About from './component/About';
import Register from './component/Register';
import MainLayout from "./layouts/MainLayout";
import BlankLayout from "./layouts/BlankLayout";
import { Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';

function App() {
  return (
    <>
    <Routes>

      <Route element={<MainLayout />}>
        <Route path="/" element={<Home />} />
        <Route path="/products" element={<Products />} />
        <Route path="/products/:id" element={<Product />} />
        <Route path="/cart" element={<Cart />} />
        <Route path="/AI" element={<AI />} />
        <Route path="/About" element={<About />} />
      </Route>
      

      
      <Route element={<BlankLayout />}>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
      </Route>
    </Routes>
    <ToastContainer position="top-right" autoClose={2000} />
    </>
  );
}

export default App;


// <>
    //   <Navbar />
    //   <Routes>
    //     <Route path="/" element={<Home />} />
    //     <Route path="/products" element={<Products />} />
    //     <Route path="/products/:id" element={<Product />} />
    //     <Route path="/cart" element={<Cart />} /> {/* Add route for Cart */}
    //     <Route path="/AI" element={<AI />} />
    //     <Route path="/login" element={<Login />} />
    //     <Route path="/register" element={<Register />} />
    //   </Routes>
    // </>