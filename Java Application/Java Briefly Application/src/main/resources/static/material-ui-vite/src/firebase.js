// src/utils/firebase.js

// Import Firebase SDKs
import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBNAape2cd-P6sKJKYApDANtxQoEL6YKQI",
  authDomain: "brieflydata.firebaseapp.com",
  projectId: "brieflydata",
  storageBucket: "brieflydata.firebasestorage.app",
  messagingSenderId: "1039911457450",
  appId: "1:1039911457450:web:2b1a06ddeeecfd05d6b6d3",
  measurementId: "G-P4ZLK2MDTF",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Function for signing in
export const loginWithEmailAndPassword = async (email, password) => {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    return userCredential.user; // Returns the authenticated user
  } catch (error) {
    throw error; // Pass error to the caller
  }
};
