// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";
import { getFirestore, doc, setDoc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyBNAape2cd-P6sKJKYApDANtxQoEL6YKQI",
    authDomain: "brieflydata.firebaseapp.com",
    projectId: "brieflydata",
    storageBucket: "brieflydata.firebasestorage.app",
    messagingSenderId: "1039911457450",
    appId: "1:1039911457450:web:2b1a06ddeeecfd05d6b6d3",
    measurementId: "G-P4ZLK2MDTF"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app); // Initialize Firestore

// Submit Button Event Listener
const submit = document.getElementById("submit");
submit.addEventListener("click", function(event) {
    event.preventDefault();

    const email = document.getElementById("email").value;
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    createUserWithEmailAndPassword(auth, email, password)
        .then(async(userCredential) => {
            // User successfully created
            const user = userCredential.user;

            // Set displayName
            await updateProfile(user, {
                displayName: username,
            });

            // Save user data to Firestore
            try {
                await setDoc(doc(db, "users", user.uid), {
                    username: username,
                    email: email,
                    createdAt: new Date().toISOString()
                });
                alert("Account created and user data saved to Firestore!");
                window.location.href = "http://127.0.0.1:5500/src/main/resources/static/post/registration%20copy.html";
            } catch (error) {
                console.error("Error saving user data to Firestore:", error);
                alert("Error saving user data. Please try again.");
            }
        })
        .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            console.error("Error creating user:", error);
            alert(errorMessage);
        });
});