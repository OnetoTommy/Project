// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-analytics.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";
import { getFirestore, collection, query, where, getDocs } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";


// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
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

// Add event listener to the submit button
document.getElementById("submit").addEventListener("click", async function(event) {
    event.preventDefault();

    const identifier = document.getElementById("username").value.trim(); // Username or Email
    const password = document.getElementById("password").value.trim();

    try {
        let email = identifier;

        // Check if the identifier is not an email (assume it's a username)
        if (!identifier.includes("@")) {
            const usersCollection = collection(db, "users");
            const usernameQuery = query(usersCollection, where("username", "==", identifier));
            const querySnapshot = await getDocs(usernameQuery);

            if (!querySnapshot.empty) {
                email = querySnapshot.docs[0].data().email; // Resolve username to email
            } else {
                throw new Error("Username not found.");
            }
        }

        // Sign in using Firebase Authentication
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Store username and email in cookies
        const thirtyOneDays = 3600 * 24 * 31; // Calculate max-age  
        document.cookie = `username=${identifier}; path=/; max-age=${thirtyOneDays}`;
        document.cookie = `email=${user.email}; path=/; max-age=${thirtyOneDays}`;

        alert(`Welcome back, ${user.displayName || identifier}!`);
        window.location.href = "./profile.html"; // Redirect to the profile page
    } catch (error) {
        console.error("Error during sign-in:", error.message);
        alert(error.message);
    }
});