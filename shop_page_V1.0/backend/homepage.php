<?php

session_start();
include("connect.php");
?>

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Homepage</title>
</head>
<body>
<div style="text-align: center; padding: 15%;">
  <p style="font-size: 50px; font-weight: bold;">
    Hello 
    <?php
    session_start(); // Make sure the session is started

    if (isset($_SESSION['email'])) {
        $email = $_SESSION['email'];

        // Query user info by email
        $query = mysqli_query($conn, "SELECT users.* FROM `users` WHERE email = '$email'");

        while ($row = mysqli_fetch_array($query)) {
            echo $row['firstName'] . ' ' . $row['lastName']; // Display full name
        }
    } else {
        echo "Guest";
    }
    ?>
  </p>
</div>
</body>
</html>