
import React, { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  IconButton,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import TopBar from "./Components/TopBar";

const FeedbackPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { feedback } = location.state || {};

  // Redirect to home if no feedback is available
  if (!feedback) {
    navigate("/");
    return null;
  }

  // State for dialog
  const [isDialogOpen, setDialogOpen] = useState(false);

  // Handlers for dialog actions
  const handleOpenDialog = () => setDialogOpen(true);
  const handleCloseDialog = () => setDialogOpen(false);
  const handleConfirm = () => {
    setDialogOpen(false);
    navigate(`/summary/${feedback.articleId}`); // Proceed to "Try Again"
  };

  // Navigate back to the home page
  const handleClosePage = () => {
    navigate("/today"); // Replace "/today" with the appropriate route for the home page
  };

  return (
    <Box sx={{ maxWidth: 800, margin: "auto", padding: 3 }}>
        <Box
            sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 2,
            }}
        >
        <TopBar title="Feedback" />
        <IconButton onClick={handleClosePage}>
            <CloseIcon />
            </IconButton>
        </Box>
        
      <Card sx={{ marginTop: 3, padding: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Score: {feedback.score}/10
          </Typography>

          <Box sx={{ whiteSpace: "pre-wrap" }}>
            {feedback.feedback.split("\n").map((line, index) => {
              const parts = line.split(":");
              if (parts.length > 1) {
                // Split into criteria and feedback text
                return (
                  <React.Fragment key={index}>
                    <Typography variant="body1" style={{ fontWeight: "bold" }}>
                      {parts[0].trim()}
                    </Typography>
                    <Typography variant="body1">{parts[1].trim()}</Typography>
                  </React.Fragment>
                );
              } else {
                // Handle lines without a colon
                return (
                  <Typography variant="body1" key={index}>
                    {line.trim()}
                    <br />
                  </Typography>
                );
              }
            })}
          </Box>

          {/* Button to trigger confirmation dialog */}
          <Button
            variant="outlined"
            onClick={handleOpenDialog}
            sx={{ marginTop: 2 }}
          >
            Try Again
          </Button>

          {/* Confirmation Dialog */}
          <Dialog
            open={isDialogOpen}
            onClose={handleCloseDialog}
            aria-labelledby="alert-dialog-title"
            aria-describedby="alert-dialog-description"
          >
            <DialogTitle id="alert-dialog-title">{"Confirm Submission"}</DialogTitle>
            <DialogContent>
              <DialogContentText id="alert-dialog-description">
                Submitting a new summary will overwrite your previous score and feedback. Are you sure you want to continue?
              </DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDialog} color="secondary">
                Cancel
              </Button>
              <Button onClick={handleConfirm} color="primary" autoFocus>
                Confirm
              </Button>
            </DialogActions>
          </Dialog>
        </CardContent>
      </Card>
    </Box>
  );
};

export default FeedbackPage;
