import React, { useState, useEffect, useRef } from "react";
import {
    Box,
    Paper,
    Typography,
    TextField,
    Button,
    Snackbar,
    Alert,
} from "@mui/material";
import MicIcon from "@mui/icons-material/Mic";
import MicOffIcon from "@mui/icons-material/MicOff";
import { useNavigate, useParams } from "react-router-dom";

import TopBar from "./Components/TopBar";
import BottomNavBar from "./Components/BottomNavBar";
import { useUser } from "./Hooks/UseUser";

const SummaryComponent = () => {
    const [summary, setSummary] = useState("");
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const navigate = useNavigate();
    const { articleId } = useParams(); // Access article ID from URL
    // const { id: userId } = useUser(); // Retrieve user ID
    const recognitionRef = useRef(null); // Store recognition instance

    let recognition; // SpeechRecognition instance

    // Initialize SpeechRecognition
    useEffect(() => {
        const SpeechRecognition =
            window.SpeechRecognition || window.webkitSpeechRecognition;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = "en-US";

            let debounceTimeout;
            recognition.onresult = (event) => {
                clearTimeout(debounceTimeout);
                let finalTranscript = "";

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const result = event.results[i];
                    if (result.isFinal) {
                        finalTranscript += result[0].transcript + " ";
                    }
                }

                // Add punctuation only if there is non-empty content
                const punctuate = (text) => {
                    if (!text.trim()) {
                        return ""; // Return empty if there's no meaningful input
                    }

                    const words = text.trim().split(" ");

                    // Capitalize the first word of the sentence
                    words[0] =
                        words[0].charAt(0).toUpperCase() + words[0].slice(1);

                    let punctuatedText = words.join(" ");

                    // Add a period if the sentence doesn't end with punctuation
                    if (!/[.?!]$/.test(punctuatedText)) {
                        punctuatedText += ".";
                    }

                    return punctuatedText;
                };

                debounceTimeout = setTimeout(() => {
                    setSummary((prev) => {
                        const punctuatedText =
                            punctuate(finalTranscript).trim();

                        // Avoid adding multiple periods due to pauses
                        if (punctuatedText) {
                            return `${prev} ${punctuatedText}`.trim();
                        }
                        return prev; // If there's no valid input, keep the previous summary
                    });
                }, 300); // Update every 300ms
            };

            // recognition.onresult = (event) => {
            //     clearTimeout(debounceTimeout);
            //     let finalTranscript = "";

            //     for (let i = event.resultIndex; i < event.results.length; i++) {
            //         const result = event.results[i];
            //         if (result.isFinal) {
            //             finalTranscript += result[0].transcript + " ";
            //         }
            //     }

            //     debounceTimeout = setTimeout(() => {
            //         setSummary((prev) => `${prev} ${finalTranscript.trim()}`);
            //     }, 300); // Update every 300ms
            // };

            // Handle errors
            recognition.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
                setError("Error with speech recognition. Try again.");
                setIsListening(false);
            };

            recognition.onend = () => {
                setIsListening(false); // Reset listening state when recognition stops
            };
            recognitionRef.current = recognition; // Store instance in ref
        } else {
            console.warn("Speech recognition not supported in this browser.");
        }

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.abort(); // Cleanup
            }
        };
    }, []);

    const handleSpeechToggle = () => {
        const recognition = recognitionRef.current;
        if (!recognition) {
            setError("Speech recognition not supported in this browser.");
            return;
        }

        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
        }
        setIsListening(!isListening);
    };

    const { id: userId } = useUser(); // Retrieve user ID
    const handleSubmit = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch("/api/summaries/evaluate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    userId: userId, // Include userId in headers
                },
                body: JSON.stringify({
                    articleId,
                    userSummary: summary,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(
                    errorData.message || "Failed to evaluate summary"
                );
            }

            const result = await response.json();
            if (result && result.score != null && result.feedback) {
                navigate(`/feedback/${articleId}`, {
                    state: { feedback: result },
                });
            } else {
                throw new Error("Unexpected response structure from backend.");
            }
        } catch (error) {
            console.error("Error submitting summary:", error);
            setError(
                error.message ||
                    "An error occurred while evaluating your summary"
            );
        } finally {
            setLoading(false);
        }
    };

    const handleTryLater = () => {
        navigate("/today"); // Navigate to the desired page (e.g., "Today" page or home)
    };
    return (
        <Box sx={{ maxWidth: 800, margin: "auto", padding: 3 }}>
            <TopBar title="Summarize" />
            <Paper elevation={3} sx={{ padding: 3 }}>
                <Typography variant="h5" gutterBottom>
                    Write Your Summary
                </Typography>
                <TextField
                    fullWidth
                    multiline
                    rows={4}
                    value={summary}
                    onChange={(e) => setSummary(e.target.value)}
                    placeholder="Enter your summary here..."
                    sx={{ marginBottom: 2 }}
                />
                <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
                    <Button
                        variant="contained"
                        onClick={handleSubmit}
                        disabled={loading || !summary.trim()}
                    >
                        {loading ? "Submitting..." : "Submit Summary"}
                    </Button>
                    <Button variant="outlined" onClick={handleTryLater}>
                        Try Later
                    </Button>
                    <Button
                        variant="outlined"
                        onClick={handleSpeechToggle}
                        color={isListening ? "secondary" : "primary"}
                    >
                        {isListening ? (
                            <>
                                <MicOffIcon sx={{ marginRight: 1 }} />
                                Stop Listening
                            </>
                        ) : (
                            <>
                                <MicIcon sx={{ marginRight: 1 }} />
                                Start Listening
                            </>
                        )}
                    </Button>
                </Box>
            </Paper>
            <Snackbar
                open={!!error}
                autoHideDuration={6000}
                onClose={() => setError(null)}
            >
                <Alert onClose={() => setError(null)} severity="error">
                    {error}
                </Alert>
            </Snackbar>
            {/* <BottomNavBar /> */}
        </Box>
    );
};

export default SummaryComponent;
