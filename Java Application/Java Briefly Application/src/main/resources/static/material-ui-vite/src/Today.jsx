import { Box, Button, Chip, Container, Stack } from "@mui/material";
import * as React from "react";
import SendIcon from "@mui/icons-material/Send";
import WhatshotIcon from "@mui/icons-material/Whatshot";
import StarIcon from "@mui/icons-material/Star";
import TopBar from "./Components/TopBar";
import { useArticles } from "./Hooks/UseArticles";
import { Link } from "react-router-dom";
import { useScores } from "./Hooks/UseScores";

function TextWithLineBreaks(props) {
    //console.log("TextWithLineBreaks", props);
    if (!props.children) {
        return <></>;
    }
    const textWithBreaks = props.children
        .split("\n")
        .map((text, index) => <p key={index}>{text}</p>);

    return <div>{textWithBreaks}</div>;
}

export default function Today() {
    const { getLatest } = useArticles();

    const { isPending, error, data, isFetching } = getLatest();

    /*
    const { getById } = useArticles();

    const { isPending, error, data, isFetching } = getById(
       "jpbc5sqcbq776xopaVyv"
    );
    */
    let article = {
        id: "",
        title: "",
        poster: "",
        content: "",
    };
    if (data) {
        article = data;
    }
    if (isPending || isFetching) {
        article.title = "loading...";
    }
    if (error) {
        article.title = error.message;
    }
    /*
    const { getLastWeek, getLastYear } = useScores();
    let scores = {};
    if (getLastWeek.data && getLastWeek.data) {
        scores = {
            lastWeek: getLastWeek?.data?.score,
            lastYear: getLastYear?.data?.score,
        };
    }
*/
    const { getScores } = useScores();
    let { data: scores } = getScores();
    if (!scores) {
        scores = {
            last7Days: "-",
            last365Days: "-",
        };
    }
    return (
        <>
            <TopBar title="Today">
                <Stack direction="row" spacing={1}>
                    {scores && (
                        <>
                            <Chip
                                icon={<WhatshotIcon />}
                                label={"" + scores?.last7Days}
                                color="primary"
                                size="small"
                            />
                            <Chip
                                icon={<StarIcon />}
                                label={"" + scores?.last365Days}
                                color="success"
                                size="small"
                            />
                        </>
                    )}
                </Stack>
            </TopBar>
            <Container>
                <article>
                    <h1>{article?.title || ""}</h1>
                    <img src={article?.poster || ""} width={"100%"}></img>
                    <TextWithLineBreaks>
                        {article?.content || ""}
                    </TextWithLineBreaks>
                </article>
                <Stack
                    direction="row"
                    spacing={2}
                    sx={{
                        justifyContent: "center",
                        alignItems: "center",
                    }}
                >
                    <Box></Box>
                    <Box>
                        <Button
                            variant="contained"
                            component={Link}
                            to={`/summary/${article?.id}`} // Pass article ID in the URL
                            // to={`/summary?id=${article.id}`}
                            endIcon={<SendIcon />}
                        >
                            Try Summarize
                        </Button>
                    </Box>
                </Stack>
            </Container>
        </>
    );
}
