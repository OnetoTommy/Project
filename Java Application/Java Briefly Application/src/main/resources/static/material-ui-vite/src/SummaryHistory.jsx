import * as React from "react";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import Divider from "@mui/material/Divider";
import ListItemText from "@mui/material/ListItemText";
import ListItemAvatar from "@mui/material/ListItemAvatar";
import Avatar from "@mui/material/Avatar";
import Typography from "@mui/material/Typography";
import TopBar from "./Components/TopBar";
import { Button, Container, Rating } from "@mui/material";
import { useSummaryHistory } from "./Hooks/UseSummaryHistory";
import { useNavigate } from "react-router-dom";

function HistoryItem({ cover, title, date, rate, feedback, userSummary, id }) {
    const navigate = useNavigate();
    return (
        <ListItem alignItems="flex-start">
            <ListItemAvatar>
                <Avatar
                    alt="Remy Sharp"
                    variant="square"
                    src={cover}
                    sx={{ width: 40, height: 40 }}
                />
            </ListItemAvatar>
            <ListItemText
                primary={title}
                secondary={
                    <>
                        <Typography
                            variant="body2"
                            color="text.primary"
                            gutterBottom
                        >
                            Summary: {userSummary || "No summary provided"}
                        </Typography>
                        <Typography
                            variant="body2"
                            color="text.secondary"
                            gutterBottom
                        >
                            Feedback: {feedback || "No feedback yet"}
                        </Typography>

                        <Rating
                            name="half-rating-read"
                            defaultValue={rate}
                            precision={0.5}
                            readOnly
                        />
                        <span
                            style={{
                                display: "block",
                                float: "right",
                                padding: 4,
                                marginRight: 4,
                            }}
                        >
                            {date}
                        </span>
                    </>
                }
            />
        </ListItem>
    );
}

export default function SummaryHistory() {
    const { getList } = useSummaryHistory();
    const {
        data,
        error,
        fetchNextPage,
        hasNextPage,
        isFetching,
        isFetchingNextPage,
        status,
    } = getList();

    if (data == undefined) {
        return <></>;
    }

    if (!data) {
        return (
            <>
                <TopBar title="History" />
                <Container>
                    <List
                        sx={{
                            width: "100%",
                            bgcolor: "background.paper",
                        }}
                    >
                        <ListItem>No records</ListItem>
                    </List>
                </Container>
            </>
        );
    }
    return (
        <>
            <TopBar title="History" />
            <Container>
                <List
                    sx={{
                        width: "100%",
                        bgcolor: "background.paper",
                    }}
                >
                    {data.pages.map((group, i) => (
                        <React.Fragment key={i}>
                            {group?.items?.map(({ summary, article }, j) => (
                                <React.Fragment key={j}>
                                    <HistoryItem
                                        key={summary?.id}
                                        cover={article?.poster}
                                        title={article?.title}
                                        date={article?.date.substring(0, 10)}
                                        rate={summary?.score / 2}
                                        feedback={summary?.feedback}
                                        userSummary={summary?.userSummary}
                                        id={summary?.id}
                                    />
                                    <Divider component="li" />
                                </React.Fragment>
                            ))}
                            {!!group.timeline && (
                                <Button onClick={() => fetchNextPage()}>
                                    show more
                                </Button>
                            )}
                        </React.Fragment>
                    ))}
                </List>
            </Container>
        </>
    );
}
