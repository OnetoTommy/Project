import ApiClient from "../Services/ApiClient";
import { useQuery, useQueries } from "@tanstack/react-query";
import { useUser } from "./UseUser";

const scoreApi = new ApiClient("/summaries/submissionCounts");

export const useScores = () => {
    const { id } = useUser();
    /*
    const [getLastWeek, getLastYear] = useQueries({
        queries: [
            {
                queryKey: ["scores", "lastWeek"],
                queryFn: scoreApi.queryFn({ path: "/lastWeek" }),
            },
            {
                queryKey: ["scores", "lastYear"],
                queryFn: scoreApi.queryFn({ path: "/lastYear" }),
            },
        ],
    });

    return { getLastWeek, getLastYear };
    */

    const getScores = () =>
        useQuery({
            queryKey: ["scores"],
            queryFn: scoreApi.queryFn({ params: { userId: id } }),
        });
    return { getScores };
};
