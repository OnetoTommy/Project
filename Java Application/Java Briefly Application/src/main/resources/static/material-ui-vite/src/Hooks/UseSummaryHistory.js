import { useInfiniteQuery } from "@tanstack/react-query";
import ApiClient from "../Services/ApiClient";
import { useUser } from "./UseUser";

const historyApi = new ApiClient("/history");

export const useSummaryHistory = () => {
    const { id } = useUser();

    const fetchProjects = async ({ pageParam }) => {
        const res = await fetch("/data/history/list.json?cursor=" + pageParam);
        return res.json();
    };

    const getList = () =>
        useInfiniteQuery({
            queryKey: ["history"],
            queryFn: historyApi.queryFn({ path: "/list", params: { uid: id } }),
            initialPageParam: {},
            getNextPageParam: (lastPage, pages) => {
                return { timeline: lastPage.timeline };
            },
        });

    return { getList };
};
