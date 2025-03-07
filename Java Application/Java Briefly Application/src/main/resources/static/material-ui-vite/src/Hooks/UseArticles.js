import ApiClient from "../Services/ApiClient";
import { useQuery } from "@tanstack/react-query";

const articleApi = new ApiClient("/articles");

export const useArticles = () => {
    const getLatest = () =>
        useQuery({
            queryKey: ["articles", "latest"],
            queryFn: articleApi.queryFn({ path: "/latest" }),
        });

    const getById = (id) =>
        useQuery({
            queryKey: ["article", id],
            queryFn: articleApi.queryFn({ path: `/id/${id}` }), // Adjusted path
            enabled: !!id, // Fetch only if `id` exists
        });

    return { getLatest, getById };
};


