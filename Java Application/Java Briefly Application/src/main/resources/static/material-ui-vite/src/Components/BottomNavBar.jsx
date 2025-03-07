import * as React from "react";
import BottomNavigation from "@mui/material/BottomNavigation";
import BottomNavigationAction from "@mui/material/BottomNavigationAction";

import HomeIcon from "@mui/icons-material/Home";
import HistoryIcon from "@mui/icons-material/History";
import PersonIcon from "@mui/icons-material/Person";
import { useLocation, useNavigate, Link } from "react-router-dom";

export default function BottomNavBar({ currentPath }) {
    const location = useLocation();
    currentPath = currentPath || location.pathname.toLowerCase();

    currentPath = currentPath.replace(/^\//, "");
    //console.log("currentPath:", currentPath);
    const map = {
        today: 0,
        history: 1,
        profile: 2,
        signin: 2,
    };

    if (!currentPath) {
        currentPath = "today";
    }
    currentPath = currentPath.toLowerCase();

    const [value, setValue] = React.useState(map[currentPath] || 0);

    return (
        <BottomNavigation
            showLabels
            value={value}
            onChange={(event, newValue) => {
                setValue(newValue);
            }}
        >
            <BottomNavigationAction
                component={Link}
                to="/today"
                label="Today"
                icon={<HomeIcon />}
            />
            <BottomNavigationAction
                component={Link}
                to="/history"
                label="History"
                icon={<HistoryIcon />}
            />
            <BottomNavigationAction
                component={Link}
                to="/profile"
                label="Profile"
                icon={<PersonIcon />}
            />
        </BottomNavigation>
    );
}
