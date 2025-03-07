import { useNavigate } from "react-router-dom";

function getCookie(cname) {
    let name = cname + "=";
    let ca = document.cookie.split(";");
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) == " ") {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}

export const useUser = () => {
    const navigate = useNavigate();
    let username = getCookie("username");
    //console.log("username", username);
    if (!username) {
        navigate("/signin");
        //window.location.href = "/user/signin";
        return { id: null };
    }
    return { id: username };
};
