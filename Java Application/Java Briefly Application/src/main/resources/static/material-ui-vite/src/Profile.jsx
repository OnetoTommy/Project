import * as React from "react";
import IframeResiser from "@iframe-resizer/react";

export default function Profile({ action }) {
    const actionMap = {
        signin: "/user/signin.html",
        signup: "/user/Register.html",
        profile: "/user/profile.html",
    };
    let src = actionMap[action] ? actionMap[action] : actionMap.profile;
    return (
        <IframeResiser
            license="GPLv3"
            src={src}
            style={{ width: "100%", height: "86vh", border: "none" }}
            waitForLoad
        />
    );
}
