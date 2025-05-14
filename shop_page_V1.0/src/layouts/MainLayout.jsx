import { Outlet } from "react-router-dom";
import Navbar from "../component/Navbar";

const MainLayout = () => (
  <>
    <Navbar />
    <Outlet />
  </>
);

export default MainLayout;
