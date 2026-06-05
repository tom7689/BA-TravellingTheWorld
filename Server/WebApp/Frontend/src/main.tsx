import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import "./index.css";
import GameMap from "./GameMap";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/main" replace />} />
        <Route path="/main" element={<GameMap dbVariant="main" title="Travel The World" />} />
        <Route path="/efficient" element={<GameMap dbVariant="efficient" title="Travel The World (Efficient)" />} />
        <Route path="/efficient-refined" element={<GameMap dbVariant="efficient_refined" title="Travel The World (Efficient Refined)" />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
