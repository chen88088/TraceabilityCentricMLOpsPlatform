import React from "react";
import { Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CategoryDetailsPage from "./pages/CategoryDetailsPage";
import AddCategoryPage from "./pages/AddCategoryPage";
import AddVersionPage from "./pages/AddVersionPage";
import VersionDetailsPage from "./pages/VersionDetailsPage";

export default function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/categories" element={<CategoryDetailsPage />} />
      <Route path="/add-category" element={<AddCategoryPage />} />
      <Route path="/versions" element={<VersionDetailsPage />} />
      <Route path="/add-version" element={<AddVersionPage />} />
    </Routes>
  );
}
