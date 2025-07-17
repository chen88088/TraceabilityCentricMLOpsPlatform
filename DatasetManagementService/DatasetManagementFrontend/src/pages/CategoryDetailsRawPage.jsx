import React, { useEffect, useState } from "react";
import { fetchCategories } from "../api/datasets";

export default function CategoryDetailsPage() {
  const [categories, setCategories] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCategories()
      .then((response) => setCategories(response.data))
      .catch((error) => setError(error.message));
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1 style={{ textAlign: "center", marginBottom: "20px" }}>分類列表</h1>
      {error && <div style={{ color: "red", textAlign: "center" }}>無法獲取分類列表: {error}</div>}
      {!error && categories.length > 0 ? (
        <div style={gridStyle}>
          {categories.map((category) => (
            <div key={category.id} style={cardStyle}>
              <h2 style={{ borderBottom: "1px solid #ccc", paddingBottom: "10px" }}>{category.name}</h2>
              <p><strong>作物類型:</strong> {category.crop_type}</p>
              <p><strong>地區:</strong> {category.region}</p>
              <p><strong>解析度:</strong> {category.resolution}</p>
              <p><strong>通道數:</strong> {category.channels}</p>
              <p><strong>描述:</strong> {category.description}</p>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ textAlign: "center" }}>沒有可用的分類資料。</div>
      )}
    </div>
  );
}

const gridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
  gap: "20px",
};

const cardStyle = {
  border: "1px solid #ddd",
  borderRadius: "10px",
  padding: "20px",
  boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
  backgroundColor: "#fff",
};
