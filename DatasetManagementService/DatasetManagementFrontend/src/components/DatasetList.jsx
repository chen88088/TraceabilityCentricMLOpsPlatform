import React, { useEffect, useState } from "react";
import { fetchCategories, deleteCategory } from "../api/datasets";

function DatasetList({ onSelect }) {
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    fetchCategories()
      .then((res) => setCategories(res.data))
      .catch((err) => console.error(err));
  }, []);

  const handleDelete = (id) => {
    if (window.confirm("Are you sure you want to delete this category?")) {
      deleteCategory(id)
        .then(() => {
          setCategories(categories.filter((cat) => cat.id !== id));
        })
        .catch((err) => console.error(err));
    }
  };

  return (
    <div>
      <h2>Dataset Categories</h2>
      <ul>
        {categories.map((category) => (
          <li key={category.id}>
            {category.name}{" "}
            <button onClick={() => onSelect(category)}>View</button>
            <button onClick={() => handleDelete(category.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default DatasetList;
