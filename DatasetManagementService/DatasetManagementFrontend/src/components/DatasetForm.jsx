import React, { useState } from "react";
import { createCategory } from "../api/datasets";

function DatasetForm({ onSuccess }) {
  const [formData, setFormData] = useState({
    name: "",
    crop_type: "",
    region: "",
    resolution: "",
    channels: 0,
    description: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    createCategory(formData)
      .then(() => {
        alert("Category created successfully!");
        onSuccess();
      })
      .catch((err) => console.error(err));
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="name" placeholder="Name" onChange={handleChange} required />
      <input name="crop_type" placeholder="Crop Type" onChange={handleChange} />
      <input name="region" placeholder="Region" onChange={handleChange} />
      <input name="resolution" placeholder="Resolution" onChange={handleChange} />
      <input name="channels" placeholder="Channels" type="number" onChange={handleChange} />
      <textarea name="description" placeholder="Description" onChange={handleChange} />
      <button type="submit">Create</button>
    </form>
  );
}

export default DatasetForm;
