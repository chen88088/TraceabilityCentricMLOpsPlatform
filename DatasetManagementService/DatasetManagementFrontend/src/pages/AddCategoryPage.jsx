import React, { useState } from "react";
import { createCategory } from "../api/datasets";

export default function AddCategoryPage() {
  const [name, setName] = useState("");
  const [cropType, setCropType] = useState("");
  const [region, setRegion] = useState("");
  const [resolution, setResolution] = useState("");
  const [channels, setChannels] = useState(0);
  const [description, setDescription] = useState("");

  const [errors, setErrors] = useState({});

  const validateInputs = () => {
    const newErrors = {};
    if (!name.trim()) newErrors.name = "分類名稱不可為空";
    if (!cropType.trim()) newErrors.cropType = "作物類型不可為空";
    if (!region.trim()) newErrors.region = "地區不可為空";
    if (!resolution.match(/^\d+:\d+$/)) newErrors.resolution = "解析度格式應為 '數字:數字'";
    if (channels <= 0) newErrors.channels = "通道數應為大於 0 的正整數";
    if (!description.trim()) newErrors.description = "描述不可為空";
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!validateInputs()) return;

    const data = { name, crop_type: cropType, region, resolution, channels, description };

    createCategory(data)
      .then(() => {
        alert("分類添加成功！");
        setName("");
        setCropType("");
        setRegion("");
        setResolution("");
        setChannels(0);
        setDescription("");
        setErrors({});
      })
      .catch((error) => alert(`添加失敗: ${error.message}`));
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>新增資料集種類</h1>
      <form onSubmit={handleSubmit} style={{ maxWidth: "600px", margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <label style={{ flex: "0 0 150px" }}>資料集名稱:</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{ flex: "1", padding: "8px" }}
            required
          />
          {errors.name && <p style={{ color: "red", marginLeft: "10px" }}>{errors.name}</p>}
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <label style={{ flex: "0 0 150px" }}>作物類型:</label>
          <input
            type="text"
            value={cropType}
            onChange={(e) => setCropType(e.target.value)}
            style={{ flex: "1", padding: "8px" }}
            required
          />
          {errors.cropType && <p style={{ color: "red", marginLeft: "10px" }}>{errors.cropType}</p>}
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <label style={{ flex: "0 0 150px" }}>地區:</label>
          <input
            type="text"
            value={region}
            onChange={(e) => setRegion(e.target.value)}
            style={{ flex: "1", padding: "8px" }}
            required
          />
          {errors.region && <p style={{ color: "red", marginLeft: "10px" }}>{errors.region}</p>}
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <label style={{ flex: "0 0 150px" }}>解析度:</label>
          <input
            type="text"
            value={resolution}
            onChange={(e) => setResolution(e.target.value)}
            style={{ flex: "1", padding: "8px" }}
            required
          />
          {errors.resolution && (
            <p style={{ color: "red", marginLeft: "10px" }}>{errors.resolution}</p>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <label style={{ flex: "0 0 150px" }}>通道數:</label>
          <input
            type="number"
            value={channels}
            onChange={(e) => setChannels(Number(e.target.value))}
            style={{ flex: "1", padding: "8px" }}
            required
          />
          {errors.channels && (
            <p style={{ color: "red", marginLeft: "10px" }}>{errors.channels}</p>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
          <label style={{ flex: "0 0 150px" }}>描述:</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            style={{ flex: "1", padding: "8px", resize: "vertical" }}
            required
          />
          {errors.description && (
            <p style={{ color: "red", marginLeft: "10px" }}>{errors.description}</p>
          )}
        </div>
        <button
          type="submit"
          style={{
            backgroundColor: "#6200ea",
            color: "#fff",
            padding: "10px 20px",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          提交
        </button>
      </form>
    </div>
  );
}
