import React, { useState } from "react";
import { createVersion } from "../api/datasets";

export default function AddVersionPage() {
  const [categoryName, setCategoryName] = useState("");
  const [version, setVersion] = useState("");
  const [description, setDescription] = useState("");
  const [features, setFeatures] = useState("");
  const [updateScope, setUpdateScope] = useState("");
  const [cropType, setCropType] = useState("");
  const [region, setRegion] = useState("");
  const [dvcRemoteStorageUrl, setDvcRemoteStorageUrl] = useState("");
  const [dvcFileRepoUrl, setDvcFileRepoUrl] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    const data = {
      version,
      description,
      features,
      update_scope: updateScope,
      crop_type: cropType,
      region,
      dvc_remote_storage_url: dvcRemoteStorageUrl,
      dvc_file_repo_url: dvcFileRepoUrl,
    };

    createVersion(categoryName, data)
      .then(() => {
        alert("版本添加成功！");
        setCategoryName("");
        setVersion("");
        setDescription("");
        setFeatures("");
        setUpdateScope("");
        setCropType("");
        setRegion("");
        setDvcRemoteStorageUrl("");
        setDvcFileRepoUrl("");
      })
      .catch((error) => alert(`添加失敗: ${error.message}`));
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h1>添加版本</h1>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "15px" }}>
        <div style={formGroupStyle}>
          <label style={labelStyle}>分類名稱:</label>
          <input
            style={inputStyle}
            type="text"
            value={categoryName}
            onChange={(e) => setCategoryName(e.target.value)}
            required
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>版本號:</label>
          <input
            style={inputStyle}
            type="text"
            value={version}
            onChange={(e) => setVersion(e.target.value)}
            required
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>版本描述:</label>
          <textarea
            style={{ ...inputStyle, height: "100px" }}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>特徵:</label>
          <input
            style={inputStyle}
            type="text"
            value={features}
            onChange={(e) => setFeatures(e.target.value)}
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>更新範圍:</label>
          <input
            style={inputStyle}
            type="text"
            value={updateScope}
            onChange={(e) => setUpdateScope(e.target.value)}
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>作物類型:</label>
          <input
            style={inputStyle}
            type="text"
            value={cropType}
            onChange={(e) => setCropType(e.target.value)}
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>地區:</label>
          <input
            style={inputStyle}
            type="text"
            value={region}
            onChange={(e) => setRegion(e.target.value)}
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>DVC 遠程存儲 URL:</label>
          <input
            style={inputStyle}
            type="text"
            value={dvcRemoteStorageUrl}
            onChange={(e) => setDvcRemoteStorageUrl(e.target.value)}
          />
        </div>
        <div style={formGroupStyle}>
          <label style={labelStyle}>DVC 文件倉庫 URL:</label>
          <input
            style={inputStyle}
            type="text"
            value={dvcFileRepoUrl}
            onChange={(e) => setDvcFileRepoUrl(e.target.value)}
          />
        </div>
        <button type="submit" style={submitButtonStyle}>提交</button>
      </form>
    </div>
  );
}

const formGroupStyle = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
};

const labelStyle = {
  width: "150px",
  textAlign: "center",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
};

const inputStyle = {
  flex: "1",
  padding: "8px",
  borderRadius: "4px",
  border: "1px solid #ccc",
};

const submitButtonStyle = {
  padding: "10px 20px",
  backgroundColor: "#6200ea",
  color: "#fff",
  border: "none",
  borderRadius: "5px",
  cursor: "pointer",
};
