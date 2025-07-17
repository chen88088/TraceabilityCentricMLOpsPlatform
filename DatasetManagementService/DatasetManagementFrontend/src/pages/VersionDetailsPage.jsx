import React, { useState } from "react";
import { fetchVersions } from "../api/datasets";

export default function VersionDetailsPage() {
  const [categoryName, setCategoryName] = useState("");
  const [versions, setVersions] = useState([]);
  const [error, setError] = useState(null);

  const handleFetchVersions = () => {
    fetchVersions(categoryName)
      .then((response) => {
        setVersions(response.data);
        setError(null);
      })
      .catch((error) => setError(error.message));
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>版本詳情</h1>
      <div style={{ marginBottom: "20px" }}>
        <label style={{ marginRight: "10px" }}>分類名稱:</label>
        <input
          type="text"
          value={categoryName}
          onChange={(e) => setCategoryName(e.target.value)}
          style={{ padding: "8px", marginRight: "10px" }}
        />
        <button
          onClick={handleFetchVersions}
          style={{
            padding: "8px 15px",
            backgroundColor: "#6200ea",
            color: "#fff",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          查詢
        </button>
      </div>
      {error && <div style={{ color: "red", marginBottom: "20px" }}>獲取版本失敗: {error}</div>}
      {versions.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "20px" }}>
          <thead>
            <tr>
              <th style={tableHeaderStyle}>版本號</th>
              <th style={tableHeaderStyle}>作物種類</th>
              <th style={tableHeaderStyle}>地區</th>
              <th style={tableHeaderStyle}>特徵</th>
              <th style={tableHeaderStyle}>更新範圍</th>
              {/* <th style={tableHeaderStyle}>存儲地址</th> */}
              <th style={tableHeaderStyle}>倉庫地址</th>
              <th style={tableHeaderStyle}>創建時間</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((version) => (
              <tr key={version.id} style={tableRowStyle}>
                <td style={tableCellStyle}>{version.version}</td>
                <td style={tableCellStyle}>{version.crop_type}</td>
                <td style={tableCellStyle}>{version.region}</td>
                <td style={tableCellStyle}>{version.features}</td>
                <td style={tableCellStyle}>{version.update_scope}</td>
                {/* <td style={tableCellStyle}>{version.dvc_remote_storage_url}</td> */}
                <td style={tableCellStyle}>
                  <a href={version.dvc_file_repo_url} target="_blank" rel="noopener noreferrer">
                    {version.dvc_file_repo_url}
                  </a>
                </td>
                <td style={tableCellStyle}>{new Date(version.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {versions.length === 0 && !error && <div>尚無版本資訊。</div>}
    </div>
  );
}

const tableHeaderStyle = {
  borderBottom: "2px solid #ddd",
  textAlign: "left",
  padding: "10px",
  backgroundColor: "#f4f4f4",
};

const tableRowStyle = {
  borderBottom: "1px solid #ddd",
};

const tableCellStyle = {
  textAlign: "left",
  padding: "10px",
  verticalAlign: "top",
};
