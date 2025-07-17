import React from "react";
import { Button, Stack } from "@mui/material";
import { useNavigate } from "react-router-dom";

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div style={{ padding: "20px" }}>
      <h1>Welcome to the Dataset Management System</h1>
      <Stack spacing={2} direction="column">
        <Button
          variant="contained"
          color="primary" // 主色调按钮
          onClick={() => navigate("/categories")}
        >
          查看資料集列表
        </Button>
        <Button
          variant="contained"
          color="secondary" // 副色调按钮
          onClick={() => navigate("/add-category")}
        >
          添加分類
        </Button>
        <Button
          variant="contained"
          style={{ backgroundColor: "#ff9800", color: "#fff" }} // 自定义橙色按钮
          onClick={() => navigate("/versions")}
        >
          查看版本詳情
        </Button>
        <Button
          variant="contained"
          style={{ backgroundColor: "#4caf50", color: "#fff" }} // 自定义绿色按钮
          onClick={() => navigate("/add-version")}
        >
          添加版本
        </Button>
      </Stack>
    </div>
  );
}
