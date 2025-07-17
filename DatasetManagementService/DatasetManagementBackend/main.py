from fastapi import FastAPI
from Routers import Router_for_datasets, Router_for_dataset_versions
from database import Base, engine
from fastapi.middleware.cors import CORSMiddleware

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 配置 (放在路由註冊之前)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://dataset_management_service_frontend:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(Router_for_datasets.router, prefix="/datasets", tags=["Dataset Categories"])
app.include_router(Router_for_dataset_versions.router, prefix="/datasets", tags=["Versions and Info of Centain Dataset"])
