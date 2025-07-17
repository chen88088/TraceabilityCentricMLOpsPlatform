from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# 分类的 Pydantic 模型
class DatasetCategoryBase(BaseModel):
    name: str
    crop_type: Optional[str] = None
    region: Optional[str] = None
    resolution: Optional[str] = None
    channels: Optional[int] = None
    description: Optional[str] = None

class DatasetCategoryCreate(DatasetCategoryBase):
    pass

class DatasetCategoryResponse(DatasetCategoryBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class DatasetCategoryUpdate(BaseModel):
    name: str = None
    crop_type: str = None
    region: str = None
    resolution: str = None
    channels: int = None
    description: str = None

# 版本的 Pydantic 模型
class DatasetVersionBase(BaseModel):
    # dataset_name: str
    version: str
    crop_type: Optional[str] = None
    region: Optional[str] = None
    dvc_remote_storage_url: Optional[str] = None
    dvc_file_repo_url: Optional[str] = None
    update_scope: Optional[str] = None
    features: Optional[str] = None

class DatasetVersionCreate(DatasetVersionBase):
    pass

class DatasetVersionResponse(DatasetVersionBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class DatasetVersionUpdate(BaseModel):
    crop_type: str = None
    region: str = None
    dvc_remote_storage_url: str = None
    dvc_file_repo_url: str = None
    update_scope: str = None
    features: str = None