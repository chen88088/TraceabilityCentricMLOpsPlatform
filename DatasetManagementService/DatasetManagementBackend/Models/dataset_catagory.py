from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from database import Base

# 分类表
class DatasetCategory(Base):
    __tablename__ = "dataset_categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)  # 数据集分类名称
    crop_type = Column(String(50))                           # 作物类型
    region = Column(String(50))                              # 地区
    resolution = Column(String(20))                          # 分辨率
    channels = Column(Integer)                               # 通道数
    description = Column(Text, nullable=True)                # 描述
    created_at = Column(TIMESTAMP, server_default="now()")   # 创建时间
