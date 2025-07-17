from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from Models.dataset_catagory import DatasetCategory
from schemas import DatasetCategoryCreate, DatasetCategoryResponse, DatasetCategoryUpdate
from typing import List
from sqlalchemy.sql import text

router = APIRouter()


# 创建新的分类
@router.post("/categories", response_model=DatasetCategoryResponse)
def create_category(category: DatasetCategoryCreate, db: Session = Depends(get_db)):
    # 检查是否存在
    existing_category = db.query(DatasetCategory).filter(DatasetCategory.name == category.name).first()
    if existing_category:
        raise HTTPException(status_code=400, detail="Category already exists")

    # 创建分类
    new_category = DatasetCategory(**category.dict())
    db.add(new_category)
    db.commit()

    # 动态创建对应的版本表
    table_name = f"{category.name.lower().replace(' ', '_')}_versions"
    create_table_sql = text(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        dataset_id INTEGER REFERENCES dataset_categories(id) ON DELETE CASCADE,
        dataset_name VARCHAR(255),
        version VARCHAR(50) NOT NULL,
        crop_type VARCHAR(50),
        region VARCHAR(50),
        dvc_remote_storage_url VARCHAR(255),
        dvc_file_repo_url VARCHAR(255),
        update_scope VARCHAR(255),
        features TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT {table_name}_unique_dataset_version UNIQUE (dataset_id, version) -- 添加唯一约束
    );
    """)
    db.execute(create_table_sql)
    db.commit()

    return new_category

# 获取所有分类
@router.get("/categories", response_model=List[DatasetCategoryResponse])
def get_all_categories(db: Session = Depends(get_db)):
    categories = db.query(DatasetCategory).all()
    return categories

#　查詢特定 dataset Category
@router.get("/categories/{category_name}", response_model=DatasetCategoryResponse)
def get_category(category_name: str, db: Session = Depends(get_db)):
    # 查找分類
    category = db.query(DatasetCategory).filter(DatasetCategory.name == category_name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category

#　刪除分類或版本
@router.delete("/categories/{category_name}")
def delete_category(category_name: str, db: Session = Depends(get_db)):
    # 查找分類
    category = db.query(DatasetCategory).filter(DatasetCategory.name == category_name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # 刪除分類以及相關版本表
    table_name = f"{category.name.lower().replace(' ', '_')}_versions"
    drop_table_sql = text(f"DROP TABLE IF EXISTS {table_name}")
    db.execute(drop_table_sql)
    
    # 刪除分類記錄
    db.delete(category)
    db.commit()
    return {"message": "Category and related versions deleted successfully"}

#　更新分類或版本
@router.patch("/categories/{category_name}")
def update_category(category_name: str, update_data: DatasetCategoryUpdate, db: Session = Depends(get_db)):
    # 查找分類
    category = db.query(DatasetCategory).filter(DatasetCategory.name == category_name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # 更新分類資料
    for key, value in update_data.dict(exclude_unset=True).items():
        setattr(category, key, value)
    db.commit()
    db.refresh(category)
    return category
