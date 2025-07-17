from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from schemas import DatasetVersionCreate, DatasetVersionResponse, DatasetVersionUpdate
from typing import List
from sqlalchemy.sql import text  # 引入 text 方法
from Models.dataset_catagory import DatasetCategory
from sqlalchemy.engine import Row
from sqlalchemy.exc import IntegrityError

router = APIRouter()


# 创建指定分类的版本
@router.post("/{category_name}/versions", response_model=DatasetVersionResponse)
def create_version(category_name: str, version_data: DatasetVersionCreate, db: Session = Depends(get_db)):
    # 查找分类是否存在
    category = db.query(DatasetCategory).filter(DatasetCategory.name == category_name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # 动态表名
    table_name = f"{category_name.lower().replace(' ', '_')}_versions"

    # 准备数据
    data_to_insert = version_data.dict()
    data_to_insert["dataset_id"] = category.id
    data_to_insert["dataset_name"] = category_name

    insert_sql = text(f"""
    INSERT INTO {table_name} 
    (dataset_id, dataset_name, version, crop_type, region, dvc_remote_storage_url, dvc_file_repo_url, update_scope, features)
    VALUES (:dataset_id, :dataset_name, :version, :crop_type, :region, :dvc_remote_storage_url, :dvc_file_repo_url, :update_scope, :features)
    RETURNING *;
    """)
    
    try:
        # 执行插入操作
        result = db.execute(insert_sql, data_to_insert)
        db.commit()

        # 读取插入结果
        row = result.fetchone()
        if row:
            return {key: value for key, value in zip(result.keys(), row)}
        else:
            raise HTTPException(status_code=500, detail="Failed to insert version.")

    except IntegrityError as e:
        db.rollback()

        # 检查是否是唯一约束冲突
        if "unique" in str(e.orig).lower():
            raise HTTPException(status_code=400, detail="This version already exists for the category.")

        # 其他异常情况
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# 获取指定分类的所有版本
@router.get("/{category_name}/versions", response_model=List[DatasetVersionResponse])
def get_versions(category_name: str, db: Session = Depends(get_db)):
    # 查找对应分类
    category = db.query(DatasetCategory).filter(DatasetCategory.name == category_name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # 动态表名
    table_name = f"{category_name.lower().replace(' ', '_')}_versions"
    
    # 查询版本信息
    query_sql = text( f"SELECT * FROM {table_name};")
    results = db.execute(query_sql).fetchall()

    # 转换 Row 对象为字典
    return [dict(row._mapping) for row in results]

#　查詢特定版本資訊
@router.get("/{category_name}/versions/{version}")
def get_version(category_name: str, version: str, db: Session = Depends(get_db)):
    # 確認分類是否存在
    category_table_name = f"{category_name.lower().replace(' ', '_')}_versions"
    query_sql = text(f"SELECT * FROM {category_table_name} WHERE version = :version")
    try:
        result = db.execute(query_sql, {"version": version}).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Version not found for the specified category")
        # 使用 `_mapping` 將 Row 對象轉換為字典
        return dict(result._mapping)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# 刪除指定版本
@router.delete("/{category_name}/versions/{version}")
def delete_version(category_name: str, version: str, db: Session = Depends(get_db)):
    # 確認分類是否存在
    table_name = f"{category_name.lower().replace(' ', '_')}_versions"
    delete_sql = text(f"DELETE FROM {table_name} WHERE version = :version RETURNING *")
    result = db.execute(delete_sql, {"version": version})
    db.commit()
    if not result.rowcount:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"message": "Version deleted successfully"}

# 更新指定版本資訊
@router.patch("/{category_name}/versions/{version}", response_model=DatasetVersionResponse)
def update_version(
    category_name: str,
    version: str,
    updates: DatasetVersionUpdate,  # Ensure updates use Pydantic model
    db: Session = Depends(get_db),
):
    # Check if the category exists
    category = db.query(DatasetCategory).filter(DatasetCategory.name == category_name).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Dynamic table name
    table_name = f"{category_name.lower().replace(' ', '_')}_versions"

    # Build update SQL dynamically
    update_fields = updates.model_dump(exclude_unset=True)  # Use model_dump instead of dict
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    update_clauses = ", ".join([f"{key} = :{key}" for key in update_fields.keys()])
    update_sql = text(f"""
    UPDATE {table_name}
    SET {update_clauses}
    WHERE version = :version
    RETURNING *;
    """)

    # Prepare update data
    update_data = update_fields
    update_data["version"] = version

    try:
        # Execute update operation
        result = db.execute(update_sql, update_data).fetchone()
        db.commit()

        if not result:
            raise HTTPException(status_code=404, detail="Version not found")

        # Return the updated record
        return dict(result._mapping)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")