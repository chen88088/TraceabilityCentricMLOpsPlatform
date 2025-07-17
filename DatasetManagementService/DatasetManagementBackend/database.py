from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 修正後的 SQLALCHEMY_DATABASE_URL
SQLALCHEMY_DATABASE_URL = "postgresql://dataset_admin:dataset_password@dataset_management_service_db:5432/dataset_management_service_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

