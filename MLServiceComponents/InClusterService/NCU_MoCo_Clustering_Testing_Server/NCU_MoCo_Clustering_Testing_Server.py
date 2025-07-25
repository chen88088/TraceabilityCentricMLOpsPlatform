from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import logging
import requests
import subprocess
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os
import shutil
import git
from DVCManager import DVCManager
from DVCWorker import DVCWorker
from typing import Dict, Any
from LoggerManager import LoggerManager
from kubernetes import client, config
from DVCManager import DVCManager
from DVCWorker import DVCWorker
from typing import Dict
from LoggerManager import LoggerManager
from DagManager import DagManager
from fastapi.responses import JSONResponse
import yaml  # 解析 DVC 檔案
import shutil
import re  # 引入正則表達式套件
from jinja2 import Environment, FileSystemLoader, Template

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from minio import Minio
from minio.error import S3Error
import uuid

import time
import json
import hashlib

logging.basicConfig(level=logging.DEBUG)

SERVER_MANAGER_URL = "http://10.52.52.136:8000"

MACHINE_ID = "machine_server_1"
MACHINE_IP = "10.52.52.136"
MACHINE_PORT = 8085
MACHINE_CAPACITY = 2

class RegisterRequest(BaseModel):
    machine_id: str
    ip: str
    port: int
    capacity: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.debug("Entering lifespan startup")
    register_data = RegisterRequest(
        machine_id=MACHINE_ID,
        ip=MACHINE_IP,
        port=MACHINE_PORT,
        capacity=MACHINE_CAPACITY
    )

    try:
        # #　去跟servermanager註冊自己的ip與資源用量
        # logging.debug("Sending registration request to Server Manager")
        # response = requests.post(f"{SERVER_MANAGER_URL}/register", json=register_data.dict())
        # logging.debug(f"Registration request response status: {response.status_code}")
        # response.raise_for_status()
        logging.info("Registered with Server Manager successfully")
    except requests.RequestException as e:
        logging.error(f"Failed to register with Server Manager: {e}")
        logging.debug(f"Response: {e.response.text if e.response else 'No response'}")

    yield

    logging.debug("Entering lifespan shutdown")
    try:
        # #　去跟servermanager 清除自己的ip與資料
        # response = requests.post(f"{SERVER_MANAGER_URL}/cleanup", json=register_data.dict())
        # response.raise_for_status()
        logging.info("Resources cleaned up successfully")
    except requests.RequestException as e:
        logging.error(f"Failed to clean up resources: {e}")



app = FastAPI(lifespan=lifespan)

# PVC 掛載路徑
STORAGE_PATH = "/mnt/storage"

# # PVC 掛載路徑 (本地測試用)
# STORAGE_PATH = "/mnt/storage/test/test_clustering_testing"


# MinIO 設定
MINIO_URL = "10.52.52.138:31000"
MINIO_ACCESS_KEY = "testdvctominio"
MINIO_SECRET_KEY = "testdvctominio"
BUCKET_NAME = "mock-dataset"


# 定義 Request Body Schema
class CreateFolderRequest(BaseModel):
    dag_id: str
    execution_id: str

class DagRequest(BaseModel):
    DAG_ID: str
    EXECUTION_ID: str
    TASK_STAGE_TYPE: str
    DATASET_NAME: str
    DATASET_VERSION: str
    CODE_REPO_URL: Dict[str, str]
    IMAGE_NAME: Dict[str, str]
    MODEL_NAME: str
    MODEL_VERSION: str
    DEPLOYER_NAME: str
    DEPLOYER_EMAIL: str
    PIPELINE_CONFIG: Dict[str, Any]


# 檢查 PVC 是否已掛載
def is_pvc_mounted():
    return os.path.exists(STORAGE_PATH) and os.path.ismount(STORAGE_PATH)

# 創建資料夾（如果不存在）
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# MinIO 客戶端初始化
def init_minio_client():
    return Minio(
        MINIO_URL,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # 如果 MinIO 沒有 SSL，設置為 False
    )

# 解析 DVC 檔案並獲取 outs 路徑
def parse_dvc_file(dvc_file_path):
    with open(dvc_file_path, 'r') as file:
        dvc_data = yaml.safe_load(file)
        outs = dvc_data.get("outs", [])
        dataset_paths = [item['path'] for item in outs]
        return dataset_paths
    
# 初始化 Kubernetes 客戶端
def init_k8s_client():
    try:
        config.load_incluster_config()  # Kubernetes 內部環境
    except config.ConfigException:
        config.load_kube_config()  # 本機開發環境
    return client.BatchV1Api()

# 實例化 LoggerManager
logger_manager = LoggerManager()
# 實例化 DVCManger
dvc_manager = DVCManager(logger_manager)
# 實例化 DagManger
dag_manager = DagManager(logger_manager)

@app.get("/health", status_code=200)
async def health_check():
    return JSONResponse(content={"status": "NCU_MoCo_Clustering_Testing_MLServingPod is healthy"})

# [Clustering/RegisterDag]
@app.post("/Clustering/RegisterDag")
async def register_dag_and_logger_and_dvc_worker(request: DagRequest):
    """
    每隻 DAG 先來註冊，並生成專屬的 logger 與 DVC Worker
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 檢查 PVC 掛載狀態
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")

    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}
    dag_root_folder_path = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")
    git_local_repo_for_dvc = os.path.join(dag_root_folder_path, "GIT_LOCAL_REPO_FOR_DVC")

    # 創建根目錄
    create_folder_if_not_exists(dag_root_folder_path)

    # 確認 DAG 是否已在 dag_manager 中註冊
    if dag_manager.is_registered(dag_id, execution_id):
        return {"status": "success", "message": "DAG is already registered."}

    # 登記 DAG 到 dag_manager
    dag_manager.register_dag(dag_id, execution_id, dag_root_folder_path)

    # 初始化並註冊 logger
    if not logger_manager.logger_exists(dag_id, execution_id):
        logger_manager.init_logger(dag_id, execution_id, dag_root_folder_path)

    # 初始化並註冊 DVCWorker
    create_folder_if_not_exists(git_local_repo_for_dvc)
    dvc_manager.init_worker(dag_id=dag_id, execution_id=execution_id, git_repo_path=git_local_repo_for_dvc)

    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/RegisterDag")

    # **打印請求的 request body**
    request_dict = request.model_dump()
    logger.info(f"Received request body:\n{json.dumps(request_dict, indent=4)}")

    logger.info(f"DAG, Logger, and DVCWorker initialized for DAG_ID: {dag_id}, EXECUTION_ID: {execution_id}")

    return {"status": "success", "message": f"DAG, Logger, and DVCWorker initialized for DAG_ID: {dag_id}, EXECUTION_ID: {execution_id}"}

# [Clustering/SetupFolder]
@app.post("/Clustering/SetupFolder")
async def setup_folders_for_training(request: DagRequest):
    """
    為 Training 設定資料夾結構並 clone 相關程式碼
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/SetupFolder")
    
    dvc_worker = dvc_manager.get_worker(dag_id, execution_id)
    
    
    # 檢查 PVC 掛載狀態
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")
    logger.info("PVC IS MOUNTED!!!!")

    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}
    dag_root_folder_path = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")
    repo_inference_path = os.path.join(dag_root_folder_path, "NCU_MoCo_Clustering_Testing")

    # 確認 DAG 根目錄是否存在
    if not os.path.exists(dag_root_folder_path):
        raise HTTPException(status_code=404, detail="dag_root_folder was not created")


    # 從 CODE_REPO_URL 中獲取對應的 Repo URL
    code_repo_url = request.CODE_REPO_URL.get(task_stage_type)
    if not code_repo_url:
        raise HTTPException(status_code=400, detail="CODE_REPO_URL for the given TASK_STAGE_TYPE not found")

    try:
        logger.info(f"Received request for DAG_ID: {dag_id}, EXECUTION_ID: {execution_id}")

        # 1. 檢查GIT LOCAL REPO FOR DVC
        dvc_worker.ensure_git_repository()
        logger.info("Ensured Git repository for DVC")

        # 2. 在 dag_root_folder 内 git clone
        if not os.path.exists(repo_inference_path):
            # 取得環境變數中的 GITHUB_TOKEN
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                raise Exception("GITHUB_TOKEN not found in environment variables.")
            
            # 解析 repo 的 owner/repo_name
            if "github.com" in code_repo_url:
                repo_path = code_repo_url.split("github.com/")[-1]
            else:
                raise HTTPException(status_code=400, detail="Invalid GitHub repository URL")

            # # 使用 subprocess.run() 執行 git clone 指令
            # # **修正：直接內嵌 GITHUB_TOKEN 到 URL 中**
            # repo_url = f"https://{github_token}:x-oauth-basic@github.com/chen88088/NCU-RSS-1.5.git"
            repo_url = f"https://{github_token}:x-oauth-basic@github.com/{repo_path}"

                        
            # repo_url = code_repo_url

            clone_command = ["git", "clone", repo_url, repo_inference_path]
            result = subprocess.run(clone_command, capture_output=True, text=True)
            
            # 紀錄輸出與錯誤訊息
            logger.info(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")

            # 檢查返回碼
            if result.returncode != 0:
                raise Exception("git clone failed")
            
            # 驗證是否成功 clone
            assert os.path.exists(repo_inference_path), "Repo NCURSS-Training clone failed"
            logger.info(f"Cloned NCU_MoCo_Clustering_Training repo to {repo_inference_path}")
        else:
            logger.info(f"NCU_MoCo_Clustering_Training repo already exists at {repo_inference_path}")

        return {"status": "success",  "message": f"Cloned NCU_MoCo_Clustering_Training repo to {repo_inference_path}"}

    except HTTPException as e:
        logger.error(f"HTTPException occurred: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# [Clustering/DownloadPreprocessingResult]
@app.post("/Clustering/DownloadPreprocessingResult")
async def download_preprocessing_result(request: DagRequest):
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE  # e.g., "Clustering"

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")

    dag_root_folder_path = f"{dag_id}_{execution_id}"

    # 獲取對應的 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/DownloadPreprocessingResult")

    dvc_worker = dvc_manager.get_worker(dag_id, execution_id)

    # 檢查 PVC 掛載狀態
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")

    try:
        # 設定本地 DAG 路徑
        # 組合路徑：/mnt/storage/{dag_id}_{execution_id}
        dag_root_folder_path = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")


        # 步驟 1. 下載Preprocessing 階段的資料於 "Temp_Preprocessing_Result_Download"
        # 設定 Preprocessing 結果存放的 Repository 位置
        temp_preprocessing_result_download_path = os.path.join(dag_root_folder_path, "Temp_Preprocessing_Result_Download")

        # **確保目錄存在**
        os.makedirs(temp_preprocessing_result_download_path, exist_ok=True)

        # DVC 檔案名稱
        dvc_filename = "result.dvc"
        result_folder = "result"
        
        logger.info(f"Downloading {dvc_filename} from MinIO to {temp_preprocessing_result_download_path}")

        # 執行 pull 操作
        pull_result = dvc_worker.pull(
            stage_type="Preprocessing",  # 指定 stage
            dvc_filename=dvc_filename,  # 下載的 .dvc 文件
            folder_path=str(temp_preprocessing_result_download_path)  # 本地儲存路徑
        )

        if pull_result["status"] == "error":
            logger.error(f"Failed to pull {result_folder} from DVC: {pull_result['message']}")
            raise HTTPException(status_code=500, detail=pull_result["message"])

        # 確認數據是否成功下載
        local_result_path = Path(temp_preprocessing_result_download_path) / result_folder
        if not local_result_path.exists():
            logger.error(f"Result folder {local_result_path} does not exist after pulling from DVC.")
            raise HTTPException(status_code=500, detail="Downloaded result folder not found.")

        logger.info(f"Successfully downloaded preprocessing results for {dag_id}_{execution_id}.")

        # 步驟 2. 設定最終放置 Preprocessing 結果的目標目錄
        # source
        source_temp_download_files_path = Path(temp_preprocessing_result_download_path) / "result"
        
        # target (follow script instruction)
        repo_inference_path = os.path.join(dag_root_folder_path, "NCU_MoCo_Clustering_Testing")
        target_result_path = Path(repo_inference_path) / "MoCo_testing" /"data"/"new_data"

        # 確保資料夾存在
        target_result_path.mkdir(parents=True, exist_ok=True)
        source_temp_download_files_path.mkdir(parents=True, exist_ok=True)


        # 步驟 3: 直接重命名 temp_download 為最終目標資料夾
        logger.info(f"Renaming {source_temp_download_files_path} to {target_result_path}")

        # 確保目標資料夾不存在，否則 shutil.move 會報錯
        if target_result_path.exists():
            shutil.rmtree(target_result_path)  # 先刪除舊目錄，確保不衝突

        shutil.move(str(source_temp_download_files_path), str(target_result_path))

        logger.info(f"Successfully moved preprocessing results to {target_result_path}")

        return {
            "status": "success",
            "message": f"Preprocessing results '{result_folder}' successfully downloaded.",
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        return {"status": "error", "message": str(e)}, 500
    
# [Clustering/FetchModel]
@app.post("/Clustering/FetchModel")
async def fetch_model(request: DagRequest):
    
    # ***************************
    # **設定 MLflow Tracking Server**
    MLFLOW_TRACKING_URI = "http://10.52.52.142:5000"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # **設定 MinIO 作為 S3 儲存**
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.52.52.142:9000"
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    # ***************************

    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    model_name = request.MODEL_NAME
    model_version = int(request.MODEL_VERSION)

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    if not model_name or not model_version:
        raise HTTPException(status_code=400, detail="MODEL_NAME and MODEL_VERSION are required.")
    
    # 獲取對應的 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/FetchModel")

    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}
    dag_root_folder_path = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")
    repo_clusteruing_path = os.path.join(dag_root_folder_path, "NCU_MoCo_Clustering_Testing")

    # 設定下載模型的目標路徑
    model_download_path = Path(os.path.join(repo_clusteruing_path, "MoCo_testing/data/saved_model_and_prediction"))
    model_download_path.mkdir(parents=True, exist_ok=True)

    try:
        # **步驟 1：從 MLflow Model Registry 取得模型 URI**
        client = MlflowClient()
        model_version_info = client.get_model_version(model_name, model_version)
        artifact_uri = model_version_info.source  # 取得模型在 MinIO/S3 的存放路徑
        
        logger.info(f"Model: {model_name} version{model_version} URI: {artifact_uri}")

        # **步驟 2：下載模型**
        run_id = model_version_info.run_id
        download_uri = f"runs:/{run_id}/moco_model"
        download_path = mlflow.artifacts.download_artifacts(download_uri, dst_path=str(model_download_path))
        logger.info(f"Model downloaded to: {download_path}")

        return {
            "status": "success",
            "message": f"Model '{model_name}' version '{model_version}' successfully downloaded.",
            "path": str(download_path)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

# TODO: Config Modify
# [Clustering/ModifyClusteringTestingConfig]
@app.post("/Clustering/ModifyClusteringTestingConfig")
async def modify_preprocessing_config(request: DagRequest):
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 從 PIPELINE_CONFIG 中取出 clusters_amount（轉換為 int）
    clusters_amount = int(request.PIPELINE_CONFIG.get("clusters_amount", 4))


    # 獲取對應的 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/NCU_MoCo_Clustering_Testing")

    try:
        dag_root_folder =Path(os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")) 
        dag_root_folder_path = Path(dag_root_folder)
        repo_clustering_path = Path(dag_root_folder_path / "NCU_MoCo_Clustering_Testing")
        config_path = repo_clustering_path / "MoCo_testing" /"config.py"

        logger.info(f"Modifying config at {config_path} with request")

        # 读取并修改配置文件
        with open(config_path, "r", encoding='utf-8') as file:
            config_content = file.read()
        
        # 替換 saved_model_path 與 folder_path（僅替換 assignment 行）

        config_content = re.sub(
            r"^folder_path\s*=.*",
            "\nBASE_DIR = os.path.dirname(os.path.abspath(__file__))\nfolder_path = os.path.join(BASE_DIR, 'data', 'new_data')",
            config_content,
            flags=re.MULTILINE
        )

        config_content = re.sub(
            r"^saved_model_path\s*=.*",
            "saved_model_path = os.path.join(BASE_DIR, 'data', 'saved_model_and_prediction', 'moco_model')",
            config_content,
            flags=re.MULTILINE
        )

        # 使用 Jinja2 渲染 clusters_amount
        template = Template(config_content)
        rendered_config = template.render(clusters_amount=clusters_amount)
        
        with open(config_path, "w", encoding='utf-8') as file:
            file.write(rendered_config)

        logger.info(f"Config file {config_path} modified successfully")

        # 验证文件内容
        with open(config_path, "r", encoding='utf-8') as file:
            verified_content = file.read()

        assert "saved_model_path = os.path.join(BASE_DIR" in verified_content, "saved_model_path not updated"
        assert "folder_path = os.path.join(BASE_DIR" in verified_content, "folder_path not updated"
        # 注意 雖然目標config 該參數是要填數字 但因為  assert 是用python "in"  檢查 ，前後都要型別 ，所以要轉str() 
        assert str(clusters_amount) in verified_content, "clusters_amount not updated"

        logger.info(f"Config file {config_path} verification successful")
        
        logger.info(f"[Clustering]ModifyClusteringTestingConfig finished for dag: {dag_id}_{execution_id}   !!!")

    except Exception as e:
        logger.error(f"An error occurred during config modification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config modification failed: {str(e)}")

    return {"status": "success", "message": "Config modified successfully"}


# [Clustering/ExecuteClusteringTestingScripts]
@app.post("/Clustering/ExecuteClusteringTestingScripts")
async def execute_clustering_testing_scripts(request: DagRequest):
    """
    根據 TASK_STAGE_TYPE 起 Pod 並執行對應腳本
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    # 從 IMAGE_NAME 中抓取對應的 Image
    image_name = request.IMAGE_NAME.get(task_stage_type)
    if not image_name:
        raise HTTPException(status_code=400, detail=f"No image found for TASK_STAGE_TYPE: {task_stage_type}")
    
    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/ExecuteClusteringTestingScripts")
    

    # v1 = init_k8s_client()
    batch_v1 = init_k8s_client()  # 初始化 Batch API 用於創建 Job
    
    # # Pod 名稱
    # pod_name = f"{dag_id}-{execution_id}-{task_stage_type}-task-pod-{uuid.uuid4().hex[:6]}"

    # Job 名稱
    raw_job_name = f"task-job-{dag_id}-{execution_id}-{task_stage_type}-{uuid.uuid4().hex[:6]}"
    # 將 _ 換成 -，轉小寫，並移除不合法字元
    job_name = re.sub(r'[^a-z0-9\-]+', '', raw_job_name.lower().replace('_', '-'))
    
    # 長度限制處理（K8s label/name 最多 63 字元）
    if len(job_name) > 63:
        hash_suffix = hashlib.sha1(job_name.encode()).hexdigest()[:6]
        job_name = f"{job_name[:56]}-{hash_suffix}"
    
    # 確保 PVC_NAME 環境變量已設定
    pvc_name = os.getenv("PVC_NAME")
    if not pvc_name:
        raise HTTPException(status_code=500, detail="PVC_NAME not set in environment variables")
    
    # 組合工作目錄
    working_dir = f"/mnt/storage/{dag_id}_{execution_id}/NCU_MoCo_Clustering_Testing/MoCo_testing"
    
    output_dir = Path(f"{working_dir}/data/new_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定義要執行的腳本
    scripts_to_run = [
        "python parcel_moco_kmeans.py"
    ]

    # 組合成 Command
    command = " && ".join([f"cd {working_dir}"] + scripts_to_run )
    
    # **正確的 Job Manifest**
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "ml-serving",
            "labels": {
                "app": "task-job",
                "type": "gpu"
            }
        },
        "spec": {
            "backoffLimit": 3,
            # **TTLAfterFinished: 完成後 600 秒自動刪除**
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {
                    "name": job_name
                },
                "spec": {
                    "nodeSelector": {
                        "gpu-node": "true"
                    },
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "task-container",
                            "image": f"harbor.pdc.tw/{image_name}",
                            "command": ["/bin/bash", "-c", command],
                            "volumeMounts": [
                                {
                                    "name": "shared-storage",
                                    "mountPath": "/mnt/storage"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": "1"
                                }
                            }
                        }
                    ],
                    "volumes": [
                        {
                            "name": "shared-storage",
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        }
                    ]
                }
            }
        }
    }
    
    # 建立 Job
    try:
        logger.info("Job creating...........")
        batch_v1.create_namespaced_job(namespace="ml-serving", body=job_manifest)

        logger.info("Job execute start...........")
        # **等待 Job 完成**
        wait_for_job_completion(batch_v1, job_name, "ml-serving",logger)

        logger.info("Job finish!!!!!")

        record_clustering_testing_result_to_mlflow(request, output_dir, logger)
        logger.info("Record to MLflow Successfully!!!!")
        logger.info(f"Task Job created and finished . job_name {job_name}")
        return {"status": "success", "message": "Task Job created and finished", "job_name": job_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Task Job: {str(e)}")

def wait_for_job_completion(batch_v1, job_name, namespace, logger, timeout=3600):
    """
    監聽 K8s Job 狀態，等待 Job 執行完成
    """
    start_time = time.time()
    v1 = client.CoreV1Api()  # 初始化 K8s API
    pod_name = None  # 儲存 task pod 名稱
    while time.time() - start_time < timeout:
        job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
        if job_status.status.succeeded == 1:
            logger.info(f"Job {job_name} completed successfully.")
            
            # **讀取 Pod 名稱**
            pod_list = v1.list_namespaced_pod(namespace, label_selector=f"job-name={job_name}").items
            
            if pod_list:
                pod_name = pod_list[0].metadata.name
                logger.info(f"Job pod name: {pod_name}")

                # **獲取並記錄 Pod 的 logs**
                try:
                    logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
                    logger.info(f"Task Pod Logs for {job_name}:\n{logs}")
                except Exception as e:
                    logger.error(f"Failed to get logs for {pod_name}: {str(e)}")

                # # 嘗試關閉 Istio sidecar
                # try:
                #     logger.info("Attempting to shut down Istio sidecar...")
                #     cmd = [
                #         "kubectl", "exec", "-n", namespace, "-it", pod_name,
                #         "--", "python3", "-c",
                #         "import http.client; conn = http.client.HTTPConnection('127.0.0.1', 15020); "
                #         "conn.request('POST', '/quitquitquit'); print(conn.getresponse().status)"
                #     ]
                #     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                #     logger.info(f"Istio shutdown result: {result.stdout.strip()}")
                #     if result.stderr:
                #         logger.warning(f"Istio shutdown stderr: {result.stderr.strip()}")
                # except Exception as e:
                    # logger.error(f"Failed to shut down Istio sidecar in pod {pod_name}: {e}")


            return
        
        elif job_status.status.failed is not None and job_status.status.failed > 0:
            raise Exception(f"Job {job_name} failed.")
        time.sleep(10)  # 每 10 秒檢查一次 Job 狀態

#  custermize for clustering testing
def record_clustering_testing_result_to_mlflow(request: DagRequest, output_dir: str, logger):
    """ 記錄執行結果到 MLflow，包括 clustering 分群結果 """
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "default-key")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "default-secret")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(request.DAG_ID)
    dag_instance_unique_id = f'{request.DAG_ID}_{request.EXECUTION_ID}'

    subfolder_path = os.path.join(output_dir, f"cluster{request.PIPELINE_CONFIG.get('clusters_amount', 4)}")

    if not os.path.exists(subfolder_path):
        logger.error(f"無法找到 clustering 結果資料夾：{subfolder_path}")
        return

    with mlflow.start_run(run_name=dag_instance_unique_id):
        # 記錄 Request Body 為 Tags
        mlflow.set_tags(request.model_dump())

        # 上傳整個 clustering 結果資料夾
        artifact_name = f"clustering_result_cluster{request.PIPELINE_CONFIG.get('clusters_amount', 4)}"
        mlflow.log_artifacts(subfolder_path, artifact_path=artifact_name)

    logger.info("成功記錄 clustering 結果到 MLflow！")


@app.post("/Clustering/UploadLogToS3")
async def upload_log_to_s3(request: DagRequest):
    
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}/LOGS
    storage_path = STORAGE_PATH
    logs_folder_path = Path(storage_path) / f"{dag_id}_{execution_id}" / "LOGS"

    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Clustering/UploadLogToS3")
    dvc_worker = dvc_manager.get_worker(dag_id, execution_id)

    try:
        

        # Log 檔案路徑
        log_file_path = logs_folder_path / f"{dag_id}_{execution_id}.txt"

        if not log_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Log file {log_file_path} not found.")

        # 重新命名檔案
        renamed_log_filename = f"{dag_id}_{execution_id}_{task_stage_type}.txt"

        # 上傳到 MinIO
        target_path = f"{dag_id}_{execution_id}/logs/{renamed_log_filename}"

        logger.info(f"Uploading log file: {log_file_path} to MinIO at {target_path}")

        # 使用 DVCWorker 的 S3 客戶端上傳
        dvc_worker.s3_client.upload_file(
            Filename=str(log_file_path),
            Bucket=dvc_worker.minio_bucket,
            Key=target_path
        )

        logger.info("Log file uploaded successfully.")

        return {
            "status": "success",
            "message": f"Log file uploaded to MinIO at {target_path}"
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}") 

#########################################################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("MachineServer:app", host=MACHINE_IP, port=MACHINE_PORT, reload=True)