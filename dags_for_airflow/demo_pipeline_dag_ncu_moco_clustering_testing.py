from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.http_hook import HttpHook
from airflow.utils.dates import days_ago
from confluent_kafka import Producer
import requests
import json
from datetime import datetime
import random
import string
import pendulum

##########################################
# 全局配置parameter
API_GLOBAL_CONFIG = {
    "DAG_NAME": "casestudy3pipelinencumococlusteringtesting",

    "DATASET_NAME": "RICE_EAST",
    "DATASET_VERSION": "v1.0",

    # code repo used in specific stage api task
    "CODE_REPO_URL": {
        "Preprocessing": "https://github.com/chen88088/NCU-RSS-1.5-Preprocessing.git",
        "Clustering": "https://github.com/chen88088/AutoClusteringTool.git"
    },
    
    # image used in specific stage api task
    "IMAGE_NAME":{
        "Preprocessing": "",
        "Clustering": "moa_ncu/ncu-moco-clustering-testing"
    },

    "MODEL_NAME": "ncu_moa_moco",
    "MODEL_VERSION": "1",

    # deployer info
    "DEPLOYER_NAME": "Jerry",
    "DEPLOYER_EMAIL": "jerry@mlops.com",

    # Pipeline Config
    "PIPELINE_CONFIG" : {
        "epochs": 20,
        "learning_rate": 0.001,
        "batch_size": 32,
        "clusters_amount": "5"
    },
        #############################
    # Pipeline Stage Service Image
    "PIPELINE_STAGE_SERVICE_IMAGE_CONFIG" :
    {
        "CLUSTERING" : 
        {
            "SERVICE_NAME": "NCU_MoCo_Clustering_Testing_Service",
            "SERVICE_IMAGE_NAME":"moa_ncu/ncu-moco-clustering-testing-server",
            "SERVICE_IMAGE_TAG":"latest",
            "SERVICE_PORT":"8011"
        }
    }
}

##########################################

KAFKA_SERVICE_CONFIG = {
    "KAFKA_BROKER_URL": 'kafka-0.kafka-headless.kafka.svc.cluster.local:9092,kafka-1.kafka-headless.kafka.svc.cluster.local:9092,kafka-2.kafka-headless.kafka.svc.cluster.local:9092',
    "KAFKA_TOPIC": "test-log"
}

##########################################
# Kafka 配置
kafka_conf = {
    'bootstrap.servers': KAFKA_SERVICE_CONFIG["KAFKA_BROKER_URL"]  
}

kafka_topic = KAFKA_SERVICE_CONFIG["KAFKA_TOPIC"]

# function send messageto Kafka 
def send_message_to_kafka(message):
    producer = Producer(kafka_conf)
    topic = kafka_topic

    # API RESPOND PROCESS LOGIC
    # dynamically add respond's status and message into  producer context
    if "TASK_API_RESPOND" in message and isinstance(message["TASK_API_RESPOND"], dict):
        message["msg"] = message["TASK_API_RESPOND"].get("message", "No message in API response")
        message["status"] = message["TASK_API_RESPOND"].get("status", "No status in API response")
    else:
        message["msg"] = "Task Finished without detailed message"
        message["status"] = "Task Finished without status"
    
    try:
        producer.produce(topic, value=json.dumps(message))
        producer.flush()
        print("Message sent to Kafka successfully!")
    except Exception as e:
        print(f"Failed to send message to Kafka: {e}")

###########################################
# 

# # set time zone
# local_tz = pendulum.timezone("Asia/Taipei")

# function to record pipeline start time as execution id
def generate_execution_id(ti):
    
    # Execution Id
    now_str = datetime.now().strftime("%Y%m%dx%H%M%S")
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    execution_id = f"{now_str}xx{rand_str}"

    # 打印 & 存進 XCom
    print(f"[Generate_Execution_ID] execution_id = {execution_id}")
    ti.xcom_push(key='execution_id', value=execution_id)


###########################################

# function to get assigned service info
def get_external_service_info(ti, service_name):
    
    # Dag Id
    dag_id = ti.dag_id
    execution_id = ti.xcom_pull(key='execution_id', task_ids='Generate_Execution_ID')

    # call service
    http_hook = HttpHook(http_conn_id='controller_connection', method='POST')
    endpoint = f'/allocate_service/{service_name}'
    
    # add request body
    body = {
        "dag_id": dag_id,
        "execution_id": execution_id
    }

    response = http_hook.run(endpoint, json=body)
       
    if response.status_code == 200:
        result = response.json()
        assigned_service_instance = result.get('assigned_service_instance_id')
        assigned_ip = result.get('assigned_service_instance_ip')
        assigned_port = result.get('assigned_service_instance_port')

        ti.xcom_push(key='assigned_service_instance', value=assigned_service_instance)
        ti.xcom_push(key='assigned_ip', value=assigned_ip)
        ti.xcom_push(key='assigned_port', value=assigned_port)
    else:
        raise Exception(f"Failed to get service info: {response.text}")

# function to release assigned service 
def release_external_service(ti, info_source_task_ids):
    
    dag_id = ti.dag_id
    execution_id = ti.xcom_pull(key='execution_id', task_ids='Generate_Execution_ID')
    
    assigned_service_instance_id = ti.xcom_pull(key='assigned_service_instance', task_ids=info_source_task_ids)
    
    # call service
    http_hook = HttpHook(http_conn_id='controller_connection', method='POST')
    endpoint = f'/release_service/{assigned_service_instance_id}'
    
    # add request body
    body = {
        "dag_id": dag_id,
        "execution_id": execution_id
    }

    response = http_hook.run(endpoint, json=body)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Service Release: {result}")
    else:
        raise Exception(f"Failed to release service: {response}")


# function to get ml_serving_pod info
def get_ml_serving_pod_info(ti, ml_serving_pod_image_name , image_tag, export_port):
    
    # Dag Id
    dag_id = ti.dag_id
    execution_id = ti.xcom_pull(key='execution_id', task_ids='Generate_Execution_ID')

    # call service
    http_hook = HttpHook(http_conn_id='controller_connection', method='POST')
    endpoint = f'/create_pod'
    
    # add request body
    body = {
        "image_name": ml_serving_pod_image_name,
        "image_tag": image_tag,
        "export_port": export_port
    }

    response = http_hook.run(endpoint, json=body)
    
    if response.status_code == 200:
        result = response.json()

        pod_name = result.get('pod_name')
        pod_ip = result.get('pod_ip')
        image = result.get('image')
        service_dns = result.get('pod_service')

        # Push 到 XCom 給後續 task 用
        ti.xcom_push(key='assigned_service_instance', value=pod_name)
        # !!! notice herer use svc dns for in cluster cross namespace communication
        ti.xcom_push(key='assigned_ip', value=service_dns)
        ti.xcom_push(key='assigned_port', value=export_port)
    else:
        raise Exception(f"Failed to create pod: {response.text}")

# function to delete ml_serving_pod 
def delete_ml_serving_pod(ti, info_source_task_ids):

    dag_id = ti.dag_id
    execution_id = ti.xcom_pull(key='execution_id', task_ids='Generate_Execution_ID')

    # 從前面任務取得 pod_name
    pod_name = ti.xcom_pull(key='assigned_service_instance', task_ids=info_source_task_ids)

    http_hook = HttpHook(http_conn_id='controller_connection', method='DELETE')
    endpoint = f'/delete_pod/{pod_name}'

    response = http_hook.run(endpoint)

    if response.status_code == 200:
        result = response.json()
        print(f"Pod deleted: {result}")
    else:
        raise Exception(f"Failed to delete pod: {response}")

#########################################
# CLASS: ApiCaller
class ApiCaller:
    def __init__(self, ti, info_source_task_ids, task_stage_type):
        self.ti = ti
        self.dag_id = ti.dag_id 
        self.execution_id = ti.xcom_pull(key='execution_id', task_ids='Generate_Execution_ID')
        self.assigned_service = ti.xcom_pull(key='assigned_service_instance', task_ids=info_source_task_ids)
        self.assigned_ip = ti.xcom_pull(key='assigned_ip', task_ids=info_source_task_ids)
        self.assigned_port = ti.xcom_pull(key='assigned_port', task_ids=info_source_task_ids)
        self.body = self.create_request_body(task_stage_type)

    # function: build request body
    def create_request_body(self, task_stage_type: str):
        
        return {
            
            "DAG_ID": self.dag_id,
            "EXECUTION_ID": self.execution_id,
            
            "DATASET_NAME": API_GLOBAL_CONFIG["DATASET_NAME"],
            "DATASET_VERSION": API_GLOBAL_CONFIG["DATASET_VERSION"],
            
            "CODE_REPO_URL": API_GLOBAL_CONFIG["CODE_REPO_URL"],
            
            "IMAGE_NAME": API_GLOBAL_CONFIG["IMAGE_NAME"],

            "MODEL_NAME": API_GLOBAL_CONFIG["MODEL_NAME"],
            "MODEL_VERSION": API_GLOBAL_CONFIG["MODEL_VERSION"],

            "DEPLOYER_NAME": API_GLOBAL_CONFIG["DEPLOYER_NAME"],
            "DEPLOYER_EMAIL": API_GLOBAL_CONFIG["DEPLOYER_EMAIL"],

            "PIPELINE_CONFIG": API_GLOBAL_CONFIG["PIPELINE_CONFIG"],

            "TASK_STAGE_TYPE": task_stage_type
        }

    # 通用的 API 调用方法
    def call_api(self, route):
        target_api_url = f"http://{self.assigned_ip}:{self.assigned_port}/{route}"
        try:
            response = requests.post(target_api_url, json=self.body)
            response.raise_for_status()
            response_result = response.json()
            print(f"API call result for {route}: {response_result}")
            # send result to  Kafka
            kafka_message = {
                "DAG_ID": self.dag_id,
                "EXECUTION_ID": self.execution_id,
                "TASK_ID": route,
                "TASK_API_RESPOND": response_result,
                "status": None,
                "msg": None
            }
            send_message_to_kafka(kafka_message)
        except Exception as e:
            raise Exception(f"Failed to call API {route}: {e}")


# 調用不同的 API router的任務
def call_api_task(ti, info_source_task_ids, task_stage_type, route):
    api_caller = ApiCaller(ti, info_source_task_ids, task_stage_type)
    api_caller.call_api(route)

###########################################
# define DAG
Dag_Name = API_GLOBAL_CONFIG["DAG_NAME"]
dag = DAG(
    Dag_Name, 
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(1)
    },
    schedule_interval=None,
)

###########################################
# define task

## Generate Dag execution id
Generate_Execution_ID= PythonOperator(
    task_id='Generate_Execution_ID',
    python_callable=generate_execution_id,
    provide_context=True,
    dag=dag,
) 

## Preprocessing

# task:  get service info
Get_Preprocessing_Service_Information = PythonOperator(
    task_id='Get_Preprocessing_Service_Information',
    python_callable=get_external_service_info,
    op_kwargs={
        'service_name': 'Preprocessing'
    },
    dag=dag,
)

# task: /Preprocessing/RegisterDag
Preprocessing_RegisterDag= PythonOperator(
    task_id='Preprocessing_RegisterDag',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/RegisterDag'
    },
    dag=dag,
)

# task: /Preprocessing/DownloadDataset
Preprocessing_DownloadDataset= PythonOperator(
    task_id='Preprocessing_DownloadDataset',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/DownloadDataset'
    },
    dag=dag,
)

# task: /Preprocessing/SetupFolder
Preprocessing_SetupFolder= PythonOperator(
    task_id='Preprocessing_SetupFolder',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/SetupFolder'
    },
    dag=dag,
)

# task: /Preprocessing/ModifyPreprocessingConfig
Preprocessing_ModifyPreprocessingConfig= PythonOperator(
    task_id='Preprocessing_ModifyPreprocessingConfig',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/ModifyPreprocessingConfig'
    },
    dag=dag,
)

# task: /Preprocessing/GenerateParcelUniqueId
Preprocessing_GenerateParcelUniqueId= PythonOperator(
    task_id='Preprocessing_GenerateParcelUniqueId',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/GenerateParcelUniqueId'
    },
    dag=dag,
)

# task: /Preprocessing/GeneratePng
Preprocessing_GeneratePng= PythonOperator(
    task_id='Preprocessing_GeneratePng',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/GeneratePng'
    },
    dag=dag,
)

# task: /Preprocessing/WriteGtFile
Preprocessing_WriteGtFile= PythonOperator(
    task_id='Preprocessing_WriteGtFile',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/WriteGtFile'
    },
    dag=dag,
)

# task: /Preprocessing/UploadPreprocessingResult
Preprocessing_UploadPreprocessingResult= PythonOperator(
    task_id='Preprocessing_UploadPreprocessingResult',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/UploadPreprocessingResult'
    },
    dag=dag,
)

# task: /Preprocessing/UploadLogToS3
Preprocessing_UploadLogToS3= PythonOperator(
    task_id='Preprocessing_UploadLogToS3',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information',
        'task_stage_type': 'Preprocessing',
        'route': 'Preprocessing/UploadLogToS3'
    },
    dag=dag,
)

# function to release service 
Release_Preprocessing_Service = PythonOperator(
    task_id='Release_Preprocessing_Service',
    python_callable=release_external_service,
    op_kwargs={
        'info_source_task_ids': 'Get_Preprocessing_Service_Information'
    },
    dag=dag
)

#################################
# 取出 config 中的設定
stage_service_config = API_GLOBAL_CONFIG["PIPELINE_STAGE_SERVICE_IMAGE_CONFIG"]["CLUSTERING"]


# function of get service info
Get_Clustering_Testing_Service_Information = PythonOperator(
    task_id='Get_Clustering_Testing_Service_Information',
    python_callable=get_ml_serving_pod_info,
    op_kwargs={
        'ml_serving_pod_image_name': stage_service_config["SERVICE_IMAGE_NAME"],
        'image_tag': stage_service_config["SERVICE_IMAGE_TAG"],
        'export_port': int(stage_service_config["SERVICE_PORT"])
    },
    dag=dag,
)

# [Clustering/RegisterDag]
Clustering_RegisterDag= PythonOperator(
    task_id='Clustering_RegisterDag',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/RegisterDag'
    },
    dag=dag,
)

# [Clustering/SetupFolder]
Clustering_SetupFolder= PythonOperator(
    task_id='Clustering_SetupFolder',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/SetupFolder'
    },
    dag=dag,
)

# [Clustering/DownloadPreprocessingResult]
Clustering_DownloadPreprocessingResult= PythonOperator(
    task_id='Clustering_DownloadPreprocessingResult',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/DownloadPreprocessingResult'
    },
    dag=dag,
)

# [Clustering/FetchModel]
Clustering_FetchModel= PythonOperator(
    task_id='Clustering_FetchModel',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/FetchModel'
    },
    dag=dag,
)

# [Clustering/ModifyClusteringTestingConfig]
Clustering_ModifyClusteringTestingConfig= PythonOperator(
    task_id='Clustering_ModifyClusteringTestingConfig',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/ModifyClusteringTestingConfig'
    },
    dag=dag,
)


# [Clustering/ExecuteClusteringTestingScripts]
Clustering_ExecuteClusteringTestingScripts= PythonOperator(
    task_id='Clustering_ExecuteClusteringTestingScripts',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/ExecuteClusteringTestingScripts'

    },
    dag=dag,
)

# [Clustering/UploadLogToS3]
Clustering_UploadLogToS3= PythonOperator(
    task_id='Clustering_UploadLogToS3',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information',
        'task_stage_type': 'Clustering',
        'route': 'Clustering/UploadLogToS3'
    },
    dag=dag,
)

Delete_Clustering_Testing_ML_Serving_Pod = PythonOperator(
    task_id='Delete_Clustering_Testing_ML_Serving_Pod',
    python_callable=delete_ml_serving_pod,
    op_kwargs={
        'info_source_task_ids': 'Get_Clustering_Testing_Service_Information'
    },
    dag=dag
)

#############################


# use () to define task dependency
(
    Generate_Execution_ID
    >> Get_Preprocessing_Service_Information 
    >> Preprocessing_RegisterDag
    >> Preprocessing_DownloadDataset
    >> Preprocessing_SetupFolder
    >> Preprocessing_ModifyPreprocessingConfig
    >> Preprocessing_GenerateParcelUniqueId
    >> Preprocessing_GeneratePng
    >> Preprocessing_WriteGtFile
    >> Preprocessing_UploadPreprocessingResult
    >> Preprocessing_UploadLogToS3
    >> Release_Preprocessing_Service
    >> Get_Clustering_Testing_Service_Information 
    >> Clustering_RegisterDag
    >> Clustering_SetupFolder
    >> Clustering_DownloadPreprocessingResult
    >> Clustering_FetchModel
    >> Clustering_ModifyClusteringTestingConfig
    >> Clustering_ExecuteClusteringTestingScripts
    >> Clustering_UploadLogToS3
    >> Delete_Clustering_Testing_ML_Serving_Pod
)

