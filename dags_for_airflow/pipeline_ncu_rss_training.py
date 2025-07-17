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

######################################
# 全局配置parameter
API_GLOBAL_CONFIG = {
    "DAG_NAME": "xxxtestdagncursstraing",

    "DATASET_NAME": "mock-dataset",
    "DATASET_VERSION": "v1.0",


    "CODE_REPO_URL": {
        "Training": "https://github.com/chen88088/NCU-RSS-1.5.git"
    },

    "IMAGE_NAME":{
        "Training": "moa_ncu/ncu-rss-training"
    },

   "MODEL_NAME": "",
   "MODEL_VERSION": "",

   # deployer info
   "DEPLOYER_NAME": "Jerry_Chen",
   "DEPLOYER_EMAIL": "chen88088@gmail.com",

   # Pipeline Config
   "PIPELINE_CONFIG" : {
       "":""
   }


#    # task definition
#    "TASK_STAGE_TYPE": "",
#    "SERVICE_USAGE_RESOURCE_TYPE": ""
#    "SERVICE_USAGE_COUNT": ""
}

##########################################

SERVICE_CONFIG = {
    "KAFKA_BROKER_URL": 'kafka-0.kafka-headless.kafka.svc.cluster.local:9092,kafka-1.kafka-headless.kafka.svc.cluster.local:9092,kafka-2.kafka-headless.kafka.svc.cluster.local:9092',
    "KAFKA_TOPIC": "test-log"
}

##########################################
# Kafka 配置
kafka_conf = {
    'bootstrap.servers': SERVICE_CONFIG["KAFKA_BROKER_URL"]  
}

kafka_topic = SERVICE_CONFIG["KAFKA_TOPIC"]

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
    print(f"[GenerateExecutionID] execution_id = {execution_id}")
    ti.xcom_push(key='execution_id', value=execution_id)


###########################################

# function to get assigned service info
def get_external_service_info(ti, service_name):
    
    # Dag Id
    dag_id = ti.dag_id
    execution_id = ti.xcom_pull(key='execution_id', task_ids='GenerateExecutionID')

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
    execution_id = ti.xcom_pull(key='execution_id', task_ids='GenerateExecutionID')
    
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
    execution_id = ti.xcom_pull(key='execution_id', task_ids='GenerateExecutionID')

    # call service
    http_hook = HttpHook(http_conn_id='controller_connection', method='POST')
    endpoint = f'/create_pod/{ml_serving_pod_image_name}'
    
    # add request body
    body = {
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
    execution_id = ti.xcom_pull(key='execution_id', task_ids='GenerateExecutionID')

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
        self.execution_id = ti.xcom_pull(key='execution_id', task_ids='GenerateExecutionID')
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
GenerateExecutionID= PythonOperator(
    task_id='GenerateExecutionID',
    python_callable=generate_execution_id,
    provide_context=True,
    dag=dag,
) 

#################################
# function of get service info
Get_NCU_RSS_Training_Service_Information = PythonOperator(
    task_id='Get_NCU_RSS_Training_Service_Information',
    python_callable=get_ml_serving_pod_info,
    op_kwargs={
        'ml_serving_pod_image_name': 'ncu-rss-training-server',
        'image_tag': 'latest',
        'export_port': 8001
    },
    dag=dag,
)

# [Training/RegisterDag]
Training_RegisterDag= PythonOperator(
    task_id='Training_RegisterDag',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_NCU_RSS_Training_Service_Information',
        'task_stage_type': 'Training',
        'route': 'Training/RegisterDag'
    },
    dag=dag,
)

# [Training/SetupFolder]
Training_SetupFolder= PythonOperator(
    task_id='Training_SetupFolder',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_NCU_RSS_Training_Service_Information',
        'task_stage_type': 'Training',
        'route': 'Training/SetupFolder'
    },
    dag=dag,
)

# [Training/DownloadDataset]
Training_DownloadDataset= PythonOperator(
    task_id='Training_DownloadDataset',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_NCU_RSS_Training_Service_Information',
        'task_stage_type': 'Training',
        'route': 'Training/DownloadDataset'
    },
    dag=dag,
)


# [Training/ExecuteTrainingScripts]
Training_ExecuteTrainingScripts= PythonOperator(
    task_id='Training_ExecuteTrainingScripts',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_NCU_RSS_Training_Service_Information',
        'task_stage_type': 'Training',
        'route': 'Training/ExecuteTrainingScripts'

    },
    dag=dag,
)

# [Training/UploadLogToS3]
Training_UploadLogToS3= PythonOperator(
    task_id='Training_UploadLogToS3',
    python_callable=call_api_task,
    op_kwargs={
        'info_source_task_ids': 'Get_NCU_RSS_Training_Service_Information',
        'task_stage_type': 'Training',
        'route': 'Training/UploadLogToS3'
    },
    dag=dag,
)

Delete_NCU_RSS_Training_ML_Serving_Pod = PythonOperator(
    task_id='Delete_NCU_RSS_Training_ML_Serving_Pod',
    python_callable=delete_ml_serving_pod,
    op_kwargs={
        'info_source_task_ids': 'Get_NCU_RSS_Training_Service_Information'
    },
    dag=dag
)

#############################


# use () to define task dependency
(
    GenerateExecutionID
    >> Get_NCU_RSS_Training_Service_Information 
    >> Training_RegisterDag
    >> Training_SetupFolder
    >> Training_DownloadDataset
    >> Training_ExecuteTrainingScripts
    >> Training_UploadLogToS3
    >> Delete_NCU_RSS_Training_ML_Serving_Pod
)


