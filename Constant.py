import os

MONGO_URL = os.environ["webappDB"]
SECRET = os.environ["SECRET_TOKEN"]

PACS_ADDR = os.environ["PACS_ADDR"] #"192.1.10.162" #   "13.229.184.70" # 127.0.0.1
PACS_PORT = os.environ["PACS_PORT"] # 104 # 11114 # 11112 # 11113

AE_TITLE_SCP = "SYNAPSEDICOM" #"AE_TITLE_NRT02" #   "MY_ECHO_SCP_AWS" #

AI_VERSION = {
    "classification_pylon_1024": "UTC_MDCU_Rad_v1.0.2.6"
}