import pickle
import io
from minio import Minio
import json
import os
import uuid
import requests


def get_client():
    mk = json.load(open('config.json'))

    try:
        client = Minio(endpoint=mk['ClientUrl'],
                       access_key=mk['MinioAccessKey'],
                       secret_key=mk['MinioSecretKey'],
                       secure=mk['Secure']
                       )
        print("loaded client with endpoint: ", mk['ClientUrl'])
    except:
        print("Couldn't load client")
        client = None

    return client

def get_config():

    try:
        config = json.load(open('config.json'))

    except:
        print("Couldn't find config.json file.")
        config = None

    return config

def save_model(model, model_id=None, name = '', model_description = ''):

    client = get_client()

    if not client.bucket_exists(bucket_name='models'):
        client.make_bucket(bucket_name='models')

    obj = pickle.dumps(model)
    if model_id is None:
        model_id = str(uuid.uuid4())
    client.put_object('models', model_id, io.BytesIO(obj), len(obj))

    # config = get_config()
    #
    # if config is not None:
    #
    #     project_name = config['ProjectName']
    #     project_id = config['ProjectId']
    #     url = 'https://platform.' + project_name + '.scaleout.se/api/models/'
    #     model_url = os.path.join('https://minio.' + project_name + '.platform.demo.scaleout.se/minio/models', )
    #
    # else:
    #
    #     url = ''
    #     project_id = ''
    #     model_url = ''
    #
    # model_name = name
    # myobj = {
    #     'uid': model_id,
    #     'name': model_name,
    #     'description': model_description,
    #     'url': model_url,
    #     'project': project_id,
    # }
    #
    # x = requests.post(url, data=myobj)
    #
    # print(x.status_code)

def load_model(model_id):

    client = get_client()

    try:
        obj = client.get_object(bucket_name='models', object_name=model_id)
        model = pickle.loads(obj.read())

    except:
        print("Couldn't find model: ", model_id)
        model = None

    return model

def load_data(name):

    client = get_client()
    try:
        data = pickle.loads(client.get_object('dataset', name).read())
    except:
        print("Couldn't load data: ", name)
        data = None

    return data

