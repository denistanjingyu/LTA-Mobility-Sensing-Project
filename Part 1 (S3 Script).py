# Part 1 MSP
# Retrieve data from S3 bucket
# Save to csv file
import boto3

_BUCKET_NAME = ''
_PREFIX = ''

client = boto3.client('s3', 
                      aws_access_key_id = '', 
                      aws_secret_access_key = '')

def ListFiles(client):
    """List files in specific S3 URL"""
    response = client.list_objects(Bucket = _BUCKET_NAME, 
                                   Prefix = _PREFIX)
    for content in response.get('Contents', []):
        yield content.get('Key')

file_list = ListFiles(client)
for file in file_list:
    print ('File found: %s' % file)
    
# Accessing s3 bucket data
s3 = boto3.resource('s3', 
                    aws_access_key_id = '',
                    aws_secret_access_key = '')
import json
import pandas as pd
import boto3
import io
bucket = s3.Bucket('')
prefix_objs = bucket.objects.filter(Prefix = "") 

# Read in accelerometer sensor data
# Check the shape
# Save as csv
df_accel = pd.DataFrame(columns = ['Altitude', 
                                   'JourneyID', 
                                   'lat', 
                                   'lng', 
                                   'Mode', 
                                   'Timestamp'])
for obj in prefix_objs:
    key = obj.key
    obje = client.get_object(Bucket = '', 
                             Key = key)
    data = obje['Body'].read().decode()
    json_content = json.loads(data)
    try:
        json_contents = json_content['accelerometer']
        dff = pd.DataFrame(json_contents)
    except KeyError:
         pass
    df_accel = df_accel.append(dff)
    
df_accel.shape
df_accel.head()
df_accel.to_csv('accelerometer.csv', 
                index = False)

# Read in gyroscope sensor data
# Check the shape
# Save as csv
for obj in prefix_objs:
    key = obj.key
    obje = client.get_object(Bucket = '', 
                             Key = key)
    data = obje['Body'].read().decode()
    json_content = json.loads(data)
    try:
        json_contents = json_content['gyroscope']
        dff = pd.DataFrame(json_contents)
    except KeyError:
         pass
    df_gyro = df_gyro.append(dff)

df_gyro.shape
df_gyro.head()
df_gyro.to_csv('gyroscope.csv',
               index = False)

# Read in magnetometer sensor data
# Check the shape
# Save as csv
df_magneto = pd.DataFrame(columns = ['Altitude', 
                                     'JourneyID', 
                                     'lat', 
                                     'lng', 
                                     'Mode', 
                                     'Timestamp'])
for obj in prefix_objs:
    key = obj.key
    obje = client.get_object(Bucket = '', 
                             Key=key)
    data = obje['Body'].read().decode()
    json_content = json.loads(data)
    try:
        json_contents = json_content['magnetometer']
        dff = pd.DataFrame(json_contents)
    except KeyError:
         pass
    df_magneto = df_magneto.append(dff)
    
df_magneto.shape
df_magneto.head()
df_magneto.to_csv('magnetometer.csv', 
                  index = False)
