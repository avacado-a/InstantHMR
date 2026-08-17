import sys
import os
import json
import urllib.parse
import boto3
from boto3.dynamodb.conditions import Key
import numpy as np
import pandas as pd
import datetime
import requests
import scipy
from skinematics import quat
import skinematics as skin
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import sosfiltfilt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import xml.etree.ElementTree as ET
import operator
from sympy import true
import math
import statistics
from numpy.core.fromnumeric import mean
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.conditions import Attr
import copy
import AngleManager as am
import AngleManager_CV as amCV
import jointScores as js
import QuaternionDictionary_CV as qd
import metricsMain as mm
import cv2
import mediapipe as mp
import time

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)

# Arguments passed
print("\nName of Python script:", sys.argv[0])

#print("\nArguments passed:", end = " ")
for i in range(1, n):
    print(sys.argv[i], end = " ")

if(n != 8):
    print("Incorrect number of arguments")

else:
    print('Loading function')

    # s3 = boto3.client('s3')
    AWSACCESSKEY= sys.argv[2]
    AWSSECRETKEY= sys.argv[3]

    #cognito auth for appsync
    session = boto3.Session(
        region_name='us-east-1',
        aws_access_key_id=AWSACCESSKEY,
        aws_secret_access_key=AWSSECRETKEY
    )
    s3 = session.client('s3', region_name='us-east-1')
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

    ## Main function - handler
    def handler(tempJson, devuser, devpass, produser, prodpass):
        
        event = json.loads(tempJson)

        # Get the object from the event and show its content type
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            original_object = response['Body'].read().decode('utf-8')
            everything = json.loads(original_object)

            original_key = 'transformed/' + key[7:] 
            sensor_data_key = 'transformed/sensorData/' + key[7:]
            joint_data_key = 'transformed/jointData/' + key[7:]
            rep_data_key = 'transformed/repData/' + key[7:]
            report_data_key = 'transformed/reportData/' + key[7:]
            score_data_key = 'transformed/jointAngleScores/' + key[7:]
            summary_metrics_key = 'transformed/overviewMetrics/' + key[7:]

            if bucket == 'bravebucket210651-dev':

                table_name = "MotionData-p24kufgrjzbm5lxpcxqrmlflni-dev"
                table = dynamodb.Table(table_name)
                db_response = table.query(IndexName = "s3Key-index", KeyConditionExpression=Key('s3Key').eq(key[7:]))

                ## Org ID
                if 'organizationID' in db_response['Items'][0]: 
                    org_id = db_response['Items'][0]['organizationID']
                else: 
                    org_id = ""

                ## CV Mode 
                if 'cvMode' in db_response['Items'][0]: 
                    cv_mode = db_response['Items'][0]['cvMode']
                else: 
                    cv_mode = ""

                ## Sensor mode 
                if 'sensorMode' in db_response['Items'][0]: 
                    sensor_mode = db_response['Items'][0]['sensorMode']
                else: 
                    sensor_mode = ""

                ## Single Sensor Recording
                if 'recordingMode' in db_response['Items'][0]: 
                    record_mode = db_response['Items'][0]['recordingMode']
                else: 
                    record_mode = ""
                
                ## Workout id 
                if 'workoutID' in db_response['Items'][0]: 
                    wkt_id = db_response['Items'][0]['workoutID']
                else: 
                    wkt_id = "" 

                ## Exercise id
                if 'exerciseID' in db_response['Items'][0] : 
                    if db_response['Items'][0]['exerciseID'] != "" and db_response['Items'][0]['exerciseID'] != None: 
                        exc_id = db_response['Items'][0]['exerciseID'] 
                    else: 
                        exc_id = "" 
                else: 
                    exc_id = "" 

                print(exc_id)

                ## Exercise Parameters
                orientation = ""
                selected_joints = []
                exc_details = ""
                if exc_id != "" and exc_id != None: 
                    exercise_table_name  = "Exercise-p24kufgrjzbm5lxpcxqrmlflni-dev"
                    exercise_table = dynamodb.Table(exercise_table_name)
                    exercise_response = exercise_table.scan(FilterExpression=Attr('id').eq(exc_id))

                    ## Get orientation if exercise/motion is in Workout
                    if 'orientation' in exercise_response['Items'][0]: 
                        orientation = exercise_response['Items'][0]['orientation'] 

                    if 'detailsID' in exercise_response['Items'][0]:
                        exc_details = exercise_response['Items'][0]['detailsID']

                    exercise_details_table_name = "ExerciseDetails-p24kufgrjzbm5lxpcxqrmlflni-dev"
                    exercise_details_table = dynamodb.Table(exercise_details_table_name)
                    exercise_details_response = exercise_details_table.scan(FilterExpression=Attr('id').eq(exercise_response['Items'][0]['detailsID']))

                    ## Get selectedJoints if exercise/motion is in Workout
                    if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 
                        if 'cvJointMetrics' in exercise_details_response['Items'][0]:  
                            selected_joints = exercise_details_response['Items'][0]['cvJointMetrics']
                        elif 'jointMetrics' in exercise_details_response['Items'][0]:  
                            selected_joints = exercise_details_response['Items'][0]['jointMetrics']
                    else: 
                        if 'jointMetrics' in exercise_details_response['Items'][0]:  
                            selected_joints = exercise_details_response['Items'][0]['jointMetrics']

                ## Athlete parameters
                athlete_height_in = 0
                athlete_table_name = "Athlete-p24kufgrjzbm5lxpcxqrmlflni-dev"
                athlete_table = dynamodb.Table(athlete_table_name)
                athlete_response = athlete_table.scan(FilterExpression=Attr('id').eq(db_response['Items'][0]['athleteID']))

                if 'currentHeight' in athlete_response['Items'][0]: 
                    athlete_height_in = athlete_response['Items'][0]['currentHeight']

                ## External Video parameters
                if 'externalVideo' in db_response['Items'][0]: 
                    external_vid = db_response['Items'][0]['externalVideo']
                else: 
                    external_vid = False

                ## External Frame Rate parameters
                if 'frameRate' in db_response['Items'][0]: 
                    frame_rate = db_response['Items'][0]['frameRate']
                else: 
                    frame_rate = ""

                transformed_object, joint_kinematics_dict, rep_data_dict, assessment_data_dict, scoring_dict, summary_metrics_dict = transformData(everything, orientation, selected_joints, wkt_id, sensor_mode, record_mode, cv_mode, exc_details, athlete_height_in, exc_id, org_id, external_vid, frame_rate, bucket, key)
                print("DONE WITH TRANSFORMING")
 
                # Backup store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(transformed_object),
                    Bucket='bravebucket210651-dev',
                    Key = original_key)

                print("Sent to OG")

                # Sensor data store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(transformed_object),
                    Bucket='bravebucket210651-dev',
                    Key = sensor_data_key)

                print("Sent to Sensor Data")

                # Joint data store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(joint_kinematics_dict),
                    Bucket='bravebucket210651-dev',
                    Key = joint_data_key) 

                print("Sent to Joint Data") 

                # Rep data store 
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(rep_data_dict, default=convert),
                    Bucket='bravebucket210651-dev',
                    Key = rep_data_key) 

                print("Sent to Rep Data")

                # Report data store
                if wkt_id != "freesession" and wkt_id != "":  
                    s3.put_object(
                        ACL = 'public-read',
                        Body=json.dumps(assessment_data_dict, default=convert),
                        Bucket='bravebucket210651-dev',
                        Key = report_data_key) 
                    
                    if len(scoring_dict) != 0:
                        s3.put_object(
                            ACL = 'public-read',
                            Body=json.dumps(scoring_dict, default=convert),
                            Bucket='bravebucket210651-dev',
                            Key = score_data_key) 

                    print("Sent to Report Data")  
                
                # Summary Metrics store 
                if summary_metrics_dict != {}:
                    s3.put_object(
                        ACL = 'public-read',
                        Body=json.dumps(summary_metrics_dict, default=convert),
                        Bucket='bravebucket210651-dev',
                        Key = summary_metrics_key) 

                ## Set transformReady attribute to true in DynamoDB
                primary_id = db_response['Items'][0]['id']

                session = requests.Session()

                APPSYNC_API_ENDPOINT_URL = 'https://uxmvro3oyvf4vo2zm4yzxxk2ii.appsync-api.us-east-1.amazonaws.com/graphql'
                print("This is before the primary id key")
                print(primary_id)
                suprajquery = """id
            athleteID
            athlete {
                id
                firstName
                lastName
            }
            organizationID
            athleteUsername
            s3Key
            createdAt
            sport
            coachUsername
            coachID
            sessionName
            sessionTags
            duration
            averageSessionVelocity
            comment {
                id
                content
                author
            }
            workout {
                id
                type
                createdAt
            }
            exercise {
                id
                details {
                    id
                    name
                    description
                }
                reps
            }
            archived
            activity {
                id
                name
                reps
            }
            transformReady
            osimReady
            repLabelsReady
            gpLabelsReady"""

                query = """mutation UpdateMotionData{ updateMotionData(input: {id: \"""" + primary_id + """\", transformReady: true}) {"""+suprajquery+"""}}"""
                devtoken = get_dev_token(devuser, devpass).get("AuthenticationResult").get("AccessToken")
                response = session.request(
                    url=APPSYNC_API_ENDPOINT_URL,
                    method='POST',
                    headers={'authorization': devtoken},
                    json={'query': query}
                )


            elif bucket == 'bravebucket143743-prod':
                
                table_name = "MotionData-qnzs5kveyndghcyihdnalxdlj4-prod"
                table = dynamodb.Table(table_name)
                db_response = table.query(IndexName = "s3Key-index", KeyConditionExpression=Key('s3Key').eq(key[7:]))

                ## Org ID
                if 'organizationID' in db_response['Items'][0]: 
                    org_id = db_response['Items'][0]['organizationID']
                else: 
                    org_id = ""

                ## CV Mode 
                if 'cvMode' in db_response['Items'][0]: 
                    cv_mode = db_response['Items'][0]['cvMode']
                else: 
                    cv_mode = ""

                ## Sensor mode 
                if 'sensorMode' in db_response['Items'][0]: 
                    sensor_mode = db_response['Items'][0]['sensorMode']
                else: 
                    sensor_mode = ""

                ## Single Sensor Recording
                if 'recordingMode' in db_response['Items'][0]: 
                    record_mode = db_response['Items'][0]['recordingMode']
                else: 
                    record_mode = ""
                
                ## Workout id 
                if 'workoutID' in db_response['Items'][0]: 
                    wkt_id = db_response['Items'][0]['workoutID']
                else: 
                    wkt_id = ""

                ## Exercise id
                if 'exerciseID' in db_response['Items'][0]: 
                    if db_response['Items'][0]['exerciseID'] != "" and db_response['Items'][0]['exerciseID'] != None: 
                        exc_id = db_response['Items'][0]['exerciseID'] 
                    else: 
                        exc_id = "" 
                else: 
                    exc_id = "" 

                print(exc_id)

                ## Exercise Parameters
                orientation = ""
                selected_joints = []
                exc_details = ""
                if exc_id != "" and exc_id != None: 
                    exercise_table_name  = "Exercise-qnzs5kveyndghcyihdnalxdlj4-prod"
                    exercise_table = dynamodb.Table(exercise_table_name)
                    exercise_response = exercise_table.scan(FilterExpression=Attr('id').eq(exc_id))

                    ## Get orientation if exercise/motion is in Workout 
                    if 'orientation' in exercise_response['Items'][0]: 
                        orientation = exercise_response['Items'][0]['orientation']

                    if 'detailsID' in exercise_response['Items'][0]:
                        exc_details = exercise_response['Items'][0]['detailsID']

                    exercise_details_table_name = "ExerciseDetails-qnzs5kveyndghcyihdnalxdlj4-prod"
                    exercise_details_table = dynamodb.Table(exercise_details_table_name)
                    exercise_details_response = exercise_details_table.scan(FilterExpression=Attr('id').eq(exercise_response['Items'][0]['detailsID']))

                    ## Get selectedJoints if exercise/motion is in Workout
                    if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 
                        if 'cvJointMetrics' in exercise_details_response['Items'][0]:  
                            selected_joints = exercise_details_response['Items'][0]['cvJointMetrics']
                        elif 'jointMetrics' in exercise_details_response['Items'][0]:  
                            selected_joints = exercise_details_response['Items'][0]['jointMetrics']
                    else: 
                        if 'jointMetrics' in exercise_details_response['Items'][0]:  
                            selected_joints = exercise_details_response['Items'][0]['jointMetrics']
                 
                ## Athlete parameters
                athlete_height_in = 0
                athlete_table_name = "Athlete-qnzs5kveyndghcyihdnalxdlj4-prod"
                athlete_table = dynamodb.Table(athlete_table_name)
                athlete_response = athlete_table.scan(FilterExpression=Attr('id').eq(db_response['Items'][0]['athleteID']))

                if 'currentHeight' in athlete_response['Items'][0]: 
                    athlete_height_in = athlete_response['Items'][0]['currentHeight']

                ## External Video parameters
                if 'externalVideo' in db_response['Items'][0]: 
                    external_vid = db_response['Items'][0]['externalVideo']
                else: 
                    external_vid = False

                ## External Frame Rate parameters
                if 'frameRate' in db_response['Items'][0]: 
                    frame_rate = db_response['Items'][0]['frameRate']
                else: 
                    frame_rate = ""

                transformed_object, joint_kinematics_dict, rep_data_dict, assessment_data_dict, scoring_dict, summary_metrics_dict = transformData(everything, orientation, selected_joints, wkt_id, sensor_mode, record_mode, cv_mode, exc_details, athlete_height_in, exc_id, org_id, external_vid, frame_rate, bucket, key)
                print("DONE WITH TRANSFORMING")
 
 
                # Backup store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(transformed_object),
                    Bucket='bravebucket143743-prod',
                    Key = original_key)

                # Sensor data store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(transformed_object),
                    Bucket='bravebucket143743-prod',
                    Key = sensor_data_key)

                # Joint data store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(joint_kinematics_dict),
                    Bucket='bravebucket143743-prod',
                    Key = joint_data_key)

                # Rep data store
                s3.put_object(
                    ACL = 'public-read',
                    Body=json.dumps(rep_data_dict, default=convert),
                    Bucket='bravebucket143743-prod',
                    Key = rep_data_key)

                # Report data store
                if wkt_id != "freesession" and wkt_id != "": 
                    s3.put_object(
                        ACL = 'public-read',
                        Body=json.dumps(assessment_data_dict, default=convert),
                        Bucket='bravebucket143743-prod',
                        Key = report_data_key)
                    
                    if len(scoring_dict) != 0:
                        s3.put_object(
                            ACL = 'public-read',
                            Body=json.dumps(scoring_dict, default=convert),
                            Bucket='bravebucket143743-prod',
                            Key = score_data_key)
                
                # Summary Metrics store 
                if summary_metrics_dict != {}:
                    s3.put_object(
                        ACL = 'public-read',
                        Body=json.dumps(summary_metrics_dict, default=convert),
                        Bucket='bravebucket143743-prod',
                        Key = summary_metrics_key) 

                ## Set transformReady attribute to true in DynamoDB
                primary_id = db_response['Items'][0]['id']

                session = requests.Session()

                APPSYNC_API_ENDPOINT_URL = 'https://vphwqva3xbf7nbcumi2bfiiq7m.appsync-api.us-east-1.amazonaws.com/graphql'
                suprajquery = """id
            athleteID
            athlete {
                id
                firstName
                lastName
            }
            organizationID
            athleteUsername
            s3Key
            createdAt
            sport
            coachUsername
            coachID
            sessionName
            sessionTags
            duration
            averageSessionVelocity
            comment {
                id
                content
                author
            }
            workout {
                id
                type
                createdAt
            }
            exercise {
                id
                details {
                    id
                    name
                    description
                }
                reps
            }
            archived
            activity {
                id
                name
                reps
            }
            transformReady
            osimReady
            repLabelsReady
            gpLabelsReady"""


                query = """mutation UpdateMotionData{ updateMotionData(input: {id: \"""" + primary_id + """\", transformReady: true}) {"""+suprajquery+"""}}"""

                prodtoken = get_prod_token(produser, prodpass).get("AuthenticationResult").get("AccessToken")
                response = session.request(
                    url=APPSYNC_API_ENDPOINT_URL,
                    method='POST',
                    headers={'authorization': prodtoken},
                    json={'query': query}
                )


            return {'status_code': 200}
        except Exception as e:
            if bucket == 'bravebucket210651-dev':

                ## Set transformReady attribute to true in DynamoDB
                table_name = "MotionData-p24kufgrjzbm5lxpcxqrmlflni-dev"
                table = dynamodb.Table(table_name)
                db_response = table.query(IndexName = "s3Key-index", KeyConditionExpression=Key('s3Key').eq(key[7:]))

                primary_id = db_response['Items'][0]['id']

                session = requests.Session()

                APPSYNC_API_ENDPOINT_URL = 'https://uxmvro3oyvf4vo2zm4yzxxk2ii.appsync-api.us-east-1.amazonaws.com/graphql'
                print("This is before the primary id key")
                print(primary_id)
                suprajquery = """id
                                athleteID
                                athlete {
                                    id
                                    firstName
                                    lastName
                                }
                                organizationID
                                athleteUsername
                                s3Key
                                createdAt
                                sport
                                coachUsername
                                coachID
                                sessionName
                                sessionTags
                                duration
                                averageSessionVelocity
                                comment {
                                    id
                                    content
                                    author
                                }
                                workout {
                                    id
                                    type
                                    createdAt
                                }
                                exercise {
                                    id
                                    details {
                                        id
                                        name
                                        description
                                    }
                                    reps
                                }
                                archived
                                activity {
                                    id
                                    name
                                    reps
                                }
                                transformReady
                                osimReady
                                repLabelsReady
                                gpLabelsReady"""

                query = """mutation UpdateMotionData{ updateMotionData(input: {id: \"""" + primary_id + """\", transformReady: false}) {"""+suprajquery+"""}}"""
                devtoken = get_dev_token(devuser, devpass).get("AuthenticationResult").get("AccessToken")
                response = session.request(
                    url=APPSYNC_API_ENDPOINT_URL,
                    method='POST',
                    headers={'authorization': devtoken},
                    json={'query': query}
                )

            elif bucket == 'bravebucket143743-prod':

                ## Set transformReady attribute to true in DynamoDB
                table_name = "MotionData-qnzs5kveyndghcyihdnalxdlj4-prod"
                table = dynamodb.Table(table_name)
                db_response = table.query(IndexName = "s3Key-index", KeyConditionExpression=Key('s3Key').eq(key[7:]))

                primary_id = db_response['Items'][0]['id']

                session = requests.Session()

                APPSYNC_API_ENDPOINT_URL = 'https://vphwqva3xbf7nbcumi2bfiiq7m.appsync-api.us-east-1.amazonaws.com/graphql'
                suprajquery = """id
                                athleteID
                                athlete {
                                    id
                                    firstName
                                    lastName
                                }
                                organizationID
                                athleteUsername
                                s3Key
                                createdAt
                                sport
                                coachUsername
                                coachID
                                sessionName
                                sessionTags
                                duration
                                averageSessionVelocity
                                comment {
                                    id
                                    content
                                    author
                                }
                                workout {
                                    id
                                    type
                                    createdAt
                                }
                                exercise {
                                    id
                                    details {
                                        id
                                        name
                                        description
                                    }
                                    reps
                                }
                                archived
                                activity {
                                    id
                                    name
                                    reps
                                }
                                transformReady
                                osimReady
                                repLabelsReady
                                gpLabelsReady"""


                query = """mutation UpdateMotionData{ updateMotionData(input: {id: \"""" + primary_id + """\", transformReady: true}) {"""+suprajquery+"""}}"""

                prodtoken = get_prod_token(produser, prodpass).get("AuthenticationResult").get("AccessToken")
                response = session.request(
                    url=APPSYNC_API_ENDPOINT_URL,
                    method='POST',
                    headers={'authorization': prodtoken},
                    json={'query': query}
                )

            print(e)
            print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
            raise e

    ## Convert for JSON
    def convert(o):
        if isinstance(o, np.int64): return int(o)  
        raise TypeError

    ##Live Subscription with Cognito - Supraj
    def get_dev_token(devuser, devpass):
        client = boto3.client('cognito-idp', region_name='us-east-1')
        response = client.initiate_auth(
            ClientId='43th6pnd8ajbs6hsebcj25is8v',
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': devuser,
                'PASSWORD': devpass
            }
        )
        return response

    def get_prod_token(produser, prodpass):
        client = boto3.client('cognito-idp', region_name='us-east-1')
        response = client.initiate_auth(
            ClientId='7k1tbckplqbgrjmk2agn7bqgsl',
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': produser,
                'PASSWORD': prodpass
            }
        )
        return response

    ## Extract and smooth the quaternions
    def extractQuat_Smooth(file, min_time, max_time) :
        min_time_mod = pd.to_datetime(min_time,unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')
        max_time_mod = pd.to_datetime(max_time,unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')
        file_df = {}

        # Smooth and resample joint angles
        if "unityJointAnglesDict" in file:
            if "time" in file["unityJointAnglesDict"] and len(file["unityJointAnglesDict"]["time"]) != 0: 
                joint_df = pd.DataFrame.from_dict(file["unityJointAnglesDict"])

                for i, row in joint_df.iterrows(): 
                    mod_time = pd.to_datetime(joint_df['time'][i],unit='s') #datetime.datetime.fromtimestamp(new_df['t'][i]) #.strftime('%Y-%m-%d %H:%M:%S.%f')
                    joint_df.at[i, 'time'] = mod_time 

                # Maybe drop duplicates? 

                # Resample and smooth
                joint_df = joint_df.set_index('time')
                joint_df = joint_df.resample('16ms').mean()

                joint_df = joint_df.interpolate()
                joint_df = joint_df.reset_index()

                for column in joint_df:
                    if column != "time": 
                        joint_df[column] = joint_df[column].ewm(span=15).mean()
                
                joint_df = joint_df[(joint_df['time'] > min_time_mod[:-4]) & (joint_df['time'] < max_time_mod[:-4])] 

                for i, row in joint_df.iterrows():
                    unix_time = (joint_df['time'][i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
                    joint_df.at[i, 'time'] = unix_time

                # Maybe start time at 0? 
                # joint_df['time'] = joint_df['time'] - joint_df['time'].iloc[0]

                file_df["unityJointAnglesDict"] = joint_df.drop(joint_df.tail(50).index)
                new_dict = joint_df.to_dict(orient="list")
                file["unityJointAnglesDict"] = new_dict
            else: 
                joint_df = pd.DataFrame.from_dict(file["unityJointAnglesDict"])

                for column in joint_df:
                    if column != "ival" and column != "frame": 
                        joint_df[column] = joint_df[column].ewm(span=10).mean()

                file_df["unityJointAnglesDict"] = joint_df
                new_dict = joint_df.to_dict(orient="list")
                file["unityJointAnglesDict"] = new_dict

        # Smooth and resample quaternions + linear acceleration
        for body_part in file :

            if "q" in body_part and len(file[body_part]) > 1:
                new_df = pd.DataFrame(file[body_part])
                try:
                    packet_col = new_df['packetNumber'].to_list()
                    suitID = new_df.at[0, 'suitUUID']
                    new_df = new_df.drop(columns=['label', 'packetNumber', 'suitUUID'])
                except:
                    packet_col = []
                    suitID = ""
                    new_df = new_df.drop(columns=['label'])
                new_df['t'] = new_df['t'].astype(object)
                antipodalID = ["FD20", "FD21", "FD22", "FD23", "FD24", "FD25", "FD30", "FD31", "FD32", "FD33", "FD34", "FD35"]

                for i, row in new_df.iterrows():
                    mod_time = pd.to_datetime(new_df['t'][i],unit='s') #datetime.datetime.fromtimestamp(new_df['t'][i]) #.strftime('%Y-%m-%d %H:%M:%S.%f')
                    new_df.at[i, 't'] = mod_time

                if suitID == "" or suitID not in antipodalID:
                    print("antipodaling")
                    for j in range(1, len(new_df)):
                        dt = new_df['w'][j-1]*new_df['w'][j] + new_df['x'][j-1]*new_df['x'][j] + new_df['y'][j-1]*new_df['y'][j] + new_df['z'][j-1]*new_df['z'][j]
                        if dt < 0:
                            dt = -dt
                            new_df.at[j, 'w'] = -new_df.at[j, 'w']
                            new_df.at[j, 'x'] = -new_df.at[j, 'x']
                            new_df.at[j, 'y'] = -new_df.at[j, 'y']
                            new_df.at[j, 'z'] = -new_df.at[j, 'z']


                new_df['t'] = drop_consecutive_duplicates(new_df['t'])
                new_df['w'] = drop_consecutive_duplicates(new_df['w'])
                new_df['x'] = drop_consecutive_duplicates(new_df['x'])
                new_df['y'] = drop_consecutive_duplicates(new_df['y'])
                new_df['z'] = drop_consecutive_duplicates(new_df['z'])

                new_df = new_df.set_index('t')
                new_df = new_df.resample('16ms').mean()

                new_df = new_df.interpolate()
                new_df = new_df.reset_index()

                new_df['w'] = new_df['w'].ewm(span = 5).mean()
                new_df['x'] = new_df['x'].ewm(span = 5).mean()
                new_df['y'] = new_df['y'].ewm(span = 5).mean()
                new_df['z'] = new_df['z'].ewm(span = 5).mean()

                new_df = new_df[(new_df['t'] > min_time_mod[:-4]) & (new_df['t'] < max_time_mod[:-4])]

                for i, row in new_df.iterrows():
                    unix_time = (new_df['t'][i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
                    new_df.at[i, 't'] = unix_time

                file_df[body_part] = new_df.drop(new_df.tail(50).index)
                new_dict = new_df.to_dict('records')
                file[body_part] = new_dict

            elif "q" in body_part and len(file[body_part]) <= 1: 
                file[body_part] = []

        for body_part in file:
            if "la" in body_part and len(file[body_part]) > 1:
                new_df = pd.DataFrame(file[body_part])
                new_df = new_df.drop(columns=['label'])
                new_df['t'] = new_df['t'].astype(object)
                for i, row in new_df.iterrows():
                    mod_time = pd.to_datetime(new_df['t'][i],unit='s') #datetime.datetime.fromtimestamp(new_df['t'][i]) #.strftime('%Y-%m-%d %H:%M:%S.%f')
                    new_df.at[i, 't'] = mod_time

                new_df['t'] = drop_consecutive_duplicates(new_df['t'])
                new_df['x'] = drop_consecutive_duplicates(new_df['x'])
                new_df['y'] = drop_consecutive_duplicates(new_df['y'])
                new_df['z'] = drop_consecutive_duplicates(new_df['z'])

                new_df = new_df.set_index('t')
                new_df = new_df.resample('16ms').mean()

                new_df = new_df.interpolate()
                new_df = new_df.reset_index()

                # EWM Smoothing (IF NEEDED)
                new_df['x'] = new_df['x'].ewm(span = 5).mean()
                new_df['y'] = new_df['y'].ewm(span = 5).mean()
                new_df['z'] = new_df['z'].ewm(span = 5).mean()

                new_df = new_df[(new_df['t'] > min_time_mod[:-4]) & (new_df['t'] < max_time_mod[:-4])]

                for i, row in new_df.iterrows():
                    unix_time = (new_df['t'][i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
                    new_df.at[i, 't'] = unix_time

                # Starting time at 0
                new_df['t'] = new_df['t'] - new_df['t'].iloc[0]

                file_df[body_part] = new_df
                new_dict = new_df.to_dict('records')
                file[body_part] = new_dict
            
            elif "la" in body_part and len(file[body_part]) <= 1:
                file[body_part] = []

        return file, file_df

    ## Extract and smooth the quaternions - CV Version
    def extractQuat_Smooth_cv(file, min_time, max_time) :
        min_time_mod = pd.to_datetime(min_time,unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')
        max_time_mod = pd.to_datetime(max_time,unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')

        trim3 = False
        if 'frontCameraUsed' in file: 
            if file['frontCameraUsed'] == True:
                print("Front camera used")
                trim3 = True
            else: 
                print("Back camera used")
                trim3 = False

        # Smooth and resample quaternions + linear acceleration
        for body_part, data in file.items():
            if isinstance(data, list) and len(data) > 0:            
                new_df = pd.DataFrame(file[body_part])
                try:
                    packet_col = new_df['packetNumber'].to_list()
                    suitID = new_df.at[0, 'suitUUID']
                    new_df = new_df.drop(columns=['label', 'packetNumber', 'suitUUID'])
                except:
                    packet_col = []
                    suitID = ""
                    new_df = new_df.drop(columns=['label'])
                new_df['t'] = new_df['t'].astype(object)
                antipodalID = ["FD20", "FD21", "FD22", "FD23", "FD24", "FD25", "FD30", "FD31", "FD32", "FD33", "FD34", "FD35"]

                new_df.loc[:, "t"] = pd.to_datetime(new_df.loc[:, "t"], unit='s')

                new_df['t'] = drop_consecutive_duplicates(new_df['t'])
                new_df['x'] = drop_consecutive_duplicates(new_df['x'])
                new_df['y'] = drop_consecutive_duplicates(new_df['y'])
                new_df['z'] = drop_consecutive_duplicates(new_df['z'])

                new_df = new_df.set_index('t')
                new_df = new_df.resample('16ms').mean()

                new_df = new_df.interpolate()
                new_df = new_df.reset_index()

                new_df['x'] = new_df['x'].ewm(span = 5).mean()
                new_df['y'] = new_df['y'].ewm(span = 5).mean()
                new_df['z'] = new_df['z'].ewm(span = 5).mean()

                new_df = new_df[(new_df['t'] > min_time_mod[:-4]) & (new_df['t'] < max_time_mod[:-4])]

                unix_time = pd.Series((new_df.loc[:, 't'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms'))
                new_df.loc[:, 't'] = unix_time
                
                new_dict = new_df.to_dict('records')
                file[body_part] = new_dict
        
        return file, trim3
    
    ## Extract and smooth the quaternions - CV Version (External Video)
    def extractQuat_SmoothExt_cv(file, min_time, max_time) :
        min_time_mod = pd.to_datetime(min_time,unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')
        max_time_mod = pd.to_datetime(max_time,unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')

        trim3 = False
        if 'frontCameraUsed' in file: 
            if file['frontCameraUsed'] == True:
                print("Front camera used")
                trim3 = True
            else: 
                print("Back camera used")
                trim3 = False

        # Smooth and resample quaternions + linear acceleration
        for body_part, data in file.items():
            if isinstance(data, list) and len(data) > 0:            
                new_df = pd.DataFrame(file[body_part])
                try:
                    packet_col = new_df['packetNumber'].to_list()
                    suitID = new_df.at[0, 'suitUUID']
                    new_df = new_df.drop(columns=['label', 'packetNumber', 'suitUUID'])
                except:
                    packet_col = []
                    suitID = ""
                    new_df = new_df.drop(columns=['label'])
                new_df['t'] = new_df['t'].astype(object)
                antipodalID = ["FD20", "FD21", "FD22", "FD23", "FD24", "FD25", "FD30", "FD31", "FD32", "FD33", "FD34", "FD35"]

                new_df.loc[:, "t"] = pd.to_datetime(new_df.loc[:, "t"], unit='s')

                new_df['t'] = drop_consecutive_duplicates(new_df['t'])
                new_df['x'] = drop_consecutive_duplicates(new_df['x'])
                new_df['y'] = drop_consecutive_duplicates(new_df['y'])
                new_df['z'] = drop_consecutive_duplicates(new_df['z'])

                # new_df = new_df.set_index('t')
                # new_df = new_df.resample('16ms').mean()

                # new_df = new_df.interpolate()
                # new_df = new_df.reset_index()

                new_df['x'] = new_df['x'].ewm(span = 15).mean()
                new_df['y'] = new_df['y'].ewm(span = 15).mean()
                new_df['z'] = new_df['z'].ewm(span = 15).mean()

                new_df = new_df[(new_df['t'] > min_time_mod[:-4]) & (new_df['t'] < max_time_mod[:-4])]

                unix_time = pd.Series((new_df.loc[:, 't'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms'))
                new_df.loc[:, 't'] = unix_time

                new_df = new_df.drop(new_df.head(25).index)
                new_dict = new_df.to_dict('records')
                file[body_part] = new_dict
        
        return file, trim3

    ## Calculate raw sample rate
    def rawSampleRate(file): 
        sample_dict = {}

        file_copy = copy.deepcopy(file)

        for body_part in file_copy :

            if "q" in body_part and len(file_copy[body_part]) != 0:
                new_df = pd.DataFrame(file_copy[body_part])
                for i, row in new_df.iterrows():
                    mod_time = pd.to_datetime(new_df['t'][i],unit='s') #datetime.datetime.fromtimestamp(new_df['t'][i]) #.strftime('%Y-%m-%d %H:%M:%S.%f')
                    new_df.at[i, 't'] = mod_time

                if len(file_copy[body_part]) > 1: 
                    total_time = new_df['t'].iloc[-1] - new_df['t'].iloc[0]
                    num_samples = len(new_df)

                    sample_dict[body_part] = num_samples / (int(total_time.total_seconds()))
                else: 
                    total_time = 1
                    num_samples = len(new_df)

                    sample_dict[body_part] = num_samples / total_time

        #Calculate average sample rate across all sensors
        avg_rate = sum(sample_dict.values()) / len(sample_dict)
        sample_dict["Avg. Rate"] = avg_rate

        return sample_dict
    
    ## Calculate raw sample rate - CV Version
    def rawSampleRate_cv(file): 
        sample_dict = {}

        file_copy = copy.deepcopy(file)

        for body_part, data in file_copy.items():
            if isinstance(data, list) and len(data) > 0:
                new_df = pd.DataFrame(file_copy[body_part])

                # for i, row in new_df.iterrows():
                #     mod_time = pd.to_datetime(new_df['t'][i],unit='s') #datetime.datetime.fromtimestamp(new_df['t'][i]) #.strftime('%Y-%m-%d %H:%M:%S.%f')
                #     new_df.at[i, 't'] = mod_time

                new_df.loc[:, "t"] = pd.to_datetime(new_df.loc[:, "t"], unit='s')

                if len(file_copy[body_part]) > 1: 
                    total_time = new_df['t'].iloc[-1] - new_df['t'].iloc[0]
                    num_samples = len(new_df)

                    sample_dict[body_part] = num_samples / (int(total_time.total_seconds()))
                else: 
                    total_time = 1
                    num_samples = len(new_df)

                    sample_dict[body_part] = num_samples / total_time

        #Calculate average sample rate across all sensors
        avg_rate = sum(sample_dict.values()) / len(sample_dict)
        sample_dict["Avg. Rate"] = avg_rate

        return sample_dict

    ## Drop duplicates in the dataset
    def drop_consecutive_duplicates(a):
        ar = a.values
        return a[np.concatenate(([True],ar[:-1]!= ar[1:]))]

    ## Find min and max timestamps
    def findTime(file) :
        min_times = []
        max_times = []
        for body_part in file :
            if "q" in body_part and len(file[body_part]) > 1 :
                t_min = file[body_part][0]["t"]
                t_max = file[body_part][-1]["t"]

                min_times.append(t_min)
                max_times.append(t_max)
            else :
                min_time = 0
                max_time = 0
        
        # Add times from joint angles 
        if "unityJointAnglesDict" in file:
            if "time" in file["unityJointAnglesDict"] and len(file["unityJointAnglesDict"]["time"]) != 0: 
                t_min = file["unityJointAnglesDict"]['time'][0]
                t_max = file["unityJointAnglesDict"]['time'][-1]

                min_times.append(t_min)
                max_times.append(t_max) 

        if len(min_times) != 0 and len(max_times) != 0:
            min_time = max(min_times)
            max_time = min(max_times)

        return min_time, max_time

    ## Find min and max timestamps - CV Version
    def findTime_CV(file) :
        min_times = []
        max_times = []
        for body_part, data in file.items():
            if isinstance(data, list) and len(data) > 0:
                t_min = file[body_part][0]["t"]
                t_max = file[body_part][-1]["t"]

                if not (t_min == 0 and t_max == 0):
                    min_times.append(t_min)
                    max_times.append(t_max)
            else :
                min_time = 0
                max_time = 0
        
        # Add times from joint angles 
        if len(min_times) != 0 and len(max_times) != 0:
            min_time = max(min_times)
            max_time = min(max_times)

        return min_time, max_time

    ## Extract and process the joint angles
    def jointKinematics(file): 
        joint_dict = {} 

        if "unityJointAnglesDict" in file: 
            if "time" in file["unityJointAnglesDict"] and len(file["unityJointAnglesDict"]["time"]) != 0: 
                step = (file["unityJointAnglesDict"]["time"][1] - file["unityJointAnglesDict"]["time"][0])/1000 

                for joint_name in file["unityJointAnglesDict"]: 

                    # Angles
                    angle_values = file["unityJointAnglesDict"][joint_name] 
                    
                    #Pelvis rotation correction (zeroing out)
                    if joint_name == "waist_rotation": 
                        angle_values = [x - angle_values[0] for x in angle_values]

                    # Velocity 
                    ang_vel = np.gradient(angle_values, step)
                    ang_vel = savgol_filter(ang_vel, 5, 2)
                    # print(ang_vel)
                    ang_vel = ang_vel.tolist() 

                    # Acceleration
                    ang_acc = np.gradient(ang_vel, step)
                    ang_acc = savgol_filter(ang_acc, 5, 2)
                    # print(ang_vel)
                    ang_acc = ang_acc.tolist()
                
                    joint_dict[joint_name] = {"Ang" : angle_values, "Vel": ang_vel, "Acc": ang_acc}
            else: 
                step = 0.016

                for joint_name in file["unityJointAnglesDict"]: 

                    # Angles
                    angle_values = file["unityJointAnglesDict"][joint_name] 
                    
                    #Pelvis rotation correction (zeroing out)
                    if joint_name == "waist_rotation" or joint_name == "waist_extension" or joint_name == "trunk_extension" or joint_name == "waist_bending": 
                        angle_values = [x - angle_values[0] for x in angle_values]

                    # Velocity 
                    ang_vel = np.gradient(angle_values, step)
                    ang_vel = savgol_filter(ang_vel, 5, 2)
                    # print(ang_vel)
                    ang_vel = ang_vel.tolist() 

                    # Acceleration
                    ang_acc = np.gradient(ang_vel, step)
                    ang_acc = savgol_filter(ang_acc, 5, 2)
                    # print(ang_vel)
                    ang_acc = ang_acc.tolist()
                
                    joint_dict[joint_name] = {"Ang" : angle_values, "Vel": ang_vel, "Acc": ang_acc} 
        

        return joint_dict 

    ## Determine reps within the data 

    def totalAcc(c):
        return np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)

    def getTotalAccDF(df, linacc_sensors, labels = []):
        # linacc_sensors = ['RLA', 'RUA', 'LLA', 'LUA', 'CUB', 'CLB', 'RUL', 'RLL', 'LUL', 'LLL']
        newdf = pd.DataFrame()

        for sensor in linacc_sensors:
            newdf[sensor] = df[['LA_la' + sensor + 'x', 'LA_la' + sensor + 'y', 'LA_la' + sensor + 'z']].apply(lambda row: totalAcc(row), axis=1)

        if len(labels) == 0:
            return newdf
        else:
            return pd.concat([df['time'], newdf, df[labels]], axis=1)

    def totalAccelerationTrialCounter(all_data):
        sensorsLA = ['laRUL', 'laRLL', 'laLUL', 'laLLL', 'laCLB', 'laCUB', 'laRLA', 'laRUA', 'laLUA', 'laLLA']

        df = pd.DataFrame()

        linacc_sensors = []

        # Extracting linear acceleration data
        for key, value in all_data.items():
            if key in sensorsLA and all_data[key] != []:
                dfTransformed = pd.DataFrame(list(value))
                print('counter_1')
                dfTransformed = dfTransformed.rename({'x': str("LA_" + key+'x'), 'y': str("LA_" + key+'y'), 'z': str("LA_" + key+'z')}, axis=1)
                print('counter_2')
                df = pd.concat([df, dfTransformed], axis=1)
                print('counter_3')
                linacc_sensors.append(key[2:])
        

        # linacc_sensors = ['RLA', 'RUA', 'LLA', 'LUA', 'CUB', 'CLB', 'RUL', 'RLL', 'LUL', 'LLL']
        distance_threshold = 150

        tad = getTotalAccDF(df, linacc_sensors)
        total_acceleration = tad[linacc_sensors].apply(lambda row: math.sqrt(np.sum(np.square(row))), axis=1)

        print('counter_4')

        height_threshold = max(5, statistics.mean(total_acceleration) + (max(total_acceleration) - statistics.mean(total_acceleration))*0.4)

        peaks, _ = find_peaks(total_acceleration, height = height_threshold, distance=distance_threshold)
        np.diff(peaks)

        windows = []
        numReps = 0
        for peak in peaks:
            windows.append(totalAccelerationWindowFinder(peak, total_acceleration))
            numReps += 1
        
        print('counter_5')

        return int(numReps), windows

    def totalAccelerationWindowFinder(peak, totalAccArray):
        window = []
    
        
        minVal = float('inf')
        index = -1
        for i in range(max(0, peak-200), peak):
            if totalAccArray[i] < minVal:
                minVal = totalAccArray[i]
                index = i

        window.append(index)
        end = min(totalAccArray.size-1, peak+80)
        window.append(end)
        return window

    ## Function for rep isolation for assessments + simple exercises
    def assessmentReps(joint_data): 
        joint_rep_dict = {}

        if not all(v == 0 for v in joint_data): 
            joint_abs = [abs(n) for n in joint_data]
            height_threshold = max(5, statistics.mean(joint_abs) + (max(joint_abs) - statistics.mean(joint_abs))*0.4)
            
            joint_peaks = find_peaks(joint_abs, height = height_threshold, distance = 10, prominence=1)
            # print(len(joint_abs))
            joint_windows = []
            joint_reps = 0 
            for peak in joint_peaks[0]:
                if peak - 10 < 0: 
                    joint_windows.append([0, peak + 10]) 
                elif peak + 10 > len(joint_abs): 
                    joint_windows.append([peak - 10, len(joint_abs)]) 
                elif peak - 10 < 0 and peak + 10 > len(joint_abs): 
                    joint_windows.append([[peak, peak]]) 
                else: 
                    joint_windows.append([peak-10, peak + 10])
                joint_reps += 1 
            
            joint_rep_dict = {"NumReps": joint_reps, "Windows": joint_windows}
         
        else: 
            joint_rep_dict = {}

        # print(joint_rep_dict)
        
        return joint_rep_dict 
     
    ## Generate report data for assessments
    def assessmentMetrics(selected_joints, orientation, osim_data, cv_mode = 0, detailsID = "",):
        ang_metrics = {}
        vel_metrics = {} 
        acc_metrics = {}
        dec_metrics = {} 

        opensim_metric_names = ["waist_extension", "waist_bending", "waist_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz", "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", \
            "knee_angle_r", "knee_angle_r_beta", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r", "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "knee_angle_l_beta", \
                "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l", "lumbar_extension", "lumbar_bending", "lumbar_rotation", "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", \
                    "wrist_flex_r", "wrist_dev_r", "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l", "knee_valgus_r", "knee_valgus_l", \
                        "shin_angle_r", "shin_angle_l", "ankle_add_r", "ankle_add_l", "arm_horz_flex_r", "arm_horz_flex_l", "ankle_inversion_r", "ankle_inversion_l"]

        opensim_simple_metric_dict_L_R = {"hip_flexion": {"Hip Flex. (°)": "+", "Hip Ext. (°)": "-"}, "hip_adduction": {"Hip Add. (°)": "-", "Hip Abd. (°)": "+"}, "hip_rotation": {"Hip ER (°)": "+", "Hip IR (°)": "-"}, "knee_angle": {"Knee Flex. (°)": "+", "Knee Ext. (°)": "-"}, \
            "ankle_angle": {"Ankle DF (°)": "+", "Ankle PF (°)": "-"}, "arm_flex": {"Shoulder Flex. (°)": "+", "Shoulder Ext. (°)": "-"}, "arm_add": {"Shoulder Add. (°)": "-", "Shoulder Abd. (°)": "+"}, "arm_rot": {"Shoulder ER (°)": "+", "Shoulder IR (°)": "-"}, \
                "elbow_flex": {"Elbow Flex. (°)": "+", "Elbow Ext. (°)": "-"}, "pro_sup": {"Elbow Pro. (°)": "+", "Elbow Sup. (°)": "-"}, "wrist_flex": {"Wrist Flex. (°)": "+", "Wrist Ext. (°)": "-"}, "wrist_dev": {"Wrist RD (°)": "-", "Wrist UD (°)": "+"}, "knee_valgus": {"Knee Val. (°)": "+", "Knee Var. (°)": "-"}, \
                    "shin_angle": {"Shin Angle (°)": "+"}, "ankle_add": {"Ankle Add. (°)": "-", "Ankle Abd. (°)": "+"}, "arm_horz_flex": {"Arm Horz. Flex. (°)": "+", "Arm Horz. Ext. (°)": "-"}, "ankle_inversion": {"Ankle Inversion (°)": "-", "Ankle Eversion (°)": "+"}}
 
        opensim_simple_metric_dict_center = {"waist_extension": {"Pel. Tilt (°)": {"F": "+" , "B": "-"}}, "waist_bending": {"Pel. SB (°)" : {"L": "-", "R": "+"}}, "waist_rotation": {"Pel. Rot. (°)": {"L": "-", "R": "+"}}, "lumbar_extension": {"Torso Ext. (°)": {"F": "+", "B": "-"}}, \
            "lumbar_bending" : {"Torso Bend (°)": {"L": "-", "R": "+"}}, "lumbar_rotation": {"Torso Rot. (°)": {"L": "-", "R": "+"}}, "trunk_extension": {"Trunk Ext. (Relative to Ground) (°)": {"F": "+", "B": "-"}}}


        single_metrics = ["Depth (in)", "Timing (s)"]

        ## Metrics to add & can add 
        # Knee Valgus/Vargus (Done), Thigh Depth/Depth (Done), Shin Angle????, Timing/Rhythm (Arm Velo.), Lateral Shift 

        # For right - knee varus is +, knee valgus is - //// For left - knee varus is -, knee valgus is + 

        ## MARK: Change joint to metric
        ## Filter joint names to be simple names
        filtered_joints = []
        for joint in selected_joints: 
            if joint[-2:] == "_l" or joint[-2:] == "_r": 
                if joint[:-2] not in filtered_joints: 
                    filtered_joints.append(joint[:-2])
            else: 
                filtered_joints.append(joint)
        # print(filtered_joints)

        # MARK: Filter out camera perspective non-specific angles
        # if cv_mode == 1: 
        #     print("Filtering Out Camera-Facing Angles")
        #     opensim_simple_metric_dict_L_R = {"hip_flexion": {"Hip Flex. (°)": "+"}, "hip_adduction": {"Hip Add. (°)": "-", "Hip Abd. (°)": "+"}, "hip_rotation": {"Hip ER (°)": "+", "Hip IR (°)": "-"}, "knee_angle": {"Knee Flex. (°)": "+"}, \
        #     "ankle_angle": {"Ankle DF (°)": "+", "Ankle PF (°)": "-"}, "arm_flex": {"Shoulder Flex. (°)": "+", "Shoulder Ext. (°)": "-"}, "arm_add": {"Shoulder Add. (°)": "-", "Shoulder Abd. (°)": "+"}, "arm_rot": {"Shoulder ER (°)": "+", "Shoulder IR (°)": "-"}, \
        #         "elbow_flex": {"Elbow Flex. (°)": "+", "Elbow Ext. (°)": "-"}, "pro_sup": {"Elbow Pro. (°)": "+", "Elbow Sup. (°)": "-"}, "wrist_flex": {"Wrist Flex. (°)": "+", "Wrist Ext. (°)": "-"}, "wrist_dev": {"Wrist RD (°)": "-", "Wrist UD (°)": "+"}, "knee_valgus": {"Knee Val. (°)": "+", "Knee Var. (°)": "-"}, \
        #             "shin_angle": {"Shin Angle (°)": "+"}, "ankle_add": {"Ankle Add. (°)": "-", "Ankle Abd. (°)": "+"}, "arm_horz_flex": {"Arm Horz. Flex. (°)": "+", "Arm Horz. Ext. (°)": "-"}, "ankle_inversion": {"Ankle Inversion (°)": "-", "Ankle Eversion (°)": "+"}}

        #     opensim_simple_metric_dict_center = {"waist_extension": {"Pel. Tilt (°)": {"F": "+" , "B": "-"}}, "waist_bending": {"Pel. SB (°)" : {"L": "-", "R": "+"}}, "waist_rotation": {"Pel. Rot. (°)": {"L": "-", "R": "+"}}, "lumbar_extension": {"Torso Ext. (°)": {"F": "+"}}, \
        #     "lumbar_bending" : {"Torso Bend (°)": {"L": "-", "R": "+"}}, "lumbar_rotation": {"Torso Rot. (°)": {"L": "-", "R": "+"}}, "trunk_extension": {"Trunk Ext. (Relative to Ground) (°)": {"F": "+", "B": "-"}}}

        # elif cv_mode == 2 or cv_angle == 3: 
        #     print("Filtering Out Side-Facing Angles")
        #     opensim_simple_metric_dict_L_R = {"hip_flexion": {"Hip Flex. (°)": "+", "Hip Ext. (°)": "-"}, "knee_angle": {"Knee Flex. (°)": "+", "Knee Ext. (°)": "-"}, \
        #     "ankle_angle": {"Ankle DF (°)": "+", "Ankle PF (°)": "-"}, "arm_flex": {"Shoulder Flex. (°)": "+", "Shoulder Ext. (°)": "-"}, "arm_rot": {"Shoulder ER (°)": "+", "Shoulder IR (°)": "-"}, \
        #         "elbow_flex": {"Elbow Flex. (°)": "+", "Elbow Ext. (°)": "-"}, "pro_sup": {"Elbow Pro. (°)": "+", "Elbow Sup. (°)": "-"}, "wrist_flex": {"Wrist Flex. (°)": "+", "Wrist Ext. (°)": "-"}, "wrist_dev": {"Wrist RD (°)": "-", "Wrist UD (°)": "+"}, "knee_valgus": {"Knee Val. (°)": "+", "Knee Var. (°)": "-"}, \
        #             "shin_angle": {"Shin Angle (°)": "+"}, "ankle_add": {"Ankle Add. (°)": "-", "Ankle Abd. (°)": "+"}, "arm_horz_flex": {"Arm Horz. Flex. (°)": "+", "Arm Horz. Ext. (°)": "-"}, "ankle_inversion": {"Ankle Inversion (°)": "-", "Ankle Eversion (°)": "+"}}
 
        #     opensim_simple_metric_dict_center = {"waist_extension": {"Pel. Tilt (°)": {"F": "+" , "B": "-"}}, "waist_bending": {"Pel. SB (°)" : {"L": "-", "R": "+"}}, "waist_rotation": {"Pel. Rot. (°)": {"L": "-", "R": "+"}}, "lumbar_extension": {"Torso Ext. (°)": {"F": "+", "B": "-"}}, \
        #     "lumbar_bending" : {"Torso Bend (°)": {"L": "-", "R": "+"}}, "lumbar_rotation": {"Torso Rot. (°)": {"L": "-", "R": "+"}}, "trunk_extension": {"Trunk Ext. (Relative to Ground) (°)": {"F": "+", "B": "-"}}}

        ## Calculate stats of all metrics and store
        for joint in filtered_joints:
            if joint in opensim_simple_metric_dict_L_R:
                # print(joint)
                # if orientation == "Right":
                #     for joint_breakdown_name in opensim_simple_metric_dict_L_R[joint]:
                #         metrics_dict = {}
                #         metrics_dict = {"R": getStats(opensim_simple_metric_dict_L_R[joint][joint_breakdown_name], osim_data[joint+"_r"])} 
                #         metrics[joint_breakdown_name] = metrics_dict 
                # elif orientation == "Left": 
                #     for joint_breakdown_name in opensim_simple_metric_dict_L_R[joint]:
                #         metrics_dict = {} 
                #         metrics_dict = {"L": getStats(opensim_simple_metric_dict_L_R[joint][joint_breakdown_name], osim_data[joint+"_l"])} 
                #         metrics[joint_breakdown_name] = metrics_dict 
                # elif orientation == None or orientation == "": 
                for joint_breakdown_name in opensim_simple_metric_dict_L_R[joint]:
                    # print(joint_breakdown_name)
                    metrics_dict = {}
                    ang_dict_r, vel_dict_r, acc_dict_r, dec_dict_r = getStats(opensim_simple_metric_dict_L_R[joint][joint_breakdown_name], osim_data[joint+"_r"], detailsID)
                    ang_dict_l, vel_dict_l, acc_dict_l, dec_dict_l = getStats(opensim_simple_metric_dict_L_R[joint][joint_breakdown_name], osim_data[joint+"_l"], detailsID)

                    # Make metric dictionary 
                    ang_metrics[joint_breakdown_name] = {"L": ang_dict_l, "R": ang_dict_r}
                    vel_metrics[joint_breakdown_name] = {"L": vel_dict_l, "R": vel_dict_r} 
                    acc_metrics[joint_breakdown_name] = {"L": acc_dict_l, "R": acc_dict_r}  
                    dec_metrics[joint_breakdown_name] = {"L": dec_dict_l, "R": dec_dict_r}   

                    # metrics_dict["R"] = getStats(opensim_simple_metric_dict_L_R[joint][joint_breakdown_name], osim_data[joint+"_r"])
                    # metrics_dict["L"] = getStats(opensim_simple_metric_dict_L_R[joint][joint_breakdown_name], osim_data[joint+"_l"]) 
                    # metrics[joint_breakdown_name] = metrics_dict

            ## For center joints (torso + pelvis) 
            elif joint in opensim_simple_metric_dict_center:
                for joint_breakdown_name in opensim_simple_metric_dict_center[joint]:
                    ang_joint_dict = {} 
                    vel_joint_dict = {}
                    acc_joint_dict = {}
                    dec_joint_dict = {}                   
                    for metric_dir in opensim_simple_metric_dict_center[joint][joint_breakdown_name]:
                        ang_dict, vel_dict, acc_dict, dec_dict = getStats(opensim_simple_metric_dict_center[joint][joint_breakdown_name][metric_dir], osim_data[joint], detailsID)
                        ang_joint_dict[metric_dir] = ang_dict
                        vel_joint_dict[metric_dir] = vel_dict 
                        acc_joint_dict[metric_dir] = acc_dict 
                        dec_joint_dict[metric_dir] = dec_dict
                    ang_metrics[joint_breakdown_name] = ang_joint_dict
                    vel_metrics[joint_breakdown_name] = vel_joint_dict
                    acc_metrics[joint_breakdown_name] = acc_joint_dict
                    dec_metrics[joint_breakdown_name] = dec_joint_dict


                    #     metrics_dict[metric_dir] = getStats(opensim_simple_metric_dict_center[joint][joint_breakdown_name][metric_dir], osim_data[joint]["Ang"])  
                    # metrics[joint_breakdown_name] = metrics_dict

            ## For single metrics or complex metrics 
            # elif joint in single_metrics:
            #     if joint == "Depth (in)":
            #         depth_list_r = pos_calcnr_df['state_1'].values.tolist()
            #         depth_list_r[:] = [abs(number - depth_list_r[0]) for number in depth_list_r]
            #         depth_list_l = pos_calcnl_df['state_1'].values.tolist()
            #         depth_list_l[:] = [abs(number - depth_list_l[0]) for number in depth_list_l] 
            #         if orientation == "Right":
            #             metrics_dict = {} 
            #             metrics_dict["R"] = {"Max": max(depth_list_r)*39.3701}
            #             metrics[joint] = metrics_dict 
            #         elif orientation == "Left":
            #             metrics_dict = {} 
            #             metrics_dict["L"] = {"Max": max(depth_list_l)*39.3701}
            #             metrics[joint] = metrics_dict 
            #         elif orientation == None or orientation == "": 
            #             metrics_dict = {} 
            #             r_val = max(depth_list_r)*39.3701
            #             l_val = max(depth_list_l)*39.3701
            #             metrics_dict["Max"] = mean([r_val, l_val]) 
            #             metrics[joint] = metrics_dict 

            if orientation == "Left" or orientation == "Right":
                try: 
                    ## Creation of shoulder arc angles
                    if 'arm_rot' in filtered_joints: 
                        if "Max" in ang_metrics["Shoulder ER (°)"]["L"] and "Max" in ang_metrics["Shoulder IR (°)"]["L"]:
                            left_arc_dict = {"Max": ang_metrics["Shoulder ER (°)"]["L"]["Max"] + ang_metrics["Shoulder IR (°)"]["L"]["Max"], "Avg": ang_metrics["Shoulder ER (°)"]["L"]["Avg"] + ang_metrics["Shoulder IR (°)"]["L"]["Avg"], "AvgMax": ang_metrics["Shoulder ER (°)"]["L"]["AvgMax"] + ang_metrics["Shoulder IR (°)"]["L"]["AvgMax"]}
                        else: 
                            left_arc_dict = {}
                        
                        if "Max" in ang_metrics["Shoulder ER (°)"]["R"] and "Max" in ang_metrics["Shoulder IR (°)"]["R"]:
                            right_arc_dict = {"Max": ang_metrics["Shoulder ER (°)"]["R"]["Max"] + ang_metrics["Shoulder IR (°)"]["R"]["Max"], "Avg": ang_metrics["Shoulder ER (°)"]["R"]["Avg"] + ang_metrics["Shoulder IR (°)"]["R"]["Avg"], "AvgMax": ang_metrics["Shoulder ER (°)"]["R"]["AvgMax"] + ang_metrics["Shoulder IR (°)"]["R"]["AvgMax"]}
                        else: 
                            right_arc_dict = {}

                        ang_metrics["Shoulder Rotation Arc"] = {"L": left_arc_dict, "R": right_arc_dict}
                except: 
                    continue

                try: 
                    ## Creation hip arc angles
                    if 'hip_rotation' in filtered_joints: 
                        if "Max" in ang_metrics["Hip ER (°)"]["L"] and "Max" in ang_metrics["Hip IR (°)"]["L"]:
                            left_arc_dict = {"Max": ang_metrics["Hip ER (°)"]["L"]["Max"] + ang_metrics["Hip IR (°)"]["L"]["Max"], "Avg": ang_metrics["Hip ER (°)"]["L"]["Avg"] + ang_metrics["Hip IR (°)"]["L"]["Avg"], "AvgMax": ang_metrics["Hip ER (°)"]["L"]["AvgMax"] + ang_metrics["Hip IR (°)"]["L"]["AvgMax"]}
                        else: 
                            left_arc_dict = {}
                        
                        if "Max" in ang_metrics["Hip ER (°)"]["R"] and "Max" in ang_metrics["Hip IR (°)"]["R"]:
                            right_arc_dict = {"Max": ang_metrics["Hip ER (°)"]["R"]["Max"] + ang_metrics["Hip IR (°)"]["R"]["Max"], "Avg": ang_metrics["Hip ER (°)"]["R"]["Avg"] + ang_metrics["Hip IR (°)"]["R"]["Avg"], "AvgMax": ang_metrics["Hip ER (°)"]["R"]["AvgMax"] + ang_metrics["Hip IR (°)"]["R"]["AvgMax"]}
                        else: 
                            right_arc_dict = {}    
                            
                        ang_metrics["Hip Rotation Arc"] = {"L": left_arc_dict, "R": right_arc_dict}
                except: 
                    continue

        assessment_metrics = {"Ang": ang_metrics, "Vel": vel_metrics, "Acc": acc_metrics, "Dec": dec_metrics} 

        return assessment_metrics

    def getStats(movement_side, data, detailsID):

        ang_dict = {}
        vel_dict = {} 
        acc_dict = {}
        dec_dict = {} 
        ## Calculate max angle, avg angle, max vel, avg vel, max acc, avg acc, max dec, avg dec 

        ## Non rep based atm

        ## Joint angles  
        if movement_side == "+":
            
            if [n for n in data["Ang"][5:None] if n>0] != []:
                ## Angle & Velocity
                # if [n for n in data["Ang"][5:None] if n>0] != []:
                ang_dict["Max"] = abs(max([n for n in data["Ang"][5:None] if n>0]))
                ang_dict["Avg"] = abs(mean([n for n in data["Ang"][5:None] if n>0]))

                vel_dict["Max"] = max([abs(t[1]) for t in zip(data['Ang'][5:None], data['Vel'][5:None]) if t[0] > 0])
                vel_dict["Avg"] = mean([abs(t[1]) for t in zip(data['Ang'][5:None], data['Vel'][5:None]) if t[0] > 0]) 
                # else: 
                #     ang_dict["Max"] = 0
                #     ang_dict["Avg"] = 0
                #     vel_dict["Max"] = 0 
                #     vel_dict["Avg"] = 0

                ## Acceleration + Decelration 
                acc_list = [t[1] for t in zip(data['Ang'][5:None], data['Acc'][5:None]) if t[0] > 0] 
                
                if [n for n in acc_list if n>0] != []:  
                    # Acc. 
                    acc_dict["Max"] = abs(max([n for n in acc_list if n>0])) 
                    acc_dict["Avg"] = abs(mean([n for n in acc_list if n>0])) 
                else: 
                    acc_dict["Max"] = 0
                    acc_dict["Avg"] = 0
            
                if [n for n in acc_list if n<0] != []:
                    # Dec. 
                    dec_dict["Max"] = max([abs(n) for n in acc_list if n<0])
                    dec_dict["Avg"] = mean([abs(n) for n in acc_list if n<0])
                else: 
                    dec_dict["Max"] = 0 
                    dec_dict["Avg"] = 0 

                joint_data_for_reps = [0 if ele < 0 else ele for ele in data["Ang"]]
                joint_rep_dict = assessmentReps(joint_data_for_reps)

                ## Avg Max Calculations
                if joint_rep_dict != {}: 
                    if joint_rep_dict["NumReps"] > 0: 
                        ang_max_list = []
                        vel_max_list = [] 
                        acc_max_list = [] 
                        dec_max_list = [] 
                        for window in joint_rep_dict["Windows"]:
                            # print(window)  
                            ## Angle
                            ang_max = abs(max([n for n in data["Ang"][window[0]:window[1]] if n>0]))
                            ang_max_list.append(ang_max)
                            # print(ang_max)
                            ## Velocity 
                            vel_max = max([abs(t[1]) for t in zip(data['Ang'][window[0]:window[1]], data['Vel'][window[0]:window[1]]) if t[0] > 0])
                            vel_max_list.append(vel_max)

                            ## Acceleration + Decelration 
                            acc_list = [t[1] for t in zip(data['Ang'][window[0]:window[1]], data['Acc'][window[0]:window[1]]) if t[0] > 0] 
                            
                            if [n for n in acc_list if n>0] != []:  
                                # Acc.
                                acc_max = abs(max([n for n in acc_list if n>0])) 
                            else: 
                                acc_max = 0
                            
                            acc_max_list.append(acc_max)
                        
                            if [n for n in acc_list if n<0] != []:
                                # Dec. 
                                dec_max = max([abs(n) for n in acc_list if n<0])
                            else: 
                                dec_max = 0

                            dec_max_list.append(dec_max)

                        ang_dict["AvgMax"] = mean(ang_max_list)
                        vel_dict["AvgMax"] = mean(vel_max_list)
                        acc_dict["AvgMax"] = mean(acc_max_list)
                        dec_dict["AvgMax"] = mean(dec_max_list)
                    else: 
                        ang_dict["AvgMax"] = ang_dict["Max"]
                        vel_dict["AvgMax"] = vel_dict["Max"]
                        acc_dict["AvgMax"] = acc_dict["Max"]
                        dec_dict["AvgMax"] = dec_dict["Max"]
                else: 
                    ang_dict["AvgMax"] = ang_dict["Max"]
                    vel_dict["AvgMax"] = vel_dict["Max"]
                    acc_dict["AvgMax"] = acc_dict["Max"]
                    dec_dict["AvgMax"] = dec_dict["Max"] 

            else: 
                ang_dict = {}
                vel_dict = {}
                acc_dict = {}
                dec_dict = {}
            #     ## Angle
            #     ang_dict["Max"] = 0
            #     ang_dict["Avg"] = 0

            #     ## Velocity 
            #     vel_dict["Max"] = 0
            #     vel_dict["Avg"] = 0

            #     ang_dict["AvgMax"] = 0
            #     vel_dict["AvgMax"] = 0
            #     acc_dict["AvgMax"] = 0
            #     dec_dict["AvgMax"] = 0


        elif movement_side == "-":
  
            if [n for n in data["Ang"][5:None] if n<0] != []:
                ## Angle & Velocity 
                # if [n for n in data["Ang"][5:None] if n<0]:

                # Logic for Shouler IR for Shoulder IR/ER
                if detailsID == "296C21D6-431C-4D09-9484-60575CE88A9D": 
                    try: 
                        test_data = np.array(data["Ang"][5:None])
                        inv_test_data = -test_data

                        height_threshold = max(5, statistics.mean(inv_test_data) + (max(inv_test_data) - statistics.mean(inv_test_data))*0.4)
                        peaks, _ = find_peaks(inv_test_data, width = 30, prominence=4)

                        local_minima = test_data[peaks]
                        local_minima =  [x for x in local_minima if x < 0]
                        print("Local minima:", local_minima)
                        ang_dict["Max"] = max([abs(n) for n in local_minima])
                        print("Max Shoulder IR")
                        print(ang_dict["Max"])
                    except Exception as e: 
                        ang_dict["Max"] = max([abs(n) for n in data["Ang"][5:None] if n<0])
                        print(e)
                else: 
                    ang_dict["Max"] = max([abs(n) for n in data["Ang"][5:None] if n<0])

                

                ang_dict["Avg"] = mean([abs(n) for n in data["Ang"][5:None] if n<0])

                ## Velocity 
                vel_dict["Max"] = max([abs(t[1]) for t in zip(data['Ang'][5:None], data['Vel'][5:None]) if t[0]<0])
                vel_dict["Avg"] = mean([abs(t[1]) for t in zip(data['Ang'][5:None], data['Vel'][5:None]) if t[0]<0]) 
                # else: 
                #     ang_dict["Max"] = 0 
                #     ang_dict["Avg"] = 0 
                #     vel_dict["Max"] = 0
                #     vel_dict["Avg"] = 0

                ## Acceleration + Decelration 
                acc_list = [t[1] for t in zip(data['Ang'][5:None], data['Acc'][5:None]) if t[0]<0] 

                if [n for n in acc_list if n>0] != []:  
                    # Acc. 
                    acc_dict["Max"] = abs(max([n for n in acc_list if n>0])) 
                    acc_dict["Avg"] = abs(mean([n for n in acc_list if n>0])) 
                else: 
                    acc_dict["Max"] = 0
                    acc_dict["Avg"] = 0

                if [n for n in acc_list if n<0] != []: 
                    # Dec. 
                    dec_dict["Max"] = max([abs(n) for n in acc_list if n<0])
                    dec_dict["Avg"] = mean([abs(n) for n in acc_list if n<0]) 
                else: 
                    dec_dict["Max"] = 0 
                    dec_dict["Avg"] = 0 


                joint_data_for_reps = [0 if ele > 0 else ele for ele in data["Ang"]]
                joint_rep_dict = assessmentReps(joint_data_for_reps)

                ## Avg Max Calculations
                if joint_rep_dict != {}: 
                    if joint_rep_dict["NumReps"] > 0: 
                        ang_max_list = []
                        vel_max_list = [] 
                        acc_max_list = [] 
                        dec_max_list = [] 
                        for window in joint_rep_dict["Windows"]:
                            # print(window) 
                            ## Angle
                            ang_max = abs(max([n for n in data["Ang"][window[0]:window[1]] if n<0]))
                            ang_max_list.append(ang_max)

                            ## Velocity 
                            vel_max = max([abs(t[1]) for t in zip(data['Ang'][window[0]:window[1]], data['Vel'][window[0]:window[1]]) if t[0] < 0])
                            vel_max_list.append(vel_max)

                            ## Acceleration + Decelration 
                            acc_list = [t[1] for t in zip(data['Ang'][window[0]:window[1]], data['Acc'][window[0]:window[1]]) if t[0] < 0] 
                            
                            if [n for n in acc_list if n>0] != []:  
                                # Acc.
                                acc_max = abs(max([n for n in acc_list if n>0])) 
                            else: 
                                acc_max = 0
                            
                            acc_max_list.append(acc_max)
                        
                            if [n for n in acc_list if n<0] != []:
                                # Dec. 
                                dec_max = max([abs(n) for n in acc_list if n<0])
                            else: 
                                dec_max = 0

                            dec_max_list.append(dec_max)

                        ang_dict["AvgMax"] = mean(ang_max_list)
                        vel_dict["AvgMax"] = mean(vel_max_list)
                        acc_dict["AvgMax"] = mean(acc_max_list)
                        dec_dict["AvgMax"] = mean(dec_max_list)
                    else: 
                        ang_dict["AvgMax"] = ang_dict["Max"]
                        vel_dict["AvgMax"] = vel_dict["Max"]
                        acc_dict["AvgMax"] = acc_dict["Max"]
                        dec_dict["AvgMax"] = dec_dict["Max"] 
                else: 
                        ang_dict["AvgMax"] = ang_dict["Max"]
                        vel_dict["AvgMax"] = vel_dict["Max"]
                        acc_dict["AvgMax"] = acc_dict["Max"]
                        dec_dict["AvgMax"] = dec_dict["Max"]  
            
            else: 
                ang_dict = {}
                vel_dict = {}
                acc_dict = {}
                dec_dict = {}
            #     ## Angle
            #     ang_dict["Max"] = 0
            #     ang_dict["Avg"] = 0

            #     ## Velocity 
            #     vel_dict["Max"] = 0
            #     vel_dict["Avg"] = 0

            #     ang_dict["AvgMax"] = 0
            #     vel_dict["AvgMax"] = 0
            #     acc_dict["AvgMax"] = 0
            #     dec_dict["AvgMax"] = 0

        ## DEFUNCT - OLD
        # stat_dict = {}
        # if movement_side == "+":
        #     if [n for n in data_list if n>0] != []:
        #         stat_dict["Max"] = max([n for n in data_list if n>0])
        #         stat_dict["Avg"] = mean([n for n in data_list if n>0])
        #         stat_dict["Min"] = min([n for n in data_list if n>0])
        # elif movement_side == "-":
        #     if [n for n in data_list if n<0] != []:
        #         stat_dict["Max"] = abs(min([n for n in data_list if n<0]))
        #         stat_dict["Avg"] = abs(mean([n for n in data_list if n<0]))
        #         stat_dict["Min"] = abs(max([n for n in data_list if n<0]))

        return ang_dict, vel_dict, acc_dict, dec_dict

    # Function to obtain MLKit segment vectors
    def get_mlkit_segment_vectors(mlkit_dict):
        array_dict = {}
        for landmark, dict_list in mlkit_dict.items():
            vec_list = []
            for vec_dict in dict_list:
                vec = np.array([vec_dict["x"], vec_dict["y"], vec_dict["z"]])
                vec_list.append(vec)
            vec_nd = np.array(vec_list)
            array_dict[landmark] = vec_nd

        temp_seg_dict = {}

        rot_quat = qd.quaternion_from_euler([0, 0, -90])
        rot_quat2 = qd.quaternion_from_euler([0, 180, 0])

        # rot_quat = qd.quaternion_from_euler([0, 0, 0])
        # rot_quat2 = qd.quaternion_from_euler([0, 0, 0])

        temp_seg_dict["RF"] = array_dict["right_heel"] - array_dict["right_foot_index"]
        temp_seg_dict["LF"]  = array_dict["left_heel"] - array_dict["left_foot_index"]

        temp_seg_dict["RF_vert"] = array_dict["right_ankle"] - array_dict["right_heel"]
        temp_seg_dict["LF_vert"]  = array_dict["left_ankle"] - array_dict["left_heel"]

        temp_seg_dict["RLL"] = array_dict["right_knee"] - array_dict["right_ankle"]
        temp_seg_dict["LLL"]  = array_dict["left_knee"] - array_dict["left_ankle"]

        temp_seg_dict["RUL"] = array_dict["right_hip"] - array_dict["right_knee"]
        temp_seg_dict["LUL"]  = array_dict["left_hip"] - array_dict["left_knee"]
        
        temp_seg_dict["CLB_horiz"]  = array_dict["right_hip"] - array_dict["left_hip"]
        temp_seg_dict["CUB_horiz"]  = array_dict["right_shoulder"] - array_dict["left_shoulder"]

        hip_mid_array = (array_dict["right_hip"] + array_dict["left_hip"]) / 2
        shoulder_mid_array = (array_dict["right_shoulder"] + array_dict["left_shoulder"]) / 2

        temp_seg_dict["CUB_vert"]  = shoulder_mid_array - hip_mid_array

        temp_seg_dict["NH"] = array_dict["nose"] - shoulder_mid_array

        temp_seg_dict["RUA"]  = array_dict["right_shoulder"] - array_dict["right_elbow"]
        temp_seg_dict["LUA"]  = array_dict["left_shoulder"] - array_dict["left_elbow"]

        temp_seg_dict["RLA"]  = array_dict["right_elbow"] - array_dict["right_wrist"]
        temp_seg_dict["LLA"]  = array_dict["left_elbow"] - array_dict["left_wrist"]


        r_finger_mid_array = (array_dict["right_pinky"] + array_dict["right_index"]) / 2
        l_finger_mid_array = (array_dict["left_pinky"] + array_dict["left_index"]) / 2

        temp_seg_dict["RH"] = array_dict["right_wrist"] - r_finger_mid_array
        temp_seg_dict["LH"]  = array_dict["left_wrist"] - l_finger_mid_array

        temp_seg_dict["RTOR"] = array_dict["right_shoulder"] - array_dict["right_hip"]
        temp_seg_dict["LTOR"] = array_dict["left_shoulder"] - array_dict["left_hip"]

        final_segment_dict = {}

        for segment, temp_vec_array in temp_seg_dict.items():
            n = temp_vec_array.shape[0]
            m = temp_vec_array.shape[1]

            final_vec_array = np.zeros((n, m))

            for i in range(n):
                vec = temp_vec_array[i, :]
                norm_vec = qd.normalize_vector3(vec)
                norm_vec = qd.rotate_vector_by_quaternion(rot_quat, norm_vec)
                norm_vec = qd.rotate_vector_by_quaternion(rot_quat2, norm_vec)

                final_vec_array[i, :] = norm_vec
            
            final_segment_dict[segment] = final_vec_array

        return final_segment_dict   

    # Function to return world landmarks + jump height (FOR NOW)
    def linearKinematics(landmark_data, athlete_height_in): 
        jump_height = 0
        athlete_height = float(athlete_height_in) * float(0.0254)

        vv_height = ((landmark_data["right_eye"][0]['x'] + landmark_data["left_eye"][0]['x'])/2) - ((landmark_data["right_heel"][0]['x'] + landmark_data["left_heel"][0]['x'])/2)
 
        prop = abs(athlete_height/vv_height)

        hip_world_data = []
        for i in range(len(landmark_data["right_hip"])): 
            hip_calc = ((landmark_data["right_hip"][i]['x'] + landmark_data["left_hip"][i]['x'])/2) * prop 
            hip_world_data.append(hip_calc)

        jump_height = max(hip_world_data) - hip_world_data[0]

        return jump_height

    # Function to download external video 
    def download_video_from_s3(bucket_name, video_key, local_file_path):
        """
        Download a video from S3, waiting up to 5 minutes for the file to appear if it is not immediately available.
        """
        s3 = boto3.client('s3')
        timeout = 300  # Timeout in seconds (5 minutes)
        poll_interval = 5  # Polling interval in seconds
        start_time = time.time()

        while True:
            try:
                # Check if the object exists
                s3.head_object(Bucket=bucket_name, Key=video_key)
                print(f"File {video_key} found in S3. Starting download...")
                break
            except s3.exceptions.ClientError as e:
                # Check if the error is due to the file not existing
                if e.response['Error']['Code'] == '404':
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout:
                        raise TimeoutError(f"File {video_key} not found in S3 after waiting for {timeout} seconds.")
                    print(f"File {video_key} not found. Retrying in {poll_interval} seconds...")
                    time.sleep(poll_interval)
                else:
                    # Re-raise other exceptions
                    raise

        # Download the file once it is found
        s3.download_file(bucket_name, video_key, local_file_path)
        print(f"Downloaded {video_key} from S3 to {local_file_path}")

    # Function to process external video via mediapipe
    def process_video_with_mediapipe(video_path):
        """Process a video with MediaPipe and generate a landmark dictionary with pixel coordinates."""
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True)
        landmark_dict = {}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1 / fps
        print(f"Frame rate: {fps}")

        frame_index = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            height, width, _ = frame.shape  # Get frame dimensions

            # Convert the BGR image to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                timestamp = time.time()
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    label = mp_pose.PoseLandmark(id).name.lower()
                    
                    # Convert normalized values to pixel coordinates
                    x_pixel = int(landmark.x * width)
                    y_pixel = int(landmark.y * height)
                    z_real = landmark.z * width  # Depth is relative to image width

                    if label not in landmark_dict:
                        landmark_dict[label] = []

                    landmark_dict[label].append({
                        "t": timestamp + frame_index * frame_time,
                        "x": y_pixel,
                        "y": -x_pixel,
                        "z": z_real,  # Keeping Z in pixel units for 3D rendering
                        "label": label
                    })

            frame_index += 1

        cap.release()
        return landmark_dict, frame_time

    # Function delete external video from EC2
    def delete_local_file(file_path):
        """Delete a local file."""
        try:
            os.remove(file_path)
            print(f"Deleted local file: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")

    ## Function to transform the data and convert to JSON
    def transformData(file, orientation, selected_joints, wkt_id, sensor_mode, record_mode, cv_mode, exc_details, athlete_height_in, exc_id, org_id, external_vid, vid_frame_time, bucket, key) :
        if external_vid: 
            print("Found external video")
            local_file_path = 'external_video.mp4'

            # Download video from S3
            video_key = 'videoCaptures/' + key[7:]
            download_video_from_s3(bucket, video_key, local_file_path)

            from instanthmr.adapter import process_video_with_instanthmr  # HERE
            landmark_data, frame_time = process_video_with_instanthmr(local_file_path)  # HERE

            if vid_frame_time != "": 
                final_frame_time = float(1/vid_frame_time)
                print("External Vid Frame Rate Found")
                print(final_frame_time)
            else: 
                final_frame_time = frame_time

            for joint in file: 
                for landmark in landmark_data: 
                    if landmark == joint: 
                        if len(landmark_data[landmark]) > 0: 
                            file[joint] = landmark_data[landmark]
                        else: 
                            print(f"No data found for {landmark}")
                    elif landmark == "mouth_left": 
                        if len(landmark_data[landmark]) > 0: 
                            file["left_mouth"] = landmark_data[landmark]
                        else: 
                            print(f"No data found for {landmark}")
                    elif landmark == "mouth_right": 
                        if len(landmark_data[landmark]) > 0: 
                            file["right_mouth"] = landmark_data[landmark]
                        else: 
                            print(f"No data found for {landmark}")

            # Delete the local video file
            delete_local_file(local_file_path)

        # Sample rates 
        try: 
            if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 
                sample_dict = rawSampleRate_cv(file)
            else: 
                sample_dict = rawSampleRate(file)
        except: 
            sample_dict = {}

        
        # All of the smoothed (original) data to be put in JSON (includes quaternions, linear acceleration, and joint angles)
        trim3 = False
        if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 
            if external_vid: 
                min_time, max_time = findTime_CV(file)
                org_smooth_data, trim3 = extractQuat_SmoothExt_cv(file, min_time, max_time) 
            else: 
                min_time, max_time = findTime_CV(file)
                org_smooth_data, trim3 = extractQuat_Smooth_cv(file, min_time, max_time)   
        else: 
            min_time, max_time = findTime(file)
            org_smooth_data, org_smooth_df = extractQuat_Smooth(file, min_time, max_time)   

        # Final transformation of data
        transformed_data = {}
        summary_metrics_dict = {}

        if record_mode != 1:
            print("Only doing multi-sensor mode")
            data_cp = copy.deepcopy(org_smooth_data)

            # Joint angle recalculations
            ## VALOR VISION
            if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 

                print("Doing CV joint angles")

                mlk_landmark_dict = {landmark : data for landmark, data in org_smooth_data.items() if isinstance(data, list) and len(data) > 0}

                # generate segments from mlk landmark data
                mlk_segment_dict = get_mlkit_segment_vectors(mlk_landmark_dict)

                dict_of_all_angles = amCV.update_angles_mlk(mlk_segment_dict, cv_mode)
  
                org_smooth_data['unityJointAnglesDict'] = dict_of_all_angles

                joint_df = pd.DataFrame.from_dict(org_smooth_data["unityJointAnglesDict"])

                for column in joint_df:
                    if column != "ival" and column != "frame": 
                        joint_df[column] = joint_df[column].ewm(span=10).mean()

                new_dict = joint_df.to_dict(orient="list")
                org_smooth_data["unityJointAnglesDict"] = new_dict

                ## Summary Metrics for Report pipeline 
                if wkt_id != "freesession" and wkt_id != "": 
                    dict_angles_copy = copy.deepcopy(dict_of_all_angles)
                    mlk_segment_copy = copy.deepcopy(mlk_segment_dict)
                    mlk_landmark_copy = copy.deepcopy(mlk_landmark_dict)
                    ang_vel_dict_copy = {}
                    try: 
                        org_data_cp = copy.deepcopy(org_smooth_data)
                        joint_kinematics_dict_cp = jointKinematics(org_data_cp) 
                        for joint in joint_kinematics_dict_cp: 
                            ang_vel_dict_copy[joint] = joint_kinematics_dict_cp[joint]["Vel"]
                    except: 
                        ang_vel_dict_copy = {}

                    ## Removing ~ last 3 seconds prior to repot overview score calcs
                    if trim3: 
                        for joint in dict_angles_copy:
                            if len(dict_angles_copy[joint]) != 0: 
                                dict_angles_copy[joint] = dict_angles_copy[joint][:-187]  
                            else:
                                dict_angles_copy[joint] = dict_angles_copy[joint]  
                        
                        for segment in mlk_segment_copy:
                            if len(mlk_segment_copy[segment]) != 0: 
                                mlk_segment_copy[segment] = mlk_segment_copy[segment][:-187]  
                            else:
                                mlk_segment_copy[segment] = mlk_segment_copy[segment]  
                        
                        for landmark in mlk_landmark_copy:
                            if len(mlk_landmark_copy[landmark]) != 0: 
                                mlk_landmark_copy[landmark] = mlk_landmark_copy[landmark][:-187]  
                            else:
                                mlk_landmark_copy[landmark] = mlk_landmark_copy[landmark]  
                        
                        if ang_vel_dict_copy != {}: 
                            for joint in ang_vel_dict_copy:
                                if len(ang_vel_dict_copy[joint]) != 0: 
                                    ang_vel_dict_copy[joint] = ang_vel_dict_copy[joint][:-187]  
                                else:
                                    ang_vel_dict_copy[joint] = ang_vel_dict_copy[joint]  

                    if external_vid: 
                        summary_metrics_dict = mm.get_metrics(dict_angles_copy, ang_vel_dict_copy, mlk_segment_copy, mlk_landmark_copy, key, bucket, cv_mode, final_frame_time)
                    else: 
                        summary_metrics_dict = mm.get_metrics(dict_angles_copy, ang_vel_dict_copy, mlk_segment_copy, mlk_landmark_copy, key, bucket, cv_mode, 0.016)

            else: 
                if "calibrationQuats" in data_cp: 
                    if sensor_mode != "": 
                        if "calMode" in file: 
                            cal_mode = file["calMode"]
                        else: 
                            cal_mode = 2

                        list_of_all_angles = am.update_angles(data_cp, sensor_mode, cal_mode)

                        org_smooth_data['unityJointAnglesDict'] = list_of_all_angles

                        joint_df = pd.DataFrame.from_dict(org_smooth_data["unityJointAnglesDict"])

                        for column in joint_df:
                            if column != "ival" and column != "frame": 
                                joint_df[column] = joint_df[column].ewm(span=10).mean()

                        new_dict = joint_df.to_dict(orient="list")
                        org_smooth_data["unityJointAnglesDict"] = new_dict

                        ## Summary Metrics for Report pipeline 
                        if wkt_id != "freesession" and wkt_id != "": 
                            dict_angles_s_copy = copy.deepcopy(new_dict)
                            ang_vel_dict_copy = {}
                            try: 
                                org_data_cp = copy.deepcopy(org_smooth_data)
                                joint_kinematics_dict_cp = jointKinematics(org_data_cp) 
                                for joint in joint_kinematics_dict_cp: 
                                    ang_vel_dict_copy[joint] = joint_kinematics_dict_cp[joint]["Vel"]
                            except: 
                                ang_vel_dict_copy = {}

                            summary_metrics_dict = mm.get_metrics_sensor(dict_angles_s_copy, ang_vel_dict_copy, key, bucket)

        # Sensor data resampled + smoothed & raw sample rates
        transformed_data['fullSet'] = org_smooth_data
        transformed_data['Sample Rates'] = sample_dict

        # Extract joint angles, velocities, and accelerations
        try: 
            joint_kinematics_dict = jointKinematics(org_smooth_data) 
        except: 
            joint_kinematics_dict = {}
       
        # Rep Data
        rep_data_dict = {}

        try: 
            num_reps, rep_windows = totalAccelerationTrialCounter(org_smooth_data)
            rep_data_dict["NumReps"] = num_reps

            rep_count = 0
            for i in range(len(rep_windows)):  
                rep_str = "Rep" + str(rep_count+1)
                rep_dict = {}
                
                # Saving window 
                rep_dict["Window"] = rep_windows[rep_count]
                rep_data_dict[rep_str] = rep_dict

                rep_count += 1 
        except: 
            rep_data_dict = {}

        # Report Data
        assessment_data_dict = {}
        scoring_dict = {}
        if wkt_id != "freesession" and wkt_id != "": 
            ## Linear report metrics pipeline (JUST FOR JUMPS)
            if athlete_height_in != 0:
                if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 
                    if exc_details == "2A6F3D6D-54B9-44CF-969B-DAA562529F2D" or exc_details == "9C12B60A-671E-4560-B4FA-3790D29EE447": 
                        try: 
                            jump_height = linearKinematics(org_smooth_data, athlete_height_in)
                            print("Jump Height")
                            print(jump_height)
                            assessment_data_dict["JumpHeight"] = {"Value1": {"Value2": {"Value3": {"Value4": jump_height}}}}
                        except: 
                            assessment_data_dict["JumpHeight"] = {}
                            
            try:
                joint_kin_copy = copy.deepcopy(joint_kinematics_dict)

                ## Removing ~ last 3 seconds prior to repot overview score calcs
                if cv_mode == 1 or cv_mode == 2 or cv_mode == 3: 
                    if trim3: 
                        for joint in joint_kin_copy: 
                            if len(joint_kin_copy[joint]["Ang"]) != 0: 
                                joint_kin_copy[joint]["Ang"] = joint_kin_copy[joint]["Ang"][:-187]  
                            else:
                                joint_kin_copy[joint]["Ang"] = joint_kin_copy[joint]["Ang"]
                            
                            if len(joint_kin_copy[joint]["Vel"]) != 0: 
                                joint_kin_copy[joint]["Vel"] = joint_kin_copy[joint]["Vel"][:-187]  
                            else:
                                joint_kin_copy[joint]["Vel"] = joint_kin_copy[joint]["Vel"]

                            if len(joint_kin_copy[joint]["Acc"]) != 0: 
                                joint_kin_copy[joint]["Acc"] = joint_kin_copy[joint]["Acc"][:-187]  
                            else:
                                joint_kin_copy[joint]["Acc"] = joint_kin_copy[joint]["Acc"]

                if exc_details == "296C21D6-431C-4D09-9484-60575CE88A9D":
                    print("Shoulder IR/ER specific metrics")
                    assess_data = assessmentMetrics(selected_joints, orientation, joint_kin_copy, cv_mode, detailsID = exc_details)
                else: 
                    assess_data = assessmentMetrics(selected_joints, orientation, joint_kin_copy, cv_mode)
                assessment_data_dict["WorkoutMetrics"] = assess_data #workoutMetrics (new folder) 
                ## GPT Scoring code
                try: 
                    ## Generating score dictionaries for GPT output
                    data_dict = copy.deepcopy(assessment_data_dict)
                    data_dict = data_dict["WorkoutMetrics"]["Ang"]
                    range_dict, ranking_dict = js.fetch_from_backend(exc_id, bucket, key, org_id)
                    data_dict = js.clean_data_dict(data_dict)
                    data_dict_max, joint_name_conversions = js.get_max_joint_angles(data_dict)
                    score_output_dict = js.score_joint_angles(data_dict_max, range_dict)
                    top_suboptimal_dict = js.get_top_suboptimal_joints(score_output_dict, ranking_dict)
                    paired_joint_dict = js.pair_joints(data_dict_max)
                    assymetry_output_dict = js.score_assymetry(paired_joint_dict)

                    ## Adding score to the assessment_data_dict (1 - Optimal/"O", 0 - SubOptimal/"S")
                    for joint in assessment_data_dict["WorkoutMetrics"]["Ang"]: 
                        for side in assessment_data_dict["WorkoutMetrics"]["Ang"][joint]: 
                            try: 
                                if score_output_dict[f"{side.lower()}_{joint_name_conversions[joint]}"]["label"] == "O":
                                    assessment_data_dict["WorkoutMetrics"]["Ang"][joint][side]["Score"] = 1
                                elif score_output_dict[f"{side.lower()}_{joint_name_conversions[joint]}"]["label"] == "S":
                                    assessment_data_dict["WorkoutMetrics"]["Ang"][joint][side]["Score"] = 0
                                elif score_output_dict[f"{side.lower()}_{joint_name_conversions[joint]}"]["label"] == "M": 
                                    assessment_data_dict["WorkoutMetrics"]["Ang"][joint][side]["Score"] = 0.5
                            except: 
                                continue
                    
                    scoring_dict["JointScores"] = score_output_dict
                    scoring_dict["topJointScores"] = top_suboptimal_dict
                    scoring_dict["AsymmetryScores"] = assymetry_output_dict

                # print("Score dict")
                # print(score_output_dict)
                # print("Top dict")
                # print(top_suboptimal_dict)
                # print("Sym dict")
                # print(assymetry_output_dict)

                except:
                    scoring_dict = {}

            except: 
                assessment_data_dict["WorkoutMetrics"] = {}


        return transformed_data, joint_kinematics_dict, rep_data_dict, assessment_data_dict, scoring_dict, summary_metrics_dict


## ACTUALLY RUNNING THE FUNCTION 
    # test_event = {
    # "Records": [
    #     {
    #     "eventVersion": "2.0",
    #     "eventSource": "aws:s3",
    #     "awsRegion": "us-east-1",
    #     "eventTime": "1970-01-01T00:00:00.000Z",
    #     "eventName": "ObjectCreated:Put",
    #     "userIdentity": {
    #         "principalId": "EXAMPLE"
    #     },
    #     "requestParameters": {
    #         "sourceIPAddress": "127.0.0.1"
    #     },
    #     "responseElements": {
    #         "x-amz-request-id": "EXAMPLE123456789",
    #         "x-amz-id-2": "EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH"
    #     },
    #     "s3": {
    #         "s3SchemaVersion": "1.0",
    #         "configurationId": "testConfigRule",
    #         "bucket": {
    #         "name": "bravebucket210651-dev",
    #         "ownerIdentity": {
    #             "principalId": "EXAMPLE"
    #         },
    #         "arn": "arn:aws:s3:::bravebucket210651-dev"
    #         },
    #         "object": {
    #         "key": "public/8f007cc1-0999-4109-b305-670a39392c30:6D6128A7-3038-4777-98F5-7CE587DDB9E9",
    #         "size": 534.4,
    #         "eTag": "a5e6bfb85652d096ea34a9c3b2016130",
    #         "sequencer": "0A1B2C3D4E5F678901"
    #         }
    #     }
    #     }
    # ]
    # }

    handler(sys.argv[1], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    # handler(json.dumps(test_event), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
