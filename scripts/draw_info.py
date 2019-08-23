import glob
import json
import csv

data_path = "/Neutron6/ranjith.reddy/traffic_signs/pytorch-yolo-v3/Captures/*"

exp_path = "/Neutron6/ranjith.reddy/Road-Infrastructure/experiments/*"

def extract_info(capture_info,capture_name):
    distance = capture_info['distance']['dist']
    topics = capture_info['topics']
    for topic in topics:
        if topic['topic']=='/fix':
            gps_messages = topic['messages']
            gps_frequency = topic['frequency']
        elif topic['topic']=='/image_raw/left':
            cam_messages = topic['messages']
            cam_frequency = topic['frequency']
    return [capture_name,distance,gps_messages,gps_frequency,cam_messages,cam_frequency]


videos = glob.glob(data_path)
with open('Capture_details.csv','w') as csv_fp:
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow(["Name","distance","gps_messages","gps_frequency","cam_messages","cam_frequency"])
    for video in videos:
        capture_name = video.split("/")[-1]
        video_data = glob.glob(video+"/*")
        info_path = video+"/info.json"
        with open(info_path) as fp:
            capture_info = json.load(fp)
            data = extract_info(capture_info,capture_name)
        csv_writer.writerow(data)