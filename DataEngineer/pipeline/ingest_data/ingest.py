import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
#Example Data#
    #176_สะพานลอยหน้าสวนจตุจักร-ถนนพ_รถมาจากแยกลาดพร้าว-มุ่งหน้า_20260830_104618#

base_dir = Path('raw_data/cctv_images')
valid_ext = {'.jpg', '.jpeg', '.png'}

store = []

for file_path in base_dir.rglob('*') :
    if not (file_path.is_file() and file_path.suffix.lower() in valid_ext):
        continue

    spilt_variable = file_path.name.split('_')
    cam_id = spilt_variable[0]
    filename = str(file_path.name)
    path = str(file_path)
    datestamp = datetime.strptime(f"{spilt_variable[3]}_{spilt_variable[4].split('.')[0]}",'%Y%m%d_%H%M%S')
    
    
    store.append({
        "camera_id" : cam_id,
        "Date_time" : datestamp,
        "filename" : filename,
        "path" : path

        })



df = pd.DataFrame(store)

df['Hour'] = df['Date_time'].dt.hour
df['minute'] = df['Date_time'].dt.minute
df['dayofweek']= df['Date_time'].dt.dayofweek
df['date'] = df['Date_time'].dt.date

print(df.head())
print(df['Date_time'])


#download
#df.to_parquet('ingest_data.parquet', index=False)


## add resolution[width , height] , file_size , 