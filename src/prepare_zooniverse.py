import json
import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval

def prepare(data_path, classification_file, out_path):
	df = pd.read_csv(Path(data_path, classifications_path))
	df = df[['classification_id', 'annotations', 'subject_data']]
	
	# Extract the video filename and annotation details
	df['annotation'] = df.apply(lambda x: ([v['filename'] for k,v in json.loads(x.subject_data).items()], \
                                literal_eval(x['annotations'])[0]['value']) \
                                if len(literal_eval(x['annotations'])[0]['value']) > 0 else None, 1)
	
	# Convert annotation to format which the tracker expects
	ds = [{"filename": i[0][0].split('_frame', 1)[0],
           "annotations": [{'class_name': item['tool_label'], 
                            'start_frame': i[0][0].split('_frame', 1)[1].replace('.jpg', ''),
                            'x': item['x'], 'y': item['y'],
                            'w': item['width'], 'h': item['height']} 
                            for item in i[1]]} for i in df.annotation if i is not None]
                            
    pd.DataFrame.from_dict(ds)[['filename', 'annotations']].to_csv(out_path)


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data folder", type=str)
    parser.add_argument("classification_file", help="output from workflow 2 in Zooniverse", type=int)
    parser.add_argument("out_path", help='path to save into text files', type=str)
    args = parser.parse_args()

    prepare(args.data_path, args.classification_file, args.out_path)

if __name__ == "__main__":
    main()