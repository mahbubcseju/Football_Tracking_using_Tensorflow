import sys
from pathlib import Path
from football_tracking import detect_object_on_video
import os,fnmatch

if(len(sys.argv)==2):
    action = sys.argv[1]
    test_image_path = []
    my_file = Path(action)
    if my_file.is_file():
        test_image_path.append(action)
        #print(test_image_path)
        detect_object_on_video(str(action))
    else:
        print("Invalid output")
else:
    print("Send test image directory or image path")