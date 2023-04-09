import PySimpleGUI as sg

import os.path
import cv2
import sys
import numpy as np
import time
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision





file_list_column = [
    [
        sg.Button('Start', size=(7, 4), font='Helvetica 14', pad=(0, 20))
    ],
    [
        sg.Button('Shot', size=(7, 4), font='Helvetica 14', pad=(0, 20))
    ],
    [
        sg.Button('Stop', size=(7, 4), font='Helvetica 14', pad=(0, 20))
    ]
]
image_viewer_column = [
    # [sg.Text("Choose an image from the list on the left:")],
    [sg.Text(size=(75, 1), key="-TOUT-")],
    [sg.Image(filename='', key='image')],
]
file_list_column1 = [
    [sg.Button('Original', size=(7, 2), font='Helvetica 14', pad=(0, 18))],
    [sg.Button('Detect', size=(7, 2), font='Helvetica 14', pad=(0, 18))],
    [sg.Button('Info', size=(7, 2), font='Helvetica 14', pad=(0, 18))],
    [sg.Button('Explorer', size=(7, 2), font='Helvetica 14', pad=(0, 18))],
    [sg.Button('Shutdown', size=(7, 2), font='Helvetica 14', pad=(0, 18))]
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
        sg.VSeperator(),
        sg.Column(file_list_column1)
    ]
]

window = sg.Window("Image Viewer", layout, no_titlebar=False, location=(0, 0), size=(800, 480), keep_on_top=False)
# Visualization parameters
_ROW_SIZE = 30  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (255, 255, 255)  # red
_FONT_SIZE = 2
_FONT_THICKNESS = 2
_FPS_AVERAGE_FRAME_COUNT = 10



base_options = core.BaseOptions(
  file_name='model.tflite', use_coral=False, num_threads=4)

classification_options = processor.ClassificationOptions(
  max_results=1, score_threshold=0.5)
options = vision.ImageClassifierOptions(
  base_options=base_options, classification_options=classification_options)

classifier = vision.ImageClassifier.create_from_options(options)

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

org = False
dt = False
sv = False
class_name = ""
f_normal = "정상 감귤"
l_normal = "정상 감귤 잎"
f_canker = '''"감귤열매 궤양병"
- 방제Tip :
1. 과수원 주변에 바람을 막아줄수있는 방품림 조성(낙과방지)
2. 귤굴나방 방제를 통한 2차전염 방지
3. 비료는 질소과다 사용을 피해야함
4. 병든 잎이나 가지, 열매를 무조건 제거 및 소각'''
l_aphid ='''"감귤 잎 진딧물"
- 방제Tip :
1. 과수원 계획시 합리적 과수배치 및 필요가지를 제외하고 전정
2. 무당벌레,풀잠자리 등의 천적을 이용
3. 과일 수확후 봄 전까지 적정농약을 두차례 살포시 월동 진딧물 제거가능'''
l_redmite ='''"감귤 잎 응애"
- 방제 Tip :
1. 동절기 기계유제의 살포로 응애밀도를 줄인다.
2. 잎당 평균 밀도가 3~4마리가 되면 즉시 응애약 살포
3. 장마가 끝난후 발생이 많으므로 우수한 전문 약제 집중 살포'''



while True:
    event, values = window.read(timeout=1)
    
    if event == "Explorer" :
        command = "open /home/zx1517/Saveimg"
        command_ = "wmctrl -r 'Saveimg' -b add,above"
        os.system(command)
        time.sleep(1)
        os.system(command_)
        
    elif event == 'Start':
        org = False
        dt = True

    elif event == 'Stop':
        org = False
        dt = False
        sv = False
        window['image'].update(data='')

    elif event == 'Original' :
        org = True
        dt = False
        sv = False

    elif event == 'Detect' :
        org = False
        dt = True

    elif event =='Shutdown' :
        #os.system("shutdown now -h")
        break
    
    elif (dt == False) and (event =='Shot') :
        sg.popup_no_frame("No photos(Only detecting)",keep_on_top=True,auto_close=True,font=4)
 
    elif (event == 'Info') and (sv==False) :
        sg.popup_no_frame("Shot first!!",keep_on_top=True,auto_close=True,font=4)
    
    elif (event == 'Info') and (sv==True) :
        if class_name == "fruit_normal" :
            sg.popup_no_frame(f_normal,keep_on_top=True,auto_close=False,font=4)
        if class_name == "fruit_canker" :
            sg.popup_no_frame(f_canker,keep_on_top=True,auto_close=False,font=4)
        if class_name == "leaf_normal" :
            sg.popup_no_frame(l_normal,keep_on_top=True,auto_close=False,font=4)
        if class_name == "leaf_aphid" :
            sg.popup_no_frame(l_aphid,keep_on_top=True,auto_close=False,font=4)
        if class_name == "leaf_redmite" :
            sg.popup_no_frame(l_redmite,keep_on_top=True,auto_close=False,font=4)
            
            
    if org == True:
        
        ret, frame = cap.read()
        frame_ = cv2.resize(frame, (530, 400))
        imgbytes = cv2.imencode('.png', frame_)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)


    if dt == True :
        sv = False
        success, image = cap.read()
        image = cv2.resize(image, (530,400))

        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
  

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create TensorImage from the RGB image
        tensor_image = vision.TensorImage.create_from_array(rgb_image)
        # List classification results
        categories = classifier.classify(tensor_image)

        # Show classification results on the image
        for idx, category in enumerate(categories.classifications[0].categories):
            category_name = category.category_name
            score = round(category.score, 2)
            result_text = category_name + ' (' + str(score) + ')'
            text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        # Calculate the FPS
        if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
            end_time = time.time()
            fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = ' + str(int(fps))
        text_location = (_LEFT_MARGIN, _ROW_SIZE)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
        imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
        if event == 'Shot':
            img_ = np.frombuffer(imgbytes,dtype=np.uint8)
            img__ = cv2.imdecode(img_,cv2.IMREAD_COLOR)
            tm = time.strftime("%Y%m%d_%H%M%S_")
            cv2.imwrite('/home/zx1517/Saveimg/'+tm+category_name+'.jpg',img__)
            sg.popup_no_frame("Save Image!",keep_on_top=True,auto_close=True,font=4)
            class_name = category_name
            org = False
            dt =False
            sv = True
        
window.close()
