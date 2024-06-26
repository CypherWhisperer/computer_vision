import cv2 as cv
from deepface import DeepFace as dp
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--output', default='output.jpg',help='path to save the output image/video')
args = parser.parse_args()

# A function to draw a rectangle on the face(s) detected 
def face(net, frame, confidence_threshold=0.7):
    """
    highlights the image section where face is detected and draws a rectangle

    Args:
        net (str): Neural Network to detect face (config file and pre-trained weights)
        frame (str): frame for face detection
        confidence_threshold (float, optional): confidence threshold. Defaults to 0.7.

    Returns:
        _type_: frame with the rectangle drawn
        list: a list of coordinates for face(s) detected, as lists
    """
    frame_copy = frame.copy()
    #get the dimensions of the copy of the frame
    frame_height = frame_copy.shape[0]
    frame_width = frame_copy.shape[1]
    
    #converting the image to a blob(Binary Large OBject)(a 4D array(numimages,numchannels,height,width)) to later pass to the neural network:
    #the BLOB represents the image in a format thst is suitable for processing by the Neural Network 
    #takes in frame,scale factor, result frame size, normalization values,swqp_red_blue_chanels ?, crop_image_if_needed ?)
    
    blob = cv.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob) #passing blob as input to the NN
    faces = net.forward()#forward propagation, to process blob via NN
    face_boxes = [] #empty list for coordinates of boxes drawn on face(s)
    
    for i in range(faces.shape[2]): #range of number of detected faces
        confidence_level = faces[0, 0, i, 2]
        if confidence_level > confidence_threshold:
            x1 = int(faces[0, 0, i, 3] * frame_width) #x coordinate of top_left corner * width
            y1 = int(faces[0, 0, i, 4] * frame_height)
            x2 = int(faces[0, 0, i, 5] * frame_width) #bottom right corner coordinates * width
            y2 = int(faces[0, 0, i, 6] * frame_height)
            
            face_boxes.append([x1, y1, x2, y2]) #append a list  of coordinates for each face deected
            cv.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), int(round(frame_height / 150)), 8)
            
     #return resultant frame with box drawn and coordinates for boxes, for each face detected       
    return frame_copy, face_boxes

# MODEL FILES AND LABELS
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

#Proto files are txt files that store the configurations of the neural network, defining its architecture
#Model files are binary files that hold the pre-trained weights for the neural network

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#normalization values for color channels.these are to be subtracted from the values in input frame.
#the result values are in a consistent range and distribution, with which the models were trained, hence speeding convergence

agelist = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# LOADING THE NETWORKS
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

# Function for emotion prediction using DeepFace
def predict_emotion(image, face_box, padding=20):
    """
    Predicts emotions of fce(s) detected in the input image

    Args:
        image_path (str): _path to the image/video file
        face_boxes (list): a list of bounding boxes for detected faces; each box is a tuple (x1,y1,x2,y2)
        padding (int, optional): padding around boxes. Defaults to 20.
    Returns:
        List: a list of dictioneries with emotion probabilities for each face
    """
    try:
        x1, y1, x2, y2 = face_box
        face = image[max(0, y1-padding):min(y2+padding, image.shape[0]-1),
                     max(0, x1-padding):min(x2+padding, image.shape[1]-1)]

        analysis = dp.analyze(img_path=face, actions=['emotion'])
        emotion_probabilities = analysis[0]['emotion']
        
        dominant_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
        
        return emotion_probabilities, dominant_emotion
    except Exception as e:
        print(f"Error: {e}")
        return None, None

video = cv.VideoCapture(args.image if args.image else 0)
padding = 20

#define the codec and create VideoWriter object
if not args.image:
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(args.output, fourcc, 20.0,(int(video.get(3)), int(video.get(4))))

while not (cv.waitKey(20) & 0xFF == ord('q')):#cv.waitKey(1) < 0:
    has_frame, frame = video.read()
    if not has_frame:
        cv.waitKey(0)
        break

    result_img, face_boxes = face(faceNet, frame)
    if not face_boxes:
        print("Error: NO faces detected !")
        continue

    for face_box in face_boxes:
        face_crop = frame[max(0, face_box[1] - padding): min(face_box[3] + padding, frame.shape[0] - 1),
                          max(0, face_box[0] - padding): min(face_box[2] + padding, frame.shape[1] - 1)]
        
        blob = cv.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        gender_predictions = genderNet.forward()
        gender = genderList[gender_predictions[0].argmax()]
        print(f"Gender: {gender}")

        ageNet.setInput(blob)
        age_predictions = ageNet.forward()
        age = agelist[age_predictions[0].argmax()]
        print(f"Age: {age}")

        emotion_probabilities, dominant_emotion = predict_emotion(frame, face_box)
        if emotion_probabilities and dominant_emotion:
            print(f"Emotion: {dominant_emotion}")

            text = f'{gender}\n, {dominant_emotion}\n, {age}'
            y_offset = max(20, face_box[1] - 20)  # Ensure text is within image boundaries
            cv.putText(result_img, text, (face_box[0], y_offset),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow("FACE PREDICTIONS ..", result_img)
    
    if not args.image:
        out.write(result_img)

# Release resources
video.release()
if not args.image:
    out.release()
cv.destroyAllWindows()

# Save the final frame if image mode
if args.image:
    cv.imwrite(args.output, result_img)

