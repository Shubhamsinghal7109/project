import streamlit as st
import cv2
import numpy as np
import tempfile
st.set_page_config(page_title="Face Detection System",page_icon="https://cdn-icons-png.flaticon.com/512/5985/5985970.png")
choice=st.sidebar.selectbox("My Menu",("HOME","IMAGE","VIDEO"))
detectface=cv2.CascadeClassifier('face.xml')
st.title("Face Detection System")
if(choice=="HOME"):
    st.header("WELCOME")
    st.image("https://cdn.dribbble.com/users/3384635/screenshots/7015064/face_detection.gif")
elif(choice=="IMAGE"):
    img=st.file_uploader("Upload Image")
    if img:
        bytes=img.getvalue()
        img=cv2.imdecode(np.frombuffer(bytes,np.uint8),cv2.IMREAD_COLOR)
        face=detectface.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in face:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
        st.image(img,channels='BGR')
elif(choice=="VIDEO"):
    vid=st.file_uploader("Upload Video")
    frame=st.empty()
    if vid:
        tfile=tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid.read())
        vid=cv2.VideoCapture(tfile.name)
        while(vid.isOpened()):
            flag,img=vid.read()
            face=detectface.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in face:
                img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
            frame.image(img,channels='BGR')

