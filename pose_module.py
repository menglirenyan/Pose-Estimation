import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    #initializing 
    def __init__(self, mode= False, upBody=False, smooth=True,detectionCon =0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        #self.pose = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)
    
    def findPose(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                
        #if pose landmarks are detected then join those pose landmarks with draw_landmarks call        
        return img #return the image with drawn pose landmarks and joined lines

    def findPosition(self,img, draw=True):
        self.lmList = [] #landmark list - making it part of the object itse;f
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id,lm)
                cx , cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return self.lmList

    #creating a new method to find the angle between 3 landmark points given their x and y coordinates
    def findAngle(self, img, p1,p2,p3,draw =True):
        #here p1 , p2 and p3 [index values] are respective 3 landmarks using which we need to find the angle
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]

        #calculating the angle between the landmarks
        angle =math.degrees( math.atan2(y3-y2,x3-x2)- math.atan2(y1-y2,x1-x2))
        #handling the case for negative angles and getting the proper range (0,360)
        if angle<0:
            angle+= 360

        if angle > 180:
            angle = 360 - angle

        #print(angle)

        #drawing circles for the 3 points in the consideration 
        if draw:
                #making visible line between point 1 and 2 with white color
                cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
                #making visible line between point 2 and 3 with white color
                cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
                cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
                cv2.circle(img,(x1,y1),15,(0,0,255),2)
                cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
                cv2.circle(img,(x2,y2),15,(0,0,255),2)
                cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
                cv2.circle(img,(x3,y3),15,(0,0,255),2)
                cv2.putText(img, str(int(angle)), (x2-20, y2+30), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255),3) #adjust the values for the angle text but keep it near the point 2

        return angle



def main():
    path = '/Users/samik/Desktop/Programming/Pose Estimation/test_footage.mp4'
    cap = cv2.VideoCapture(path)
    pTime = 0
    detector= poseDetector() #object of poseDetector class made
    while True:
        #read the frames from the video capture object
        success,img = cap.read()
        img =  detector.findPose(img)
        lmList = detector.findPosition(img,draw= False)
        if len(lmList)!=0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0,0,255), cv2.FILLED)
        #to calculate FPS
        cTime = time.time()
        fps = 1 /(cTime- pTime)
        pTime = cTime
        #display fps somewhere on the screen window of the resultant image/video frame
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("Image",img)
        cv2.waitKey(1) #delay of 1s

if __name__ == "__main__":
    main()