import numpy as np
import cv2
from objloader_simple import *

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    print(imgpts)
    # draw ground floor in green

    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (20,0,0), 3)
    for i in range(4):
        temp = np.array([tuple(imgpts[i%8]),tuple(imgpts[(i+1)%8]),tuple(imgpts[(i+5)%8]),tuple(imgpts[(i+4)%8])])
        img = cv2.drawContours(img, [temp], -1, (255,0, 0), -3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), -3)
    return img

def draw_faces(img, faces,colors):
    for face, color in zip(faces, colors):
        imgpts = np.int32(face).reshape(-1, 2)
        img = cv2.drawContours(img, [imgpts], -1, color, -3)
    return img

def draw_object(img,faces, vertices,rotation_vector, translation_vector, camera_matrix,scale = 1,color=(155,155,155)):
    scale_matrix = np.eye(3) * scale
    scale_matrix[-1][-1] *= - 1

    for face in faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points]) # Translation to the center of image

        imgpts, _ = cv2.projectPoints(points, rotation_vector, translation_vector, camera_matrix, None)
        cv2.fillConvexPoly(img, np.int32(imgpts), color,1)

    return img

def draw_lines(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    print(imgpts)
    for i in range(4):
        img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[i]), (255), 3)
    return img


#3D Object model parameters (faces, vertices and scale)
obj = OBJ('deer.obj', swapyz=True)
faces = obj.faces
vertices = obj.vertices
scale = 1
dist_coeffs = np.zeros((4, 1)) #Disstortion coefficients


#Model Parameters
model = cv2.imread('teste2.png',0)
h, w = model.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
pts3d = np.float32([[0, 0,0], [0, h - 1,0], [w - 1, h - 1,0], [w - 1, 0,0]]).reshape(-1, 1, 3)

# Initiate ORB detector
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints with ORB
kp_model, des_model = orb.detectAndCompute(model, None)
bf = cv2.BFMatcher()

#  Loads the camera
cam = cv2.VideoCapture(0)

while True:
    ret_val, frame = cam.read()
    # create brute force  matcher object

    # compute the descriptors with ORB
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    # Compute scene keypoints and its descriptors
    matches = bf.knnMatch(des_model, des_frame, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    img_2 = frame
    #Para tirar possíveis falsos positivos
    if len(good)<21:
        good = []
    else:
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()



        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        if M is not None:
            dst = cv2.perspectiveTransform(pts, M)
            try:
                dst_temp = ([0.5*(a + b) for a,b in zip(dst,dst_prev)])
                dst_temp = np.float32([[list(a[0])] for a in dst_temp]).reshape(-1,1,2)
                dst_prev = dst
                dst = dst_temp
            except:
                dst_prev = dst

            (_, rotation_vector, translation_vector) = cv2.solvePnP(pts3d, dst, camera_matrix,
                                                                dist_coeffs)
            img_2 = draw_object(frame,faces, vertices,rotation_vector, translation_vector, camera_matrix,scale = scale, color = (100,120,200))

    #img_2 = cv2.drawMatches(model, kp_model, frame, kp_frame, good, frame, flags=2)
    cv2.imshow('frame', img_2)
    kp_frame_prev = kp_frame
    des_frame_prev = des_frame
    goods_prev = good
    M_prev = np.eye(3)

    keyboard = cv2.waitKey(1)

    if keyboard == 27:
        break  # esc to quit
    elif keyboard == 97:
        scale = scale + 0.1
    elif keyboard == 115:
        if scale > 0.1:
            scale = scale - 0.1
#Destroy the windows from openCV
cv2.waitKey(0)
cv2.destroyAllWindows()