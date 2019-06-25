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
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        imgpts, _ = cv2.projectPoints(points, rotation_vector, translation_vector, camera_matrix, None)
        cv2.fillConvexPoly(img, np.int32(imgpts), color,1)

    return img

def draw_lines(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    print(imgpts)
    for i in range(4):
        img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[i]), (255), 3)
    return img



obj = OBJ('cow.obj', swapyz=True)
faces = obj.faces
vertices = obj.vertices
scale = 1

model = cv2.imread('teste2.png',0)


# Initiate ORB detector
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints with ORB
kp_model, des_model = orb.detectAndCompute(model, None)
bf = cv2.BFMatcher()

# compute the descriptors with ORB
cam = cv2.VideoCapture(0)
while True:
    ret_val, frame = cam.read()
    # create brute force  matcher object

    # Compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    matches = bf.knnMatch(des_model, des_frame, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    img_2 = frame
    #Para tirar possíveis falsos positivos
    if len(good)<=22:
        good = []
    else:
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        pts3d = np.float32([[0, 0,0], [0, h - 1,0], [w - 1, h - 1,0], [w - 1, 0,0]]).reshape(-1, 1, 3)

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

            dist_coeffs = np.zeros((4, 1)) #Distortion coefficients
            (_, rotation_vector, translation_vector) = cv2.solvePnP(pts3d, dst, camera_matrix,
                                                                    dist_coeffs)

            #for i in range(num_objs):
            #    for j in range(faces_por_obj):
            #        lista_colors.append(colors[i])
            (_, rotation_vector, translation_vector) = cv2.solvePnP(pts3d, dst, camera_matrix,
                                                                    dist_coeffs)
            img_2 = draw_object(frame,faces, vertices,rotation_vector, translation_vector, camera_matrix,scale = scale)


    cv2.imshow('frame', img_2)

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