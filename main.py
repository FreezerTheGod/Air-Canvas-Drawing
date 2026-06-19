import cv2
import mediapipe as mp
import time
import numpy as np

# We create a container to hold the asset file path (hand_landmarker.task)
BaseOptions = mp.tasks.BaseOptions
# We create a container to hold the actual machine learning model
HandLandmarker = mp.tasks.vision.HandLandmarker
# We also configure the blueprint specifically for tracking hands
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# We use this to specify the running mode of the hand landmarker (IMAGE, VIDEO, LIVE_STREAM)
VisionRunningMode = mp.tasks.vision.RunningMode

# We set a collection of helpers to draw the hand landmarks and connections on the image
mp_draw = mp.tasks.vision.drawing_utils
# We use a hardcoded map of how the human skeleton anatomy works.
mp_connections = mp.tasks.vision.HandLandmarksConnections

# We initialize the webcam feed and set the resolution to 1280x720
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# We define the main function that will run the application
def main():
    # We set canvas to None
    canvas = None
    # We initialize the previous x and y coordinates to 0
    xp, yp = 0, 0

    # We define a list of colors for the brush and eraser
    colors = [
        (0, 255, 0),    
        (0, 0, 255),    
        (255, 255, 255),
        (0, 255, 255),  
        (0, 0, 0)       
    ]
    # We set the initial brush color and thickness, as well as the eraser thickness
    brush_color = colors[0]
    brush_thickness = 10
    eraser_thickness = 50

    # We set mediapipe options
    options = HandLandmarkerOptions(
        # We specify the path to the hand_landmarker.task file
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        # We specify the running mode to be IMAGE, which means we will process one image at a time
        running_mode=VisionRunningMode.IMAGE,
        # We specify the number of hands to detect (1 in this case)
        num_hands=1,
        # We specify the minimum confidence for hand detection and presence
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )

    # We create a hand landmarker object using the options we defined above
    with HandLandmarker.create_from_options(options) as hands:
        while True:
            # We set the defualt attempt to 0
            attempt = 0
            # We read a frame from the webcam feed
            success, img = cap.read()

            # The saftey loop checks if success is false, it should wait for 0.2 seconds and try to read the frame again, it will do this up to 5 times before giving up and printing an error message 
            while not success and attempt < 5:
                time.sleep(0.2)
                success, img = cap.read()
                attempt += 1
            if not success:
                print("Failed to capture video.")
                break
             
            # We flip the image horizontally to create a mirror effect 
            img = cv2.flip(img, 1)
            # We get the height, width, and number of channels of the image
            h, w, _ = img.shape
            
            # We check if the canvas is None
            if canvas is None:
                # We create a black canvas of the same size as the image
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # We divided the width of the image into 5 equal parts to create 5 boxes for each color and the erasor
            box_w = w // 5
            # This goes through your list of 5 colors and extracts both their index sequence position number (i) and their actual BGR color tuple (col). 
            for i, col in enumerate(colors):
                # We draw a filled rectangle for each color box at the top of the image
                cv2.rectangle(img, (i * box_w, 0), ((i + 1) * box_w, 80), col, cv2.FILLED)
                # If the index is 4, we put the text "ERASER" on the rectangle to indicate that it is the eraser tool
                if i == 4:
                    cv2.putText(img, "ERASER", (i * box_w + 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # If the current brush color is the same as the color of the rectangle, we draw a gray border around it to indicate that it is selected
                if col == brush_color:
                    cv2.rectangle(img, (i * box_w, 0), ((i + 1) * box_w, 80), (128, 128, 128), 4)

            # We convert the image from BGR to RGB format, as mediapipe expects images in RGB format 
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # We create a mediapipe image object from the RGB image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # We use the hand landmarker to detect hands in the image and get the results
            results = hands.detect(mp_image)

            # We set the current thickness to be either the eraser thickness or the brush thickness depending on whether the brush color is black (eraser) or not   
            current_thickness = eraser_thickness if brush_color == (0, 0, 0) else brush_thickness

            # We check if any hand landmarks were detected in the image 
            if results.hand_landmarks:
                # We loop through each detected hand and draw the landmarks and connections on the image
                for hand_landmarks in results.hand_landmarks:
                    # We draw the hand landmarks and connections on the image using mediapipe's drawing utils
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_connections.HAND_CONNECTIONS)
                    # We get the coordinates of the index finger tip (landmark 8), index finger pip (landmark 6), middle finger tip (landmark 12), and middle finger pip (landmark 10) and convert them to pixel values based on the image width and height
                    x8, y8 = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                    x6, y6 = int(hand_landmarks[6].x * w), int(hand_landmarks[6].y * h)
                    x12, y12 = int(hand_landmarks[12].x * w), int(hand_landmarks[12].y * h)
                    x10, y10 = int(hand_landmarks[10].x * w), int(hand_landmarks[10].y * h)
                     
                    # We check if the index finger is up (y8 < y6) and if the middle finger is up (y12 < y10) 
                    index_up = y8 < y6
                    middle_up = y12 < y10
                    
                    # We check if both the index finger and middle finger are up, which indicates that the user is selecting a color or eraser
                    if index_up and middle_up:
                        # We reset the previous x and y coordinates to 0, as we are not drawing on the canvas in this mode
                        xp, yp = 0, 0
                        # We draw a white circle at the tip of the index finger to indicate that the user is in selection mode
                        cv2.circle(img, (x8, y8), 6, (255, 255, 255), cv2.FILLED)
                        # We check if the y-coordinate of the index finger tip is less than 80, which means it is in the area of the color boxes at the top of the image
                        if y8 < 80:
                            # We calculate the index of the selected color or eraser based on the x-coordinate of the index finger tip and the width of each color box
                            selected_idx = x8 // box_w
                            # We check if the selected index is within the range of the colors list, and if so, we update the brush color to the selected color or eraser
                            if selected_idx < len(colors):
                                brush_color = colors[selected_idx]
                    # We check if the index finger is up and the middle finger is not up, which indicates that the user is in drawing mode 
                    elif index_up and not middle_up:
                        # We draw a circle at the tip of the index finger to indicate that the user is in drawing mode
                        if brush_color == (0, 0, 0):
                            # We draw a white circle with a thickness of 2 to indicate that the user is using the eraser tool
                            cv2.circle(img, (x8, y8), current_thickness, (255, 255, 255), 2)
                        else:
                            # We draw a filled circle with the current brush color to indicate that the user is using the brush tool
                            cv2.circle(img, (x8, y8), current_thickness, brush_color, cv2.FILLED)
                        # We check if the previous x and y coordinates are both 0, which means this is the first point being drawn on the canvas
                        if xp == 0 and yp == 0:
                            # We set the previous x and y coordinates to the current index finger tip coordinates
                            xp, yp = x8, y8
                        # We draw a line on the canvas from the previous point to the current point using the current brush color and thickness
                        cv2.line(canvas, (xp, yp), (x8, y8), brush_color, current_thickness)    
                        # We update the previous x and y coordinates to the current index finger tip coordinates for the next iteration
                        xp, yp = x8, y8
                    # We check if the index finger is not up and the middle finger is up, which indicates that the user is in drawing mode with the middle finger
                    else:
                        xp, yp = 0, 0
            # We check if no hand landmarks were detected in the image, which means the user is not interacting with the application            
            else:
                xp, yp = 0, 0

            # We convert the canvas to grayscale and create an inverted binary mask of the canvas    
            img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            # We apply a binary threshold to the grayscale canvas to create an inverted mask, where the drawn areas are black and the background is white
            _, img_inv = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
            # We convert the inverted mask back to BGR format so that it can be used for bitwise operations with the original image
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            
            # We use bitwise operations to combine the original image, the inverted mask, and the canvas to create the final output image
            img = cv2.bitwise_and(img, img_inv)
            # We use bitwise OR to overlay the canvas on top of the original image, so that the drawn content appears on the screen
            img = cv2.bitwise_or(img, canvas)
            # We put text on the image to indicate the controls for clearing the canvas and quitting the application
            cv2.putText(img, "Press 'C' to Clear Canvas", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Press 'Q' to Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # We display the final output image in a window titled "Air Canvas Paintbrush" 
            cv2.imshow("Air Canvas Paintbrush", img)  
            
            # We wait for a key press for 1 millisecond and check if the pressed key is 'q' or 'c' to quit or clear the canvas respectively
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # We release the webcam feed and destroy all OpenCV windows when the application is closed
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()