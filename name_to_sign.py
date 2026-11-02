"""
Mode 2: Type a name and display its sign-language letters.
Place images in sign_images/A.png, B.png, ... Z.png
SPACE/ENTER: Next letter | ESC: Quit
"""

import cv2
import os

SIGN_FOLDER = "sign_images"
name = input("Enter a name: ").upper().strip()

if not name:
    print("Please enter a valid name.")
    exit()

index = 0
while index < len(name):
    character = name[index]
    if character == " ":
        print("SPACE")
        cv2.waitKey(0)
        index += 1
        continue

    image_path = os.path.join(SIGN_FOLDER, f"{character}.jpg")
    if not os.path.exists(image_path):
        print(f"No image found for letter: {character}")
        index += 1
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        index += 1
        continue

    image = cv2.resize(image, (600, 600))
    cv2.putText(image, f"Letter: {character}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(image, f"Name: {name}", (30, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Name to Sign Language", image)

    if (cv2.waitKey(0) & 0xFF) == 27:
        break
    index += 1

cv2.destroyAllWindows()
