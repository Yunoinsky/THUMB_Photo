import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# Load the jpg files into numpy arrays

Fu_Ruolin_image = face_recognition.load_image_file("Image_V0.1/Fu_Ruolin/Fu_Ruolin1.jpg")
Wang_Yuling_image = face_recognition.load_image_file("Image_V0.1/Wang_Yuling/Wang_Yuling2.jpg")
Li_Jinpu_image = face_recognition.load_image_file("Image_V0.1/Li_Jinpu/Li_Jinpu1.jpg")
Yu_Zihong_image = face_recognition.load_image_file("Image_V0.1/Yu_Zihong/Yu_Zihong1.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    Fu_Ruolin_face_encoding = face_recognition.face_encodings(Fu_Ruolin_image)[0]
    Wang_Yuling_face_encoding = face_recognition.face_encodings(Wang_Yuling_image)[0]
    Li_Jinpu_face_encoding = face_recognition.face_encodings(Li_Jinpu_image)[0]
    Yu_Zihong_face_encoding = face_recognition.face_encodings(Yu_Zihong_image)[0]

except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_face_encodings = [
    Fu_Ruolin_face_encoding,Wang_Yuling_face_encoding,Li_Jinpu_face_encoding,Yu_Zihong_face_encoding
]

known_face_names = [
    "Fu_Ruolin",
    "Wang_Yuling","Li_Jinpu","Yu_Zihong"
]
unknown_image_list = []
Wang_Yuling_test_image = face_recognition.load_image_file("Test_V0.1/Wang_Yuling_Test1.JPG")
unknown_image_list.append(Wang_Yuling_test_image)
Li_Jinpu_test_image = face_recognition.load_image_file("Test_V0.1/Li_Jinpu_Test1.JPG")
unknown_image_list.append(Li_Jinpu_test_image)
Fu_Ruolin_test_image = face_recognition.load_image_file("Test_V0.1/Fu_Ruolin_Test1.JPG")
unknown_image_list.append(Fu_Ruolin_test_image)
Yu_Zihong_test_image = face_recognition.load_image_file("Test_V0.1/Yu_Zihong_Test1.JPG")
unknown_image_list.append(Yu_Zihong_test_image)

Wang_Yuling_test_face_encoding = face_recognition.face_encodings(Wang_Yuling_test_image)[0]
Fu_Ruolin_test_face_encoding = face_recognition.face_encodings(Fu_Ruolin_test_image)[0]
Li_Jinpu_test_face_encoding = face_recognition.face_encodings(Li_Jinpu_test_image)[0]
Yu_Zihong_test_face_encoding = face_recognition.face_encodings(Yu_Zihong_test_image)[0]

index = 0

for unknown_image in unknown_image_list:
    index += 1

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    path = ("image_with_boxes{}.jpg".format(index))
    pil_image.save(path)
