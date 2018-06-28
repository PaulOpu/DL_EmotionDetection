from PIL import Image
import face_recognition
import os


im = "images/testing/obama.jpg"


def  recognize_faces(im, image_rescale=(48, 48), save_loc=None, new_ext=None):

    filename, ext = os.path.splitext(im)
    if new_ext is not None:
        ext = new_ext

    filename = os.path.basename(filename)

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(im)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_g = pil_image.convert("LA")
        pil_g = pil_g.resize(image_rescale, Image.ANTIALIAS)

        if save_loc is None:
            pil_g.show()
        else:
            pil_g.save(save_loc + os.sep + filename + ext)



def recognize_from_folder(folder, save_loc=None):
    for im in os.listdir(folder):
        recognize_faces(os.path.join(folder, im), save_loc)


if __name__ == "__main__":
    folder = "images/testing/"
    #recognize_faces(im)
    recognize_from_folder(folder)
