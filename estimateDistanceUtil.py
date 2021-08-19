import cv2

# distance from camera to object measured(meter)
KNOWN_PRESON_DISTANCE = 0.762
KNOWN_BUS_DISTANCE = 3.33
KNOWN_CAR_DISTANCE = 1.18
KNOWN_MOTORCYCLE_DISTANCE = 1.67
# width of object in the real world or object plane(meter)
KNOWN_PERSON_WIDTH = 0.6
KNOWN_BUS_WIDTH = 2.63
KNOWN_CAR_WIDTH = 1.86
KNOWN_MOTORCYCLE_WIDTH = 0.56
# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX

# focal length finder function
def focal_length(width_in_rf_image, measured_distance, real_width):
    focal_length_value = (width_in_rf_image*measured_distance)/real_width
    # return focal_length_value
    return focal_length_value

# distance estimation function
def distance_finder(focal_length, object_width_in_frame, real_object_width):
    distance = (real_object_width*focal_length)/object_width_in_frame
    return distance





