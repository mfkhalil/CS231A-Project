import numpy as np

from landmarks import *

# Angle margin of error, change as needed
MARGIN_OF_ERROR = 8

JOINTS = {
    # Only joints relevant to dumbbell row were added, feel free to add more as needed
    'right elbow': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    'left elbow': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    'right shoulder': (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    'left shoulder': (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP)
}


def vector(a, b):
    return np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])


def angle_between_lines(a, b, c):
    vector_ab = vector(a, b)
    vector_bc = vector(b, c)
    cos_angle = vector_ab.dot(vector_bc) / (np.linalg.norm(vector_ab) * np.linalg.norm(vector_bc))
    angle = math.degrees(math.acos(cos_angle))
    return angle


def landmark_as_array(output, landmark):
    x = output.pose_world_landmarks.landmark[landmark].x
    y = output.pose_world_landmarks.landmark[landmark].y
    z = output.pose_world_landmarks.landmark[landmark].z
    return np.array([x, y, z])


# Start Position for Baseline
bl_start = get_landmarks('bl_start.png')
# End Position for Baseline
bl_end = get_landmarks('bl_end.png')

# Start Position for Test
t_start = get_landmarks('t_start.png')
# End Position for Test
t_end = get_landmarks('t_end.png')


def check_angle(jointname):
    joint = JOINTS[jointname]

    right_start_one = landmark_as_array(bl_start, joint[0])
    right_start_two = landmark_as_array(bl_start, joint[1])
    right_start_three = landmark_as_array(bl_start, joint[2])

    right_end_one = landmark_as_array(bl_end, joint[0])
    right_end_two = landmark_as_array(bl_end, joint[1])
    right_end_three = landmark_as_array(bl_end, joint[2])

    bl_angle_start = angle_between_lines(right_start_three, right_start_two, right_start_one)
    print('Baseline ' + jointname + ' angle at start position is ' +
          str(bl_angle_start) + ' degrees.')

    bl_angle_end = angle_between_lines(right_end_three, right_end_two, right_end_one)
    print('Baseline ' + jointname + ' angle at end position is ' +
          str(bl_angle_end) + ' degrees.')

    right_start_one = landmark_as_array(t_start, joint[0])
    right_start_two = landmark_as_array(t_start, joint[1])
    right_start_three = landmark_as_array(t_start, joint[2])

    right_end_one = landmark_as_array(t_end, joint[0])
    right_end_two = landmark_as_array(t_end, joint[1])
    right_end_three = landmark_as_array(t_end, joint[2])

    t_angle_start = angle_between_lines(right_start_three, right_start_two, right_start_one)
    print('Test ' + jointname + ' angle at start position is ' +
          str(t_angle_start) + ' degrees.')

    t_angle_end = angle_between_lines(right_end_three, right_end_two, right_end_one)
    print('Test ' + jointname + ' angle at end position is ' +
          str(t_angle_end) + ' degrees.')

    if abs(t_angle_start - bl_angle_start) > MARGIN_OF_ERROR:
        print('Your ' + jointname + ' start position was off. Please adjust by ' +
              str(abs(t_angle_start - bl_angle_start) - MARGIN_OF_ERROR) + ' degrees!')
    if abs(t_angle_end - bl_angle_end) > MARGIN_OF_ERROR:
        print('Your ' + jointname + ' end position was off. Please adjust by ' +
              str(abs(t_angle_start - bl_angle_start) - MARGIN_OF_ERROR) + ' degrees!')
    print('\n')
    return


check_angle('right elbow')
check_angle('left elbow')
check_angle('right shoulder')
check_angle('left shoulder')
