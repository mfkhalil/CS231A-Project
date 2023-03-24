import numpy as np

from landmarks import *

# Angle margin of error, change as needed
MARGIN_OF_ERROR = 10

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
    if angle < 90: return angle
    return 180 - angle


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

    start_one = landmark_as_array(bl_start, joint[0])
    start_two = landmark_as_array(bl_start, joint[1])
    start_three = landmark_as_array(bl_start, joint[2])

    end_one = landmark_as_array(bl_end, joint[0])
    end_two = landmark_as_array(bl_end, joint[1])
    end_three = landmark_as_array(bl_end, joint[2])

    bl_angle_start = round(angle_between_lines(start_three, start_two, start_one), 2)
    bl_start_str = 'Baseline ' + jointname + ' angle at start position is ' + str(bl_angle_start) + ' degrees.'

    bl_angle_end = round(angle_between_lines(end_three, end_two, end_one), 2)
    bl_end_str = 'Baseline ' + jointname + ' angle at end position is ' + str(bl_angle_end) + ' degrees.'

    start_one = landmark_as_array(t_start, joint[0])
    start_two = landmark_as_array(t_start, joint[1])
    start_three = landmark_as_array(t_start, joint[2])

    end_one = landmark_as_array(t_end, joint[0])
    end_two = landmark_as_array(t_end, joint[1])
    end_three = landmark_as_array(t_end, joint[2])

    t_angle_start = round(angle_between_lines(start_three, start_two, start_one), 2)
    t_start_str = 'Input ' + jointname + ' angle at start position is ' + str(t_angle_start) + ' degrees.'

    t_angle_end = round(angle_between_lines(end_three, end_two, end_one), 2)
    t_end_str = 'Input ' + jointname + ' angle at end position is ' + str(t_angle_end) + ' degrees.'

    print(bl_start_str)
    print(t_start_str)
    if abs(t_angle_start - bl_angle_start) > MARGIN_OF_ERROR:
        print('Your ' + jointname + ' start position was off. Please adjust by at least ' +
              str(round(abs(t_angle_start - bl_angle_start) - MARGIN_OF_ERROR)) + ' degrees!')
    print('\n')

    print(bl_end_str)
    print(t_end_str)
    if abs(t_angle_end - bl_angle_end) > MARGIN_OF_ERROR:
        print('Your ' + jointname + ' end position was off. Please adjust by at least ' +
              str(round(abs(t_angle_end - bl_angle_end) - MARGIN_OF_ERROR)) + ' degrees!')
    print('\n')
    return


check_angle('right elbow')
check_angle('left elbow')
check_angle('right shoulder')
check_angle('left shoulder')
