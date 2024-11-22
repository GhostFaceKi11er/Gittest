import numpy as np

def xyz_to_transMatrix(xyzrpy): #将xyzrpy转换成奇次变换矩阵
    x = xyzrpy[0]
    y = xyzrpy[1]
    z = xyzrpy[2]
    r = xyzrpy[3]
    p = xyzrpy[4]
    yaw = xyzrpy[5]

    trans_mat = np.eye(4)
    trans_mat[0:3, 3] = [x, y, z]
    mr = np.array([[1, 0, 0],
          [0, np.cos(r), -np.sin(r)],
          [0, np.sin(r),  np.cos(r)]])

    mp = np.array([[np.cos(p), 0, np.sin(p)],
          [0,      1,    0  ],
          [-np.sin(p),0, np.cos(p)]])

    my = np.array([[np.cos(yaw), -np.sin(yaw), 0],
          [np.sin(yaw),  np.cos(yaw), 0],
          [0,            0,     1]])

    rotation_matrix = my @ mp @ mr
    trans_mat[0:3, 0:3] = rotation_matrix
    return trans_mat


def FK(prev_matrix, curr_matrix): #计算正运动学
    matrix = np.dot(prev_matrix , curr_matrix)
    return matrix


'''def Damped_least_square(J, damping_ratio, err): #使用阻尼最小二乘法来计算每次迭代的更新值
    J1 = J.T @ J + damping_ratio ** 2 * np.eye(J.shape[1])
    JTJ_damped_inv = np.linalg.inv(J1)
    delta_theta = JTJ_damped_inv @ J.T @ err
    return delta_theta'''

def trans_Matrix_to_quaternion(matrix):
    matrix = matrix[:3, :3]
    epsilon = 1e-7
    tr = matrix.trace()
    if tr > 0:
        w = 0.5 * np.sqrt(1 + tr)
        x = (matrix[2, 1] - matrix[1, 2]) / (4 * w + epsilon)
        y = (matrix[0, 2] - matrix[2, 0]) / (4 * w + epsilon)
        z = (matrix[1, 0] - matrix[0, 1]) / (4 * w + epsilon)
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            x = 0.5 * np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / (4 * x + epsilon)
            y = (matrix[0, 1] + matrix[1, 0]) / (4 * x + epsilon)
            z = (matrix[0, 2] + matrix[2, 0]) / (4 * x + epsilon)
        elif matrix[1, 1] > matrix[2, 2]:
            y = 0.5 * np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / (4 * y + epsilon)
            x = (matrix[0, 1] + matrix[1, 0]) / (4 * y + epsilon)
            z = (matrix[1, 2] + matrix[2, 1]) / (4 * y + epsilon)
        else:
            z = 0.5 * np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / (4 * z + epsilon)
            x = (matrix[0, 2] + matrix[2, 0]) / (4 * z + epsilon)
            y = (matrix[1, 2] + matrix[2, 1]) / (4 * z + epsilon)
    q = np.array([x,y,z,w],dtype=np.float64)
    q = q / np.linalg.norm(q)

    return q

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1= q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([x, y, z, w])

def quaternion_to_rpy(q):
    w, x, y, z = q

    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    rpy = np.array([roll, pitch, yaw], dtype=np.float64)

    return rpy

def define_joints():
    a = np.random.uniform(-1.57, 1.57)
    b = np.random.uniform(0, 0.2)
    c = np.random.uniform(0.785, 1.785)
    d = np.random.uniform(-1, -0.2)
    e = np.random.uniform(0, 0.785)
    f = np.random.uniform(1, 2)
    theta = np.array([[a], [b], [c], [d], [e], [f]], dtype=np.float64)

    joint1 = np.array([0, 0, 0, 0, theta[0, 0], -3.14159265359],dtype=np.float64) # 0
    joint2 = np.array([0, 0, theta[1, 0], 0, 0, 0],dtype=np.float64) # 0.15185
    joint3 = np.array([0, 0, 0, theta[2, 0], 0, 0],dtype=np.float64) # 1.570796327
    joint4 = np.array([theta[3, 0], 0, 0, 0, 0, 0],dtype=np.float64) # -0.24355
    joint5 = np.array([-0.2132, 0, theta[4, 0], 0, 0, 0],dtype=np.float64) # 0.13105
    joint6 = np.array([0, -0.08535, -1.75055776238e-11, theta[5, 0], 0, 0],dtype=np.float64) # 1.570796327

    joints_matrixs = [joint1, joint2, joint3, joint4, joint5, joint6]
    joints_types = {1:['R',1], 2:['P', 2], 3:['R', 0], 4:['P', 0], 5:['P', 2], 6:['R', 0]} #设置好所有的joints以及标明它们是转动关节还是平移关节

    return theta, joints_matrixs, joints_types, joint1, joint2, joint3, joint4, joint5, joint6

def Jacobian_calculate(dof, joint_position_WorldFrame, joints_types):
    Jacobian_matrix = np.zeros((6,dof)) #计算雅可比矩阵
    for i in range(dof):
        if joints_types[i+1][0] == "R":
            col = joints_types[i+1][1]
            Jacobian_matrix[0:3, i] = np.cross(joint_position_WorldFrame[i][0:3, col], (joint_position_WorldFrame[5][:3, 3] - joint_position_WorldFrame[i][:3,3]))
            Jacobian_matrix[3:6, i] = joint_position_WorldFrame[i][0:3, col]
        else:
            Jacobian_matrix[0:3, i] = joint_position_WorldFrame[i][0:3, col]
            Jacobian_matrix[3:6, i] = [0, 0, 0]

    return Jacobian_matrix

def update(joint1, joint2, joint3, joint4, joint5, joint6, theta):
    joint1[4] = theta[0]# 更新机器人的位姿
    joint2[2] = theta[1]
    joint3[3] = theta[2]
    joint4[0] = theta[3]
    joint5[2] = theta[4]
    theta[5] = theta[5] % np.pi
    joint6[3] = theta[5]

def main():
    target_position = [0.4567499999999729, 0.13104999996508332, 0.06649999997312121, -3.1415926531795866, 0, -3.1415926535895866]
    target_position = np.array(target_position, dtype=np.float64)    
    e = 1e-7 # epsilon
    count = 0
    alpha_steplenth = 0.05

    theta, joints_matrixs, joints_types, joint1, joint2, joint3, joint4, joint5, joint6 = define_joints()
    dof = len(theta.flatten())

    while(count < 500):
        prev_matrix = np.eye(4) #计算并存储每个joint世界坐标系下的其次变换矩阵
        joint_position_WorldFrame = []

        for xyzrpy in joints_matrixs:
            matrix = xyz_to_transMatrix(xyzrpy)
            prev_matrix = FK(prev_matrix, matrix)
            joint_position_WorldFrame.append(prev_matrix)

        q_current = trans_Matrix_to_quaternion(joint_position_WorldFrame[5]) # 计算四元数
        rpy_current = quaternion_to_rpy(q_current)
        rpy_error = target_position[3:] - rpy_current
        rpy_error_norm = np.linalg.norm(rpy_error) # 计算四元数的误差

        now = joint_position_WorldFrame[5][:3, 3]
        t = target_position[0:3]
        xyz_error = t - now
        deviat = np.linalg.norm(xyz_error) # 计算误差，如果足够小就结束迭代
        if deviat < e and rpy_error_norm < e:
            print("Deviate is small enough")
            print(theta)
            return 0

        Jacobian_matrix = Jacobian_calculate(dof, joint_position_WorldFrame, joints_types) # 计算雅可比矩阵
        count += 1
        
        error = np.concatenate((xyz_error, rpy_error)).reshape(6, 1)
        J_pseudo_inverse = np.linalg.pinv(Jacobian_matrix) # 计算伪逆雅可比矩阵


        theta = theta + alpha_steplenth * J_pseudo_inverse @ error
        '''delta_theta = Damped_least_square(J_pseudo_inverse, 0.02, error).reshape(6,1)
        theta += delta_theta'''
        update(joint1, joint2, joint3, joint4, joint5, joint6, theta) # update joints

        if count % 2 == 0:
            print(theta)
            print(count)
            print('   ') #6x2
            print(deviat)
            print(joint_position_WorldFrame[5][:3,3], quaternion_to_rpy(trans_Matrix_to_quaternion(joint_position_WorldFrame[5])))

if __name__ == "__main__":
    main()


