一个6joint的机械臂，每个joint上xyzrpy中的某一个参数随即初始化。先通过正运动学算出末端执行器的位姿，和目标位姿比较后通过误差迭代。
迭代完成后得到的每个joint的参数需要再通过正运动验证是否与目标姿态一致
