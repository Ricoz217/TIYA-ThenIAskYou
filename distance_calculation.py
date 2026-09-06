import math


try:
    base_x, base_y = 0.0, 0.0
    while True:
        input_data_1 = input("输入第一组坐标: ")
        if not input_data_1:
            x1, y1 = base_x, base_y

        else:
            splited = input_data_1.split(' ')
            if not len(splited) == 2:
                print("输入的数组长度不符")
                continue

            try:
                x1 = float(splited[0])
                y1 = float(splited[1])

            except ValueError:
                print("输入的数组不是数字")
                continue

            base_x, base_y = x1, y1

        input_data_2 = input("输入第二组坐标: ")
        splited = input_data_2.split(' ')
        if not len(splited) == 2:
            print("输入的数组长度不符")
            continue

        try:
            x2 = float(splited[0])
            y2 = float(splited[1])

        except ValueError:
            print("输入的数组不是数字")
            continue

        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dx, dy))
        if angle < 0:
            angle += 360

        print(f"distance: {distance:.4f}; angle: {angle:.4f}")

except KeyboardInterrupt:
    exit(0)