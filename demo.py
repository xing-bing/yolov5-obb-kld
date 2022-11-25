import os
import glob
import os

if __name__ == "__main__":
    task_dir = "./runs/detect/exp4/ship.txt"
    task_file = open(task_dir, 'w', encoding='utf-8')  # 打开类别文件
    result_dir = "./runs/detect/exp4/labels/"
    ids = os.listdir(result_dir)
    for id in ids:
        if not id.endswith("txt"):
            continue
        name = os.path.join(result_dir, id)
        with open(name, 'r') as f:
            boxes = f.readlines()  # 读取所有的行
            for box in boxes:  # 遍历每一行
                lines = box.split()
                if lines[0] == "1":
                    task_file.write(id.replace(".txt", "") + " " + lines[9] + " " + lines[1] + " " + lines[2] + " " + lines[3] + " " + lines[4] + " " + lines[5] + " " + lines[6] + " " + lines[7] + " " + lines[8] + "\n")
            f.close()
    task_file.close()