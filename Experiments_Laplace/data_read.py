def data_read(str):

    # 打开并读取文件
    with open(str, 'r') as file:
        lines = file.readlines()

    # 去除每行末尾的换行符
    lines = [int(float(line.strip())) for line in lines]

    return lines

def star_2_ori(data_list):

    return sum([(i*(i-1)/2) for i in data_list])