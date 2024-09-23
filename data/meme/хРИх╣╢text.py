# 定义文件名列表
filenames = ['data/dblp/cascades.txt', 'data/dblp/cascades1.txt']
# 定义目标文件名
output_filename = 'data/dblp/cascades2.txt'

# 使用'with'语句确保文件适当地打开和关闭
with open(output_filename, 'w', encoding='utf-8') as outfile:
    # 循环遍历文件名列表
    for fname in filenames:
        # 再次使用'with'语句打开每个文件
        with open(fname, 'r', encoding='utf-8') as infile:
            # 读取文件内容并写入目标文件
            outfile.write(infile.read())
            # 如果需要在文件内容之间添加分隔符（例如空行），可以在这里添加


print("文件合并完成。")
