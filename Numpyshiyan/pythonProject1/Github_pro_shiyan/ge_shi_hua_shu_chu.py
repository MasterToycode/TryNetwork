def arithmetic_arranger(problems, show_answers=False):
    # 检查问题数量
    if len(problems) > 5:
        return "Error: Too many problems."

    first_line = []
    second_line = []
    dashes = []
    answers = []

    for problem in problems:
        # 将问题拆分为操作数和运算符
        parts = problem.split()
        #将 problem 字符串按空格分割成一个列表，列表中的每个元素是由 split() 方法提取出来的子字符串
        if len(parts) != 3:
            return "Error: Each problem must have two operands and an operator."

        num1, operator, num2 = parts

        # 检查运算符是否合法
        if operator not in ["+", "-"]:
            return "Error: Operator must be '+' or '-'."

        # 检查操作数是否为数字
        if not num1.isdigit() or not num2.isdigit():
            return "Error: Numbers must only contain digits."

        # 检查操作数的长度
        if len(num1) > 4 or len(num2) > 4:
            return "Error: Numbers cannot be more than four digits."

        # 计算每个问题的宽度
        width = max(len(num1), len(num2)) + 2
        # 因为数字最大是4位，而且规定，最长的数字和符号之间必须有一个空格，算上符号占一位，所有每一行的输出的字符串之间的间隔为：1+4+1

        # 将格式化的字符串添加到各自的列表中
        first_line.append(num1.rjust(width))
        # rjust() 是 Python 字符串方法之一，用于在字符串的左侧填充空格或指定字符
        second_line.append(operator + num2.rjust(width - 1))
        # 因为符号已经占了一位，所以输出的数字的长度的字符串长度必须减一
        dashes.append("-" * width)

        if show_answers:
            if operator == "+":
                result = str(int(num1) + int(num2))
            else:
                result = str(int(num1) - int(num2))
            answers.append(result.rjust(width))

    # 将各行拼接成最终字符串
    arranged_problems = "    ".join(first_line) + "\n" + "    ".join(second_line) + "\n" + "    ".join(dashes)

    if show_answers:
        arranged_problems += "\n" + "    ".join(answers)

    return arranged_problems

# 示例调用：
print(arithmetic_arranger(["32 + 698", "3801 - 2", "45 + 43", "123 + 49"], True))
