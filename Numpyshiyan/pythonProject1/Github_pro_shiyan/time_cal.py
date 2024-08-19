def add_time(start, duration, day=None):
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    Time = start.split()
    start_hour, start_minute = map(int, Time[0].split(':'))

    # map()将指定的函数应用于可迭代对象（如列表、元组等）的每一个元素，并返回一个迭代器
    # 需要对序列中的每个元素进行相同操作时，而不需要显式地写出循环。它使代码更加简洁且易于阅读。
    if Time[1] == 'PM':
        start_hour += 12

    duration_hour, duration_minute = map(int, duration.split(':'))

    # 计算新的时间
    new_hour = start_hour + duration_hour
    new_minute = start_minute + duration_minute
    if new_minute >= 60:
        new_minute -= 60
        new_hour += 1
    days_later = new_hour // 24
    new_hour %= 24

    # 转换回12小时制
    if new_hour == 0:
        new_hour = 12
        period = 'AM'
    elif new_hour < 12:
        period = 'AM'
    elif new_hour == 12:
        period = 'PM'
    else:
        new_hour -= 12
        period = 'PM'

    new_minute = str(new_minute).zfill(2)

    new_time = f"{new_hour}:{new_minute} {period}"

    # 处理星期的变化
    if day:
        day = day.lower()
        current_day_index = days_of_week.index(day)
        new_day_index = (current_day_index + days_later) % 7
        new_day = days_of_week[new_day_index].capitalize()  # 将字符串的第一个字母转换为大写，而其余的字母转换为小写。
        new_time += f", {new_day}"

    # 如果需要处理跨天的情况
    if days_later == 1:
        new_time += " (next day)"
    elif days_later > 1:
        new_time += f" ({days_later} days later)"

    return new_time


if __name__ == '__main__':
    result = add_time('8:16 PM', '466:02', 'tuesday')
    print(result)
