import re


with open('test', 'r') as f:
    reader = f.readlines()
    nums = []
    for i, line in enumerate(reader):
        num = re.findall("\d+", line)
        for item in num:
            if item in nums:
                print(i, line)
            else:
                nums.append(item)




