def combine_strings1(str1, str2):
    result = []
    for i in str1:
        for j in str2:
            if i != j:
                result.append(i+j)
    return result

def combine_strings(*strings, output_length=2):
    if output_length < 1:
        return []

    result = []
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            for s1 in strings[i]:
                for s2 in strings[j]:
                    if s1 != s2:
                        combined = s1 + s2
                        if len(combined) == output_length:
                            result.append(combined)
    return result

print(combine_strings("abc", "def","ghi", output_length=3))


print(combine_strings1("abc", "def"))
