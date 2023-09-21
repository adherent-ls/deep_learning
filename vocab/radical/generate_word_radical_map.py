import json


def radical_split(word, radical_map):
    if word not in radical_map:
        return [word]
    if len(radical_map[word]) == 1:
        return radical_map[word]

    radicals = radical_map[word]
    new_radicals = []
    for item in radicals:
        v = radical_split(item, radical_map)
        new_radicals.extend(v)
    return new_radicals


def main():
    chinese = open('../word/ch_dict_chinese.txt', 'r', encoding='utf-8').readlines()
    number = open('../word/number', 'r', encoding='utf-8').readlines()
    english = open('../word/english', 'r', encoding='utf-8').readlines()

    vocab = chinese + number + english
    vocab = [item.strip('\n') for item in vocab]
    vocab.sort()

    radical_map = {}
    radical_lines = open('./IDS_dictionary.txt', 'r', encoding='utf-8').readlines()
    for i, line in enumerate(radical_lines):
        word, radicals = line.strip('\n').split(':')
        radical_map[word] = radicals.split(' ')[:-1]

    radical_json = {}
    for item in vocab:
        radicals = radical_split(item, radical_map)
        radical_json[item] = radicals
        # if item in radical_map:
        #     radical_json[item] = radical_map[item]
        # else:
        #     radical_json[item] = [item]
        #     print(item)
    json.dump(radical_json, open('./word_radical.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
