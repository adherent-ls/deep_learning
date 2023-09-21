import json


def main():
    data = json.load(open('./word_radical.json', 'r', encoding='utf-8'))

    radicals = set()
    word = set()
    max_l = 0
    for k, v in data.items():
        word.add(k)
        max_l = max(max_l, len(v))
        for item in v:
            radicals.add(item)
    radicals = list(radicals)
    radicals.sort()
    word = list(word)
    word.sort()
    print(len(radicals))
    print(len(word))
    print(max_l)
    f = open('./radical_v2.txt', 'w', encoding='utf-8')
    for item in radicals:
        f.write(f'{item}\n')
    f.close()
    f = open('./word_v2.txt', 'w', encoding='utf-8')
    for item in word:
        f.write(f'{item}\n')
    f.close()


if __name__ == '__main__':
    main()
