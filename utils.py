def load_csv(path, delimiter=','):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.rstrip().split(delimiter))

    return data


def split_data(data, target_attr):
    copy = data[:]
    cols = copy[0]
    values = copy[1:]
    target_index = cols.index(target_attr)

    x = []
    y = []
    for i in range(0, len(values)):
        y.append(values[i].pop(target_index))
        x.append(values[i])

    return [x, y]
