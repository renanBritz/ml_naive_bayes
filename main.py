from utils import load_csv, split_data
from naive_bayes import NaiveBayes


def main():
    data = load_csv('./data.csv')
    x, y = split_data(data, 'target')
    nb = NaiveBayes()
    nb.fit(x, y)
    print(nb.predict(['low', 'low', 'low']))


if __name__ == '__main__':
    main()
