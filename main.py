from utils import load_csv, split_data
from naive_bayes import NaiveBayes


def main():
    data = load_csv('./data.csv')
    x, y = split_data(data, 'target')
    model = NaiveBayes()
    model.fit(x, y)
    classes = model.get_classes()

    print('Questionário Item B:')
    print('A priori:', model.a_priori)

    print('\nQuestionário Item C:')
    print('P(price=med|target=acc) =', model.P(
        attr=0, val='med', target='acc'))

    x_test = ['low', 'small', 'high']
    print('\nQuestionário item D')
    print('Probabilidades:', model.get_a_posteriori(x_test))
    print('Classe prevista:', classes[model.predict(x_test)])

    print('\nQuestionário item E')
    print('P(safety=low|target=acc) =', model.P(
        attr=2, val='low', target='acc'))


if __name__ == '__main__':
    main()
