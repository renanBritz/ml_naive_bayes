class NaiveBayes:
    x = []
    y = []
    a_priori = {}
    fitted = False

    def fit(self, x, y):
        self.x = x
        self.y = y

        if len(x) != len(y):
            raise 'x and y must have the same length!'

        self.__calculate_a_priori()
        self.fitted = True

    def __calculate_a_priori(self):
        for i in range(0, len(self.y)):
            if self.y[i] in self.a_priori:
                self.a_priori[self.y[i]] += 1
            else:
                self.a_priori[self.y[i]] = 1

        for _class in self.a_priori:
            self.a_priori[_class] = self.a_priori[_class] / len(self.y)

    def __get_class_instances(self, _class):
        instances = []
        for i in range(0, len(self.x)):
            if self.y[i] == _class:
                instances.append(self.x[i])
        return instances

    def __argmax(self, proba):
        argmax = None
        for i in range(0, len(proba)):
            if argmax is None:
                argmax = i
                continue

            if proba[i] > proba[argmax]:
                argmax = i

        return argmax

    def predict(self, x):
        if not self.fitted:
            raise 'You must fit the model before predicting'

        proba = []
        for _class in self.a_priori:
            prod = 1
            instances = self.__get_class_instances(
                _class
            )

            for xi in range(0, len(x)):
                value_instances = list(
                    filter(lambda instance: instance[xi] == x[xi], instances)
                )
                prod *= len(value_instances) / len(instances)

            prod *= self.a_priori[_class]
            proba.append(prod)

        print(proba)
        return self.__argmax(proba)
