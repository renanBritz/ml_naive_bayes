class NaiveBayes:
    x = []
    y = []
    a_priori = {}
    instances = {}
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
        if _class in self.instances:
            return self.instances[_class]

        instances = []
        for i in range(0, len(self.x)):
            if self.y[i] == _class:
                instances.append(self.x[i])

        self.instances[_class] = instances
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

    def get_classes(self):
        return list(self.a_priori.keys())

    def P(self, attr, val, target):
        instances = self.__get_class_instances(
            target
        )
        value_instances = list(
            filter(lambda instance: instance[attr] == val, instances)
        )

        return len(value_instances) / len(instances)

    def get_a_posteriori(self, x):
        if not self.fitted:
            raise 'You must fit the model first!'

        a_posteriori = []
        for _class in self.a_priori:
            prod = 1

            for xi in range(0, len(x)):
                prod *= self.P(xi, x[xi], _class)

            prod *= self.a_priori[_class]
            a_posteriori.append(prod)
        return a_posteriori

    def predict(self, x):
        return self.__argmax(self.get_a_posteriori(x))
