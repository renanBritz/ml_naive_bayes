class NaiveBayes:
    x = []
    y = []
    priori = {}

    def fit(self, x, y):
        self.x = x
        self.y = y

        if len(x) != len(y):
            raise 'x and y must have the same length!'
        
        self.__calculate_priori()
    
    def __calculate_priori(self):
        for i in range(0, len(self.y)):
            if self.y[i] in self.priori:
                self.priori[self.y[i]] += 1
            else:
                self.priori[self.y[i]] = 1

        for _class in self.priori:
            self.priori[_class] = self.priori[_class] / len(self.y)

    def predict(self, x):
        for _class in self.priori:
            prod = 1

            for xi in self.x:

        pass