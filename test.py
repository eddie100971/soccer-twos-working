class Test:
    def __init__(self, p1, p2):
        self.population1 = p1
        self.population2 = p2

        self.rollout = lambda: 6

    def fill(self, utility):
        self.utilities = utility
        for i in range(len(self.population1)):
                self.agents = self.population1[i]
                for j in range(len(self.population2)):
                    self.opponents = self.population2[j]
                    if len(self.utilities) == i: # new row
                        self.utilities.append([self.rollout()])
                    elif len(self.utilities[i]) == j: # existing row
                        self.utilities[i].append(self.rollout())
        for i in self.utilities:
            print(i)


if __name__ == "__main__":
    population1 = [[1.2], [4,5], [1,2]]
    population2 = [[2,3], [4,5], [1,2]]

    t = Test(population1, population2)

    t.fill(
         [
            [1,2],[1,2]
          ]
    )