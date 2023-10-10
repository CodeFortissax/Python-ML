import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class SupportVectorMachine:
    def __init__(self, visualization):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        self.w = None  # Initialize w to None
        self.b = None  # Initialize b to None
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1],
                     [-1, 1],
                     [-1, 1],
                     [1, 1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:

                for feature in featureset:

                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        #for yi in self.data:
         #   for featureset in self.data[yi]:
          #      for feature in featureset:
           #         if max_feature_value < feature:
            #            max_feature_value = feature
             #       if min_feature_value > feature:
              #          min_feature_value = feature

        #for featureset in self.data[yi]:
         #   max_feature_value = featureset[0]
         #   min_feature_value = featureset[0]
          #  for feature in featureset:
           #     if max_feature_value < feature:
            #        max_feature_value = feature
             #   if min_feature_value > feature:
              #      min_feature_value = feature


        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #point of expense
                      self.max_feature_value * 0.001]

        #extremely expensive
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        #constraint: yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])

            # Check if the norms list is empty
            if not norms:
                print('No optimal solution found')
                return

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            #self.w = opt_choice[0]

            latest_optimum = opt_choice[0][0]+step*2


    def predict(self, features):
        if self.w is None or self.b is None:
            print("Model has not been trained. Cannot make predictions.")
            return 0  # Return a default classification
        # (w.x+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self, ax):
        [[ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane: w.x+b
        #v = w.x+b
        #psv = 1
        #ngv = -1
        #db = 0
        def hyperplane(w, x, b, v):
            # Ensure w is a valid vector
            if np.isscalar(w):
                w = np.array([w, 0])

            # Check if the denominator is zero
            if np.abs(w[1]) < 1e-10:
                return None

            return (-np.dot(w, x) - b + v) / w[1]

            #if np.isscalar(w):
             #   w = np.array([w])
            #return (-np.dot(w, x) - b + v) / w[1] if w[1] != 0 else None
            #return (-(w[0]*x-b+v)) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b)=1
        #positive support vector (psv)
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b)=-1
        # negative support vector (nsv)
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b)=0
        # decision boundary (db)
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1:np.array([[1, 7],
                          [2, 8],
                          [3, 8]]),
             1:np.array([[5, 1],
                         [6, -1],
                         [7, 3]])}

svm_ = SupportVectorMachine(visualization=False)
#svm_.visualization = False
svm_.fit(data=data_dict)

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]

for p in predict_us:
    svm_.predict(p)

fig, ax = plt.subplots()
#svm_.visualize = ax
svm_.visualize(ax)
plt.show


#This code does not get the hyperplanes and the decision boundary