import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class MoralProbe:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
        self.top_n = 10  # Number of top neurons to track
    
    def train(self, activations, labels):
        X = activations.reshape(activations.shape[0], -1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        
        # Get important neurons (absolute value of coefficients)
        self.important_neurons = np.argsort(np.abs(self.model.coef_[0]))[-self.top_n:]
        
        print(f"Probe accuracy: {score:.2f}")
        print(f"Top {self.top_n} important neurons: {self.important_neurons}")
        
        return self.important_neurons
    
    def get_important_neurons(self):
        return self.important_neurons