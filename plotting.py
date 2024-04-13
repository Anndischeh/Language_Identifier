import matplotlib.pyplot as plt

class Plotting:
    def plot_learning_curve(self, hist):
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Categorical Crossentropy')
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='val')
        plt.legend()
        plt.show()

    def plot_accuracy_curve(self, hist):
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(hist.history['accuracy'], label='train')
        plt.plot(hist.history['val_accuracy'], label='val')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, cm, lang_list):
        plt.figure(figsize=(12,10))
        plt.title('Confusion Matrix', fontsize=20)
        sns.heatmap(cm, xticklabels=lang_list, yticklabels=lang_list, cmap='rocket_r', linecolor='white', linewidth=.005)
        plt.xlabel('Predicted Language', fontsize=15)
        plt.ylabel('True Language', fontsize=15)
        plt.show()
