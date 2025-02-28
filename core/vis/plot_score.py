import numpy as np
import matplotlib.pyplot as plt


class PlotScore:
    """Plot the training and testing scores."""
    def __init__(self, metrics, modes, save_dir=None, display=False):    
        self.metrics = metrics 
        self.modes = modes
        self.score = {mode: {metric: [] for metric in metrics} for mode in modes}
        self.save_dir = save_dir
        self.display = display
        
        
    def append(self, mode, metric, value):
        self.score[mode][metric].append(value)
        
        
    def plot(self, num_epoch, save_dir=None):
        
        len_metrics = len(self.metrics)
        len_modes = len(self.modes)
        
        fig, ax = plt.subplots(len_modes,  len_metrics , figsize=(15 * len_metrics, 5*len_modes))
        
        for i, mode in enumerate(self.modes):
            for j, metric in enumerate(self.metrics):
                ax[i, j].plot(np.arange(num_epoch), self.score[mode][metric])
                ax[i, j].set_title(f"{mode} - {metric}")
                ax[i, j].set_xlabel("Epoch")
                ax[i, j].set_ylabel(metric)
                
        if save_dir is not None:
            fig.savefig(f"{save_dir}/score_plot.png")

        if self.display:
            plt.show()
            
    def save_results(self, save_dir=None, params=None):
        if save_dir is not None:
            with open(f"{save_dir}/score.txt", "w") as f:
                
                f.write("Parameters:\n")
                for key, value in params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                for mode in self.modes:
                    f.write(f"{mode}\n")
                    for metric in self.metrics:
                        f.write(f"{metric}: {self.score[mode][metric]}\n")
                    f.write("\n")
                f.write("\n")

        

