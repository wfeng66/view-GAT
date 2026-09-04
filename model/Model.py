import torch
import torch.nn as nn
import os
import glob
class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        torch.save(self.state_dict(),
                   os.path.join(complete_path,
                                "model-{}.pth".format(str(epoch).zfill(5))))

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path + "/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf, weights_only=True))
    
    def load_best_model(self, path):
        """Load the best model from the given path"""
        best_model_path = os.path.join(path, f"{self.name}_best.pth")
        if os.path.exists(best_model_path):
            try:
                self.load_state_dict(torch.load(best_model_path, weights_only=True))
                print(f"Best model loaded from {best_model_path}")
                return True
            except Exception as e:
                print(f"Error loading best model from {best_model_path}: {e}")
                return False
        else:
            print(f"Best model file not found at {best_model_path}")
            return False
    
    def save_best_model(self, path, epoch):
        """Save the current model as the best model"""
        best_model_path = os.path.join(path, f"{self.name}_best.pth")
        try:
            # Ensure the directory exists
            os.makedirs(path, exist_ok=True)
            torch.save(self.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} to {best_model_path}")
        except Exception as e:
            print(f"Error saving best model to {best_model_path}: {e}")
    
    def get_best_model_path(self, path):
        """Get the path to the best model file"""
        return os.path.join(path, f"{self.name}_best.pth")


