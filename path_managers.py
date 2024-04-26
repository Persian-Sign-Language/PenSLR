from datetime import datetime
import os
import names

class PathManager:
    def __init__(
            self,
            exp_dir='exp',
            exp_path=None,
            model_name=None
    ):
        if exp_path:
            self.exp_path = os.path.join(exp_dir, exp_path)
            self.models_path = os.path.join(self.exp_path, "models")
            self.visualization_path = os.path.join(self.exp_path, "visualization")
        
        else:
            if model_name:
                self.exp_path = os.path.join(exp_dir, model_name +"_"+ "_".join(names.get_full_name().split(" ")))
                while os.path.exists(self.exp_path):
                    self.exp_path = os.path.join(exp_dir, model_name +"_"+ "_".join(names.get_full_name().split(" ")))
                os.makedirs(self.exp_path, exist_ok=True)
                self.models_path = os.path.join(self.exp_path, "models")
                os.makedirs(self.models_path, exist_ok=True)
                self.visualization_path = os.path.join(self.exp_path, "visualization")
                os.makedirs(self.visualization_path, exist_ok=True)
            else:
                raise ValueError("Please put model name or exp path")
        if not os.path.exists(self.exp_path):
            raise ValueError("Experiment directory not found!")
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.visualization_path, exist_ok=True)
        print("##################################################################")
        print('Experiment name      :', self.exp_path[4:])
        print("##################################################################")
    
    def find_last_model(self):
        # Get all model files
        files = [f for f in os.listdir(self.models_path) if f[:2] == 'M_']
        # if there is no file
        if len(files) == 0:
            print("Path To Model        :", None)
            print("Path To Optimizer    :", None)
            print("Path To dataloaders  :", None)
            return None, None, None, None, 1
        # Sort model files with respect to epoch number
        files = sorted(files, reverse=True, key=self.custom_sort_key)
        model = files[0]
        starting_epoch = model.replace("M_", "").split(".")[0]
        model_path = os.path.join(self.models_path, model)
        # optimizer path
        optimizer_path = os.path.join(self.models_path, f'O_{starting_epoch}.pth')
        optimizer_path = optimizer_path if os.path.exists(optimizer_path) else None
        # metrics path
        metrics_path = model_path.replace("M_", "METRICS_").replace("pth", "met")
        metrics_path = metrics_path  if os.path.exists(metrics_path) else None
        # dataloader path
        dl_path = os.path.join(self.models_path, 'XY.dl')
        dl_path = dl_path if os.path.exists(dl_path) else None
        # print warning if one of them is missing
        if optimizer_path is None or dl_path is None or metrics_path is None:
            print(f"Warning: optimizer path or data loader path or metrics path corresponding to model({model_path}) is missing.")
            print("##################################################################")
        print("Path To Model        :", model)
        print("Path To Optimizer    :", optimizer_path)
        print("Path To dataloaders  :", dl_path)
        print("Path To metrics      :", metrics_path)
        return model_path, optimizer_path, dl_path, metrics_path, int(starting_epoch) + 1
    
    def summary_writer_path(self):
        now = datetime.now()
        run_name = f'{now.strftime("%Y_%m_%d_%H_%M_%S")}'
        return os.path.join(self.visualization_path, run_name), now.strftime("%Y/%m/%d %H:%M:%S")
    
    def custom_sort_key(self, file_name):
        parts = file_name.split('_')  # Split the filename by underscore
        num = int(parts[1].split('.')[0])  # Extract the epoch 
        return num
