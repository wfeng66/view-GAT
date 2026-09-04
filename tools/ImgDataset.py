import numpy as np
import glob
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms
import h5py
import cv2

class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True, dataset='modelnet40'):
        if dataset == 'modelnet40':
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                             'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                             'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                             'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                             'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        elif dataset == 'scanobjectnn':
            self.classnames=['bag','bin','box','cabinet','chair','desk','display',
                             'door','shelf','table','bed','pillow','sink','sofa','toilet']
        elif dataset == 'colombia':
            self.classnames=['0','1','2','3','4','5']
        elif dataset == 'roofn3d':
            self.classnames=['0','1']
        self.dataset = dataset
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]

        self.filepaths = []
        for i in range(len(self.classnames)):
            class_dir = self.classnames[i]
            # print(f"DEBUG: Searching for files in: {parent_dir+'/'+class_dir+'/'+set_+'/*.png'}")
            all_files = sorted(glob.glob(parent_dir+'/'+class_dir+'/'+set_+'/*.png'))
            
            # Group files by model (assuming 20 views per model in the original dataset)
            views_per_model_original = 20
            num_models_in_class = len(all_files) // views_per_model_original
            
            # Select exactly num_views from each model using uniform spacing
            selected_files = []
            for model_idx in range(num_models_in_class):
                model_files = all_files[model_idx * views_per_model_original : (model_idx + 1) * views_per_model_original]
                # Use linspace to select evenly spaced indices
                indices = np.linspace(0, len(model_files) - 1, self.num_views, dtype=int)
                selected_files.extend([model_files[j] for j in indices])

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(selected_files)
            else:
                # Limit to num_models models (each with num_views views)
                max_files = num_models
                self.filepaths.extend(selected_files[:min(max_files, len(selected_files))])
                
        print(f"DEBUG: Total filepaths collected: {len(self.filepaths)} (should be num_models * num_views if num_models > 0)")
        print(f"DEBUG: Sample filepaths: {self.filepaths[:10]}")

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])

class MultiviewImgDataset_polar(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True, dataset='modelnet40'):
        print(f'DEBUG: Dataset init with root_dir={root_dir}, dataset={dataset}')
        if dataset == 'modelnet40':
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                             'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                             'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                             'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                             'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        elif dataset == 'scanobjectnn':
            self.classnames=['bag','bin','box','cabinet','chair','desk','display',
                             'door','shelf','table','bed','pillow','sink','sofa','toilet']
        elif dataset == 'colombia':
            self.classnames=['0','1','2','3','4','5']
        elif dataset == 'roofn3d':
            self.classnames=['0','1']
        self.dataset = dataset
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            class_dir = self.classnames[i]
            all_files = sorted(glob.glob(parent_dir+'/'+class_dir+'/'+set_+'/*.png'))
            
            # Group files by model (assuming 20 views per model in the original dataset)
            views_per_model_original = 20
            num_models_in_class = len(all_files) // views_per_model_original
            
            # Select exactly num_views from each model using uniform spacing
            selected_files = []
            for model_idx in range(num_models_in_class):
                model_files = all_files[model_idx * views_per_model_original : (model_idx + 1) * views_per_model_original]
                # Use linspace to select evenly spaced indices
                indices = np.linspace(0, len(model_files) - 1, self.num_views, dtype=int)
                selected_files.extend([model_files[j] for j in indices])

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(selected_files)
            else:
                # Limit to num_models models (each with num_views views)
                max_files = num_models
                self.filepaths.extend(selected_files[:min(max_files, len(selected_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):

            im = cv2.imread(self.filepaths[idx * self.num_views + i],0)
            # cv2.imshow('0',im)
            # cv2.waitKey()
            # im = Image.open(self.filepaths[idx * self.num_views + i])
            # im = im.load()
            # s = im[125,125]
            # im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            w = im.shape[0]
            # this will create some dead pixels
            m = w / np.log(w * np.sqrt(2) / 2)
            # but this may crop parts of the original img:
            # m = w/np.log(w/2)
            im = cv2.logPolar(im, ((w - 1.) / 2., (w - 1.) / 2.), m,
                              cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
            # cv2.imshow('1',im2)
            # cv2.waitKey()
            im= np.expand_dims(im,-1).repeat(3,axis=-1)
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])

class SingleImgDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True, dataset='modelnet40'):
        if dataset == 'modelnet40':
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                             'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                             'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                             'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                             'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        elif dataset == 'scanobjectnn':
            self.classnames=['bag','bin','box','cabinet','chair','desk','display',
                             'door','shelf','table','bed','pillow','sink','sofa','toilet']
        elif dataset == 'colombia':
            self.classnames=['0','1','2','3','4','5']
        elif dataset == 'roofn3d':
            self.classnames=['0','1']
        self.dataset = dataset
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            class_dir = self.classnames[i]
            print(parent_dir+'/'+class_dir+'/'+set_+'/*.png')
            all_files = sorted(glob.glob(parent_dir+'/'+class_dir+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # Shuffle the filepaths
            np.random.shuffle(self.filepaths)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)


class ScanObjectNNDataset(torch.utils.data.Dataset):
    """Dataset class for ScanObjectNN point cloud data with view rendering"""
    
    def __init__(self, h5_file_path, num_views=20, test_mode=False, num_models=0):
        self.h5_file_path = h5_file_path
        self.num_views = num_views
        self.test_mode = test_mode
        
        # ScanObjectNN has 15 classes
        self.classnames = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 
                          'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
        
        # Load data from HDF5 file
        with h5py.File(h5_file_path, 'r') as f:
            self.point_clouds = f['data'][:]  # (N, 2048, 3)
            self.labels = f['label'][:]       # (N,)
            self.masks = f['mask'][:]         # (N, 2048)
        
        # Limit number of models if specified
        if num_models > 0:
            self.point_clouds = self.point_clouds[:num_models]
            self.labels = self.labels[:num_models]
            self.masks = self.masks[:num_models]
        
        # Define view angles (same as original view-GCN)
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            self.view_angles = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                               [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                               [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                               [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                               [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            self.view_angles = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                               [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                               [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                               [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        else:
            # Use Fibonacci sphere sampling for arbitrary number of views
            self.view_angles = []
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(self.num_views):
                theta = 2 * np.pi * i / golden_ratio  # azimuthal angle
                phi_angle = np.arccos(1 - 2 * (i + 0.5) / self.num_views)  # polar angle
                x = np.sin(phi_angle) * np.cos(theta)
                y = np.sin(phi_angle) * np.sin(theta)
                z = np.cos(phi_angle)
                self.view_angles.append([x, y, z])
        
        # Normalize view angles
        self.view_angles = np.array(self.view_angles)
        self.view_angles = self.view_angles / np.linalg.norm(self.view_angles, axis=1, keepdims=True)
        
        # Define transforms
        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.point_clouds)
    
    def render_view(self, point_cloud, mask, view_angle, img_size=224):
        """Render a single view from point cloud"""
        # Filter valid points using mask
        valid_indices = mask > 0
        points = point_cloud[valid_indices]
        
        if len(points) == 0:
            # Return black image if no valid points
            return np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Normalize points to unit sphere
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Project points to 2D using orthographic projection
        # View direction is the negative of view_angle
        view_dir = -view_angle
        
        # Create rotation matrix to align view direction with z-axis
        z_axis = np.array([0, 0, 1])
        if np.allclose(view_dir, z_axis):
            R = np.eye(3)
        elif np.allclose(view_dir, -z_axis):
            R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            v = np.cross(view_dir, z_axis)
            s = np.linalg.norm(v)
            c = np.dot(view_dir, z_axis)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
        
        # Rotate points
        rotated_points = np.dot(points, R.T)
        
        # Project to 2D (orthographic projection)
        x_2d = rotated_points[:, 0]
        y_2d = rotated_points[:, 1]
        
        # Scale to image coordinates
        x_2d = (x_2d + 1) * img_size / 2
        y_2d = (y_2d + 1) * img_size / 2
        
        # Create depth image
        depth_img = np.zeros((img_size, img_size), dtype=np.float32)
        color_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Filter points within image bounds
        valid_mask = (x_2d >= 0) & (x_2d < img_size) & (y_2d >= 0) & (y_2d < img_size)
        x_2d = x_2d[valid_mask]
        y_2d = y_2d[valid_mask]
        z_vals = rotated_points[valid_mask, 2]
        
        if len(x_2d) > 0:
            # Convert to integer coordinates
            x_int = np.round(x_2d).astype(int)
            y_int = np.round(y_2d).astype(int)
            
            # Clamp to image bounds
            x_int = np.clip(x_int, 0, img_size - 1)
            y_int = np.clip(y_int, 0, img_size - 1)
            
            # Create depth buffer (closer points overwrite farther ones)
            for i in range(len(x_int)):
                if z_vals[i] > depth_img[y_int[i], x_int[i]]:
                    depth_img[y_int[i], x_int[i]] = z_vals[i]
                    # Color based on depth (closer = brighter)
                    intensity = int(255 * (z_vals[i] + 1) / 2)
                    color_img[y_int[i], x_int[i]] = [intensity, intensity, intensity]
        
        return color_img
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        
        # Render views
        views = []
        for i in range(self.num_views):
            view_img = self.render_view(point_cloud, mask, self.view_angles[i])
            # Convert to PIL Image
            view_pil = Image.fromarray(view_img)
            if self.transform:
                view_pil = self.transform(view_pil)
            views.append(view_pil)
        
        return (label, torch.stack(views), f"scanobjectnn_{idx}")
    
    @property
    def filepaths(self):
        """Compatibility property for Trainer class"""
        # Generate fake filepaths for compatibility with existing trainer
        return [f"scanobjectnn_{i}" for i in range(len(self.point_clouds) * self.num_views)]

class ScanObjectNNSingleDataset(torch.utils.data.Dataset):
    """Single view dataset for ScanObjectNN (for stage 1 training)"""
    
    def __init__(self, h5_file_path, num_views=20, test_mode=False, num_models=0):
        self.h5_file_path = h5_file_path
        self.num_views = num_views
        self.test_mode = test_mode
        
        # ScanObjectNN has 15 classes
        self.classnames = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 
                          'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
        
        # Load data from HDF5 file
        with h5py.File(h5_file_path, 'r') as f:
            self.point_clouds = f['data'][:]  # (N, 2048, 3)
            self.labels = f['label'][:]       # (N,)
            self.masks = f['mask'][:]         # (N, 2048)
        
        # Limit number of models if specified
        if num_models > 0:
            self.point_clouds = self.point_clouds[:num_models]
            self.labels = self.labels[:num_models]
            self.masks = self.masks[:num_models]
        
        # Define view angles (same as original view-GCN)
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            self.view_angles = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                               [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                               [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                               [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                               [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            self.view_angles = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                               [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                               [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                               [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]
        else:
            # Use Fibonacci sphere sampling for arbitrary number of views
            self.view_angles = []
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(self.num_views):
                theta = 2 * np.pi * i / golden_ratio  # azimuthal angle
                phi_angle = np.arccos(1 - 2 * (i + 0.5) / self.num_views)  # polar angle
                x = np.sin(phi_angle) * np.cos(theta)
                y = np.sin(phi_angle) * np.sin(theta)
                z = np.cos(phi_angle)
                self.view_angles.append([x, y, z])
        
        # Normalize view angles
        self.view_angles = np.array(self.view_angles)
        self.view_angles = self.view_angles / np.linalg.norm(self.view_angles, axis=1, keepdims=True)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.point_clouds) * self.num_views
    
    def render_view(self, point_cloud, mask, view_angle, img_size=224):
        """Render a single view from point cloud"""
        # Filter valid points using mask
        valid_indices = mask > 0
        points = point_cloud[valid_indices]
        
        if len(points) == 0:
            # Return black image if no valid points
            return np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Normalize points to unit sphere
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Project points to 2D using orthographic projection
        # View direction is the negative of view_angle
        view_dir = -view_angle
        
        # Create rotation matrix to align view direction with z-axis
        z_axis = np.array([0, 0, 1])
        if np.allclose(view_dir, z_axis):
            R = np.eye(3)
        elif np.allclose(view_dir, -z_axis):
            R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            v = np.cross(view_dir, z_axis)
            s = np.linalg.norm(v)
            c = np.dot(view_dir, z_axis)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
        
        # Rotate points
        rotated_points = np.dot(points, R.T)
        
        # Project to 2D (orthographic projection)
        x_2d = rotated_points[:, 0]
        y_2d = rotated_points[:, 1]
        
        # Scale to image coordinates
        x_2d = (x_2d + 1) * img_size / 2
        y_2d = (y_2d + 1) * img_size / 2
        
        # Create depth image
        depth_img = np.zeros((img_size, img_size), dtype=np.float32)
        color_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Filter points within image bounds
        valid_mask = (x_2d >= 0) & (x_2d < img_size) & (y_2d >= 0) & (y_2d < img_size)
        x_2d = x_2d[valid_mask]
        y_2d = y_2d[valid_mask]
        z_vals = rotated_points[valid_mask, 2]
        
        if len(x_2d) > 0:
            # Convert to integer coordinates
            x_int = np.round(x_2d).astype(int)
            y_int = np.round(y_2d).astype(int)
            
            # Clamp to image bounds
            x_int = np.clip(x_int, 0, img_size - 1)
            y_int = np.clip(y_int, 0, img_size - 1)
            
            # Create depth buffer (closer points overwrite farther ones)
            for i in range(len(x_int)):
                if z_vals[i] > depth_img[y_int[i], x_int[i]]:
                    depth_img[y_int[i], x_int[i]] = z_vals[i]
                    # Color based on depth (closer = brighter)
                    intensity = int(255 * (z_vals[i] + 1) / 2)
                    color_img[y_int[i], x_int[i]] = [intensity, intensity, intensity]
        
        return color_img
    
    def __getitem__(self, idx):
        # Calculate which point cloud and which view
        point_idx = idx // self.num_views
        view_idx = idx % self.num_views
        
        point_cloud = self.point_clouds[point_idx]
        label = self.labels[point_idx]
        mask = self.masks[point_idx]
        
        # Render single view
        view_img = self.render_view(point_cloud, mask, self.view_angles[view_idx])
        view_pil = Image.fromarray(view_img)
        
        if self.transform:
            view_pil = self.transform(view_pil)
        
        return (label, view_pil, f"scanobjectnn_{point_idx}_{view_idx}")
    
    @property
    def filepaths(self):
        """Compatibility property for Trainer class"""
        # Generate fake filepaths for compatibility with existing trainer
        return [f"scanobjectnn_{i}" for i in range(len(self.point_clouds) * self.num_views)]

