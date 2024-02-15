######################################################################

# WARNING: ForceSight builds on an array of technologies. Hello Robot
# has not verified that the ForceSight_LICENSE is compatible with the
# licenses for these technologies. USE AT YOUR OWN RISK.

# While working at Hello Robot, Charlie Kemp (Charles C. Kemp) created
# this file. He derived the code from the official ForceSight GitHub
# repository, which he downloaded on February 1, 2024. You can find
# the ForceSight repository's MIT License copied below.

# Hello Robot and Charlie Kemp assign the same MIT License (included
# below) to any contributions they have made to this file.

# Official ForceSight websites:

#          https://github.com/force-sight/forcesight

#          https://force-sight.github.io/

# --------------------------------------------------------------------

# forcesight_min.py is a simplified single file that can be used to
# run a pretrained ForceSight model with minimal dependencies. The
# associated forcesight_min_requirements.txt file can be used to
# install the required Python packages in a Python virtual
# environment.

# To use this code, you will need to copy a pretrained ForceSight
# model into the same directory. The code has only been tested with
# "model_best.pth". As of February 2, 2024, you can download this 1.29
# GB model from the following OneDrive folder.

# https://onedrive.live.com/?authkey=%21ALvdUAiUg4s8LPY&id=79F9A071FA899B37%2179715&cid=79F9A071FA899B37

# --------------------------------------------------------------------

# ForceSight is the result of academic research at Georgia Tech, as
# reported in the following peer-reviewed publication.

# Jeremy A. Collins and Cody Houff and You Liang Tan and Charles
# C. Kemp. ForceSight: Text-Guided Mobile Manipulation with
# Visual-Force Goals. Accepted to the IEEE International Conference on
# Robotics and Automation (ICRA), 2024.

# @InProceedings{collins2023forcesight,
#       title={ForceSight: Text-Guided Mobile Manipulation with Visual-Force Goals}, 
#       author={Jeremy A. Collins and Cody Houff and You Liang Tan and Charles C. Kemp},
#       booktitle={Accepted to the IEEE International Conference on Robotics and Automation (ICRA)},  
#       year={2024},
# }

# URL for the ArXiv version of the publication:

#          https://arxiv.org/abs/2309.12312

######################################################################

# FORCESIGHT LICENSE FROM THE OFFICIAL FORCESIGHT REPOSITORY

# To the extent permitted by the component technologies and the
# sources from which this code was derived, this license covers the
# content of this file.
# --------------------------------------------------------------------

# MIT License

# Copyright (c) 2023 force-sight and Copyright (c) 2024 Hello Robot

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

######################################################################

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance
import timm
from transformers import T5Tokenizer
from skimage.feature import peak_local_max


########################################################################

# when the gripper is -0.3 orientated downwards
EEF_PITCH_WEIGHT_OFFSET = np.array([-0.22010994, -0.02645493,  0.3561182, 0, 0, 0])

########################################################################

def ft_to_cam_rotation(custom_pitch=(10/90)*np.pi/2):
    """
    custom_pitch: camera pitch angle, down is -ve and up is +ve
    """
    custom_pitch = np.array([[1, 0, 0],
                [0, np.cos(custom_pitch), -np.sin(custom_pitch)],
                [0, np.sin(custom_pitch), np.cos(custom_pitch)]])
    return np.array([[0, 1, 0],
                     [-1, 0, 0],
                     [0, 0, 1]])@custom_pitch

########################################################################

def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                     [0,             0,              1]])

def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

class DefaultIntrinsic:
    """
    This is the default intrinsic from the D405 realseense rgb camera with 848
    """
    ppx = 415.507537841797
    ppy = 237.871643066406
    fx = 431.125
    fy = 430.667
    coeffs = [-0.053341, 0.0545209, 0.000824648, 0.000749805, -0.0171459]

    # https://github.com/IntelRealSense/librealsense/issues/3473#issuecomment-474637827
    depth_scale = 9.9999e-05

    def cam_mat(self):
        return camera_matrix(self)

    def cam_dist(self):
        return fisheye_distortion(self)

class Intrinsic640(DefaultIntrinsic):
    ppx = 311.508
    ppy =  237.872
    fx = 431.125
    fy = 430.667
    coeffs = [-0.053341, 0.0545209, 0.000824648, 0.000749805, -0.0171459]

class CustomIntrinsic(DefaultIntrinsic):
    def __init__(self, camera_info):
        self.coeffs = camera_info['distortion_coefficients']
        cam_mat = camera_info['camera_matrix']
        self.ppx = cam_mat[0,2]
        self.ppy = cam_mat[1,2]
        self.fx = cam_mat[0,0]
        self.fy = cam_mat[1,1]

########################################################################

class RGBDViT(nn.Module):
    def __init__(self, size, num_classes=0, pretrained=True):
        super(RGBDViT, self).__init__()
        print(f'Loading ViT-{size}...')
        print('Pretrained: ', pretrained)
            
        if size == 'tiny':
            self.model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
            self.embed_dim = 192
        elif size == 'small':
            self.model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
            self.embed_dim = 384
        elif size == 'base':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
            self.embed_dim = 768
        elif size == 'large':
            # ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
            self.model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=num_classes) 
            self.embed_dim = 1024
        
        # Modify the patch embedding layer to accept 4 channels (RGBD)
        original_patch_embedding = self.model.patch_embed.proj # (768, 3, 16, 16)
        # (768, 4, 16, 16). Conv with stride=16 is equivalent to having a linear transformation for each patch
        self.model.patch_embed.proj = nn.Conv2d(4, self.model.embed_dim, kernel_size=(16, 16), stride=(16, 16), bias=False) 

        # Initialize the depth channel weights by averaging the weights from the RGB channels
        with torch.no_grad():
            self.model.patch_embed.proj.weight[:, :3] = original_patch_embedding.weight.clone()
            # Average the weights from the RGB channels
            self.model.patch_embed.proj.weight[:, 3] = original_patch_embedding.weight.mean(dim=1) 

        # Register a forward hook for each block
        self.features = {}
        for idx, block in enumerate(self.model.blocks):
            block.register_forward_hook(self.hook)

        print('vit params: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def hook(self, module, input, output):
        self.features[module] = output

    def forward(self, x):
        self.features.clear()  # Clear the feature dictionary at the start of each forward pass
        self.model.patch_embed.proj.requires_grad = True # always train the patch embedding layer
        return self.model(x)


class ConvDecoder(nn.Module):
    def __init__(self, image_model, patch_size, num_patches, num_channels=1):
        # takes in a batch of patch embeddings (196, 1024) and outputs a batch of images (1, 224, 224)
        super(ConvDecoder, self).__init__()
        self.image_model = image_model

        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.image_model.embed_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # 14 x 14 x 512
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 28 x 28 x 256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 56 x 56 x 1
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 112 x 112 x 1
        )

    def forward(self, x):
        # x.shape will be (batch_size, 196, 1024)
        x = x.transpose(1, 2)  # swap the patch and embedding dimensions
        # x.shape is now (batch_size, 1024, 196)
        
        B, C, N = x.shape
        H = W = int(N ** 0.5)  # assumes number of patches is a perfect square

        x = x.view(B, C, H, W)  # rearrange patches to 2D grid
        # x.shape is now (batch_size, 1024, 14, 14)

        # upscale to 224x224
        x = self.up(x) # (batch_size, 1, 224, 224)
        return x  

        
########################################################################


class ClassifierFreeGuidance(nn.Module):
    def __init__(self, image_model, text_model, hidden_dim=256, num_outputs=10):
        super(ClassifierFreeGuidance, self).__init__()
        
        self.image_model = image_model

        #  Dimensions for image and text features
        image_feature_dim = self.image_model.embed_dim
         
        # FiLM
        self.text_conditioner = TextConditioner(
            model_types = 't5',    
            hidden_dims = (1024,),
            hiddens_channel_first = False,
            cond_drop_prob = 0.2  # conditional dropout 20% of the time, must be greater than 0. to unlock classifier free guidance
        ).cuda()


    @classifier_free_guidance # magic decorator
    def forward(self, image, cond_fns):

        image_features = self.image_model(image)
        print('intermediate layers: ', self.image_model.model.get_intermediate_layers(x=image,n=3))
        print('cond_fns', cond_fns)
        cond_fn = cond_fns[0] # get the first function in the list

        # Access the intermediate activations
        for idx, feature in enumerate(self.image_model.features.values()):
            print(f"Block {idx} output shape: {feature.shape}")
            cond_feature = cond_fn(feature).cuda() # condition the feature
            print(f"Block {idx} conditioned output shape: {cond_feature.shape}")
            # replace the original feature with the conditioned feature so that the next block can use it
            self.image_model.features[idx] = cond_feature 

        x = torch.zeros((4, 10)).cuda()
        x.requires_grad = True

        output = {
            'left_fingertip': x[:, 0:3],
            'right_fingertip': x[:, 3:6],
            'force': x[:, 6:9],
            'pitch': x[:, 9]
        }

        return output  

    
class ConditionedVisionTransformer(nn.Module):
    def __init__(self, vit_model, text_model, num_timesteps, hidden_dim=256, num_outputs=10):
        super(ConditionedVisionTransformer, self).__init__()
        self.vit_model = vit_model
        
        # Cross-Attention
        self.text_conditioner = AttentionTextConditioner(
            model_types =  text_model,    
            hidden_dims = tuple([self.vit_model.embed_dim] * len(self.vit_model.model.blocks)),
            cond_drop_prob = 0.0 # 0.2
        ).cuda()

        self.mlp_fingertips = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )

        self.mlp_force = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.mlp_grip_force = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_timesteps),
        )

        self.mlp_width = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mlp_yaw = nn.Sequential(
            nn.Linear(self.vit_model.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # CONV DECODER
        self.pixel_decoder = ConvDecoder(self.vit_model, patch_size=16, num_patches=196) 
        self.depth_decoder = ConvDecoder(self.vit_model, patch_size=16, num_patches=196)

    @classifier_free_guidance # magic decorator
    # model must take in "texts" as an arg when called, and cond_fns as an arg here
    def forward(self, image, cond_fns): 
        # Preprocessing steps of the ViT model
        x = self.vit_model.model.patch_embed(image) # patch embed input is RGBD
        x = self.vit_model.model._pos_embed(x) # includes pos_drop
        x = self.vit_model.model.norm_pre(x)

        # Apply each block with conditioning
        for idx, (block, cond_fn) in enumerate(zip(self.vit_model.model.blocks, cond_fns)):
            x = block(x)
            # method is'xattn':
            x = cond_fn(x.permute(0, 2, 1)).permute(0, 2, 1) # permuting for xattn then permuting back

        # Postprocessing steps of the ViT model
        x = self.vit_model.model.norm(x)
        x = self.vit_model.model.fc_norm(x)
        x = self.vit_model.model.head(x)

        # avg pooled patch embeddings
        patch_feats = x[:, 1:]

        # cls token
        # avg pool the patch embeddings instead of using the cls token. shape=(batch_size, embed_dim)
        cls = torch.mean(patch_feats, dim=1) 
        force = self.mlp_force(cls)
        grip_force = self.mlp_grip_force(cls)
        timestep = self.classifier(cls)

        pixel_output = self.pixel_decoder(x[:, 1:]) # skip the cls token
        pixel_output = pixel_output.view(pixel_output.shape[0], -1)
        depth_output = self.depth_decoder(x[:, 1:]) # skip the cls token
        depth_output = depth_output.view(depth_output.shape[0], -1)
        output = torch.cat((pixel_output, depth_output), dim=1)
        
        output = torch.cat((output, force), dim=1)
        output = torch.cat((output, grip_force), dim=1)

        output = torch.cat((output, timestep), dim=1)

        width = self.mlp_width(cls)
        output = torch.cat((output, width), dim=1)
        
        yaw = self.mlp_yaw(cls)
        output = torch.cat((output, yaw), dim=1)

        # classifier-free guidance library needs logits, not dict
        return output  

        
########################################################################

def t2np(tensor):
    return tensor.detach().cpu().numpy()[0]

def t2float(tensor):
    return tensor.detach().cpu().numpy().item()

def load_model(image_model, pretrained, num_timesteps, checkpoint_path=None):
    print('loading model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_model.split('-')[0] == 'vit':

        image_model = RGBDViT(size=image_model.split('-')[1],
                              num_classes=0,
                              pretrained=pretrained).to(device)  # tiny, small, base, large
        # train the patch embedding layer
        image_model.model.patch_embed.proj.requires_grad = True
    else:
        print('load_model: Image model not recognized.')

    model = ConditionedVisionTransformer(
        image_model,
        text_model='t5',  # text model can only be t5 or clip for now
        num_timesteps=num_timesteps,
        hidden_dim=256
    )
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    return model, tokenizer


########################################################################

def visualize_points(img, points, camera_intrinsics, colors=[(0, 100, 255), (0, 255, 100)],
                     show_depth=True, show_point=True):
    # points is a list of numpy arrays of shape (3,) in the camera frame
    # we want to draw a circle at each point
    intr =  camera_intrinsics
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()
    h, w, _ = img.shape

    # for point in points:
    for i in range(len(points)):
        color = colors[i]
        point = points[i]
        if point is None:
            continue
        point = point.reshape(3, 1) # (3,) -> (3, 1)
        # if points are Nan, skip
        if np.isnan(point).any():
            return img

        point2d = cv2.projectPoints(
            point, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]
        
        if show_point:
            img = cv2.circle(img, (int(point2d[0]), int(point2d[1])), 4, color, -1)
        
        # clip the point that is out of the image
        point2d[0] = np.clip(point2d[0], 0, w)
        point2d[1] = np.clip(point2d[1], 0, h)
    
        if show_depth:
            if i == 0:
                x_offset = int(point2d[0]) - 50 - 80
            else:
                x_offset = int(point2d[0]) - 50 + 80
            x_offset = np.clip(x_offset, 0, w)

            # add z-distance text to the image
            z_dist = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)[0]

            cv2.putText(img, 
                        "D: {:.2f}m".format(z_dist),  # text
                        (x_offset, int(point2d[1] - 10)),  # bottom left coordinate
                        cv2.FONT_HERSHEY_SIMPLEX,  # font family
                        0.7,  # font size
                        (255, 200, 0),  # font color
                        2,  # font stroke
                        lineType=cv2.LINE_AA)


def filled_arrowedLine(img, pt1, pt2, color, thickness=1, line_type=8, tipLength=20):
    # Convert points to numpy arrays
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    
    # Draw main line of the arrow
    cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness, line_type)
    
    # Calculate the angle of the arrow
    angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    
    # Calculate the tip size based on the length of the arrow and the tipLength factor
    tipSize = tipLength

    # # moving pt2 to the tip of the arrow
    pt2_long = np.array([pt2[0] - tipSize * np.cos(angle),
                    pt2[1] - tipSize * np.sin(angle)], dtype=np.float32)

    # Calculate points for the arrowhead
    p1 = np.array([pt2_long[0] + tipSize * np.cos(angle + np.pi/12), 
                   pt2_long[1] + tipSize * np.sin(angle + np.pi/12)], dtype=np.int32)
    p2 = np.array([pt2_long[0] + tipSize * np.cos(angle - np.pi/12), 
                   pt2_long[1] + tipSize * np.sin(angle - np.pi/12)], dtype=np.int32)
    
    # Draw the filled arrowhead
    cv2.fillPoly(img, np.array([[tuple(pt2_long.astype(int)), p1, p2]], dtype=np.int32), color, lineType=line_type)



def visualize_forces(img, origin, ft, camera_intrinsics, color=(255, 255, 0), force_scale=2e-2):
    """
    ft is a numpy array of shape (6,) in the camera frame
    the first three elements are the force vector
    we want to draw an arrow at the center of the image
    """
    intr =  camera_intrinsics
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    force_ref_cam = ft[:3] @ ft_to_cam_rotation()

    camera_force = force_ref_cam.reshape(3, 1)  # (3,) -> (3, 1)
    located_camera_force = origin.reshape(3, 1) - force_scale * camera_force.reshape(3, 1)  # Subtract instead of adding
    origin_coords = cv2.projectPoints(
        origin.reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]
    force_coords = cv2.projectPoints(
        located_camera_force, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]

    if np.isnan(force_coords).any() or np.isnan(origin_coords).any():
        return img

    pixel_mag = np.linalg.norm(force_coords - origin_coords)
    # if the force is too small or too large, don't draw it
    if pixel_mag < 2 or pixel_mag > 1000:
        return img

    filled_arrowedLine(img, (int(origin_coords[0]), int(origin_coords[1])), (int(force_coords[0]), int(force_coords[1])), color, 4, line_type=cv2.LINE_AA, tipLength=20)



def visualize_grip_force(img, grip_force, points, camera_intrinsics, color=(0, 255, 0), force_scale=1e1):   
    if len(points) != 2:
        return img

    intr =  camera_intrinsics
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    # draw arrowline for each fingertip pointing horizontally
    left_fingertip = points[0]
    right_fingertip = points[1]

    left_fingertip = cv2.projectPoints(
        points[0].reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]
    right_fingertip = cv2.projectPoints(
        points[1].reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]

    grip_force = max(0.01, grip_force) # to avoid backwards arrows
    tiplength = 1 / grip_force
    cv2.arrowedLine(img, (int(left_fingertip[0]- force_scale * grip_force), int(left_fingertip[1])),
                          (int(left_fingertip[0] ), int(left_fingertip[1])), color, 4, line_type=cv2.LINE_AA, tipLength=min(tiplength, 1))  # = 0.3)
    cv2.arrowedLine(img, (int(right_fingertip[0] + force_scale * grip_force), int(right_fingertip[1])),
                          (int(right_fingertip[0] ), int(right_fingertip[1])), color, 4, line_type=cv2.LINE_AA, tipLength=min(tiplength, 1)) # = 0.3)


def visualize_prompt(image, prompt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.8
    text_size = cv2.getTextSize(prompt, font, font_size, 5)
    (text_width, text_height), text_baseline = text_size
    location = np.array([20, text_height + 20])
    location = location.astype(np.int32)
    cv2.putText(image, prompt, location, font, font_size, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, prompt, location, font, font_size, (255, 255, 255), 2, cv2.LINE_AA)
    
    
########################################################################

def preprocess_rgbd(image_size, rgb_image, depth_image):
    # process the raw images from the realsense for the model
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = cv2.resize(rgb_image, (image_size, image_size))
    rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

    depth_image = depth_image.astype(np.float32)
    depth_image = cv2.resize(depth_image, (image_size, image_size),
                             interpolation=cv2.INTER_NEAREST)
    depth_image = torch.from_numpy(depth_image).unsqueeze(0).float() / 65535.0
    return rgb_image, depth_image


def postprocess_output(image_size, num_timesteps, output, stage):
    # convert to dict if not already
    if isinstance(output, dict):
        return output

    img_t_size = image_size**2
    # taking softmax of cls_img over the pixels (it's a tensor of shape (batch_size, 1, 224, 224))
    cls_output = output[:, 0:img_t_size]
    if stage == 'metrics':
        cls_output= torch.nn.functional.softmax(cls_output, dim=1)
    cls_output = cls_output.view(-1, 1, image_size, image_size)
    output_dict = {
            'cls_img': cls_output,
            'reg_img': output[:, img_t_size:2*img_t_size].reshape(-1, 1, image_size, image_size),
            'force': output[:, 2*img_t_size:2*img_t_size+3],
            'grip_force': output[:, 2*img_t_size+3],
            'timestep': output[:, 2*img_t_size+4:2*img_t_size+4 + num_timesteps]
        }

    output_dict['width'] = output[:, 2*img_t_size+4 + num_timesteps]
    output_dict['yaw'] = output[:, 2*img_t_size+4 + num_timesteps + 1]

    return output_dict


def recover_pixel_space_represention(cls_img, reg_img):
    cls_img = cls_img.detach().squeeze(0).cpu().numpy()[0]
    reg_img = reg_img.detach().squeeze(0).cpu().numpy()[0]
    reg_img = reg_img * 65535.0
    
    # resize back to 480x640
    cls_img = cv2.resize(cls_img, (640, 480), interpolation=cv2.INTER_NEAREST)
    reg_img = cv2.resize(reg_img, (640, 480), interpolation=cv2.INTER_NEAREST)

    # return contact
    return cls_img, reg_img


def pixel_space_to_centroid(cls_img, reg_img, camera_intrinsics, threshold=0.005, avg_distance_within_radius=None):
    intr = camera_intrinsics
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    # get height width form cls_img
    h, w = cls_img.shape
    
    # Find local peaks in the cls_img
    coordinates = peak_local_max(cls_img,
                                 num_peaks=1,
                                 threshold_abs=threshold,
                                 min_distance=20,
                                 exclude_border=False)
    if len(coordinates) == 0:
        return None, None, None
    y, x = coordinates[0]
    centroid_pixel_value = cls_img[y,x]

    if avg_distance_within_radius is not None:
        # get the reg_img value at a radius around y, x and average it.
        # make sure y and x are within bounds
        start_x = max(0, x - avg_distance_within_radius)
        start_y = max(0, y - avg_distance_within_radius)
        end_x = min(w, x + avg_distance_within_radius)
        end_y = min(h, y + avg_distance_within_radius)
        region = reg_img[start_y:end_y, start_x:end_x]
        ray_z_dist = np.mean(region) * intr.depth_scale
    else:
        ray_z_dist = reg_img[y, x] * intr.depth_scale

    # Construct a 2D point in normalized image coordinates
    
    point2d = np.array([[[(x - cam_mat[0, 2]) / cam_mat[0, 0],
        (y - cam_mat[1, 2]) / cam_mat[1, 1]]]], dtype=np.float32)
    xy_dist = np.sqrt(point2d[0, 0, 0]**2 + point2d[0, 0, 1]**2)
    z_dist = np.sqrt(ray_z_dist**2/(1 + xy_dist**2))

    # Project the 2D point to 3D. Here we make use of the known depth (z-coordinate)
    centroid_3d_point = np.array([point2d[0, 0, 0] * z_dist, point2d[0, 0, 1] * z_dist, z_dist])

    centroid_pixel_coordinates = np.array([y, x])
    return centroid_3d_point, centroid_pixel_coordinates, centroid_pixel_value


def centroid_to_fingertips(grip_center, grip_width, gripper_yaw=0.0):
    """Estimate the 3D coordinates for the fingertips based on the 3D
point between the fingertips (grip_center), the distance between the
fingertips (grip_width), and the yaw angle of the gripper
(gripper_yaw).

A yaw angle of zero corresponds to the gripper pointing in the
direction of the camera's optical axis. Positive yaw angles correspond
with rotating the gripper toward the left of the image. Negative yaw
angles correspond with rotating the gripper toward the right of the
image.

This is an approximation that is sufficient for visual
servoing. Ideally, the rotational axis for yaw would be parallel to
gravity rather than parallel to the upward pointing axis of the
camera's frame of reference. This would require modeling the
relationship between the camera's reference frame and the world frame,
such as how far downward the camera is pointing.

    """
    
    fingertip_vector = (grip_width/2.0) * np.array([np.cos(gripper_yaw), 0.0, np.sin(gripper_yaw)])
    left_fingertip = grip_center - fingertip_vector
    right_fingertip = grip_center + fingertip_vector
    return left_fingertip, right_fingertip

########################################################################

class ForceSightMin():
    def __init__(self, model_filename):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.default_confidence_threshold = 0.008 #0.006 #0.01 #0.008

        self.image_size = 224
        self.num_timesteps = 4
        self.pretrained = True
        self.image_model = 'vit-large'
        self.model_filename = model_filename
        self.model_path = './' + self.model_filename
        self.model, self.tokenizer = load_model(self.image_model, self.pretrained, self.num_timesteps, self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.camera_intrinsics = Intrinsic640()
        

    def get_default_confidence_threshold(self):
        return self.default_confidence_threshold

    def set_camera_info(self, camera_info):
        self.camera_intrinsics = CustomIntrinsic(camera_info)
        
    def apply(self, color_image, depth_image, prompt, camera_info, depth_scale):

        left_fingertip = None
        right_fingertip = None

        if not (prompt == 'Do Nothing'): 
            prompt_input = list([prompt])

            # stacking rgb and depth on the channel dimension
            rgb_input, depth_input = preprocess_rgbd(self.image_size, color_image, depth_image)
            rgbd_input = torch.cat((rgb_input, depth_input), dim=0)
            rgbd_input = rgbd_input.to(self.device)
            rgbd_input = rgbd_input.unsqueeze(0)

            pred = self.model(rgbd_input, texts=prompt_input, cond_scale=1.)
            pred = postprocess_output(self.image_size, self.num_timesteps, pred, stage='metrics')

            timestep_confidence = t2np(pred['timestep'])
            timestep_index = np.argmax(timestep_confidence)
            applied_force = t2np(pred['force'])
            applied_force -= EEF_PITCH_WEIGHT_OFFSET[:3]
            grip_force = t2float(pred['grip_force'])
            grip_width = t2float(pred['width'])
            gripper_yaw = t2float(pred['yaw'])
            grip_center_image, grip_distance_image = recover_pixel_space_represention(pred['cls_img'], pred['reg_img'])

            grip_center_xyz, grip_center_pix, confidence = pixel_space_to_centroid(
                grip_center_image,
                grip_distance_image,
                self.camera_intrinsics,
                threshold=0.002,
                avg_distance_within_radius=2)


            if grip_center_xyz is not None: 
                left_fingertip, right_fingertip = centroid_to_fingertips(grip_center_xyz, grip_width, gripper_yaw)

            print('confidence =', confidence)

            applied_force_camera_n = None
            if applied_force is not None:
                 applied_force_camera_n = applied_force[:3] @ ft_to_cam_rotation()
        else:
            confidence = None
            timestep_index = None
            grip_center_xyz = None
            grip_center_pix = None
            gripper_yaw = None
            grip_width = None
            grip_force = None
            applied_force = None
            applied_force_camera_n = None
            grip_center_image = None
            grip_distance_image = None
                 
        prediction = {
            'prompt': prompt,
            'confidence': confidence,
            'timestep_index': timestep_index, 
            'fingertips': {
                'left': {
                    'xyz_m': left_fingertip,
                    'xy_pix': None
                },
                'right': {
                    'xyz_m': right_fingertip,
                    'xy_pix': None
                }
            },
            'grip_center': {
                'xyz_m': grip_center_xyz,
                'xy_pix': grip_center_pix
            },
            'gripper_yaw_rad': gripper_yaw,
            'grip_width_m': grip_width,
            'grip_force_n': grip_force,
            'applied_xyz_force_n': applied_force,
            'applied_force_camera_n' : applied_force_camera_n,
            'grip_center_image': grip_center_image,
            'grip_distance_image': grip_distance_image
        }
            
        return prediction

    
    def draw_prediction(self, image, prediction, confidence_threshold=None, show_depth=True): 
                
        if prediction is not None:
            left_fingertip = prediction['fingertips']['left']['xyz_m']
            right_fingertip = prediction['fingertips']['right']['xyz_m']
            fingertips = [left_fingertip, right_fingertip]
            grip_force = prediction['grip_force_n']
            applied_force = prediction['applied_xyz_force_n']
            confidence = prediction['confidence']
            timestep = prediction['timestep_index']
            
            print('predicted timestep index =', timestep)
            if (confidence is not None) and (confidence_threshold is not None):
                above_threshold = confidence >= confidence_threshold
            else:
                above_threshold = True
                
            if above_threshold:
                if (left_fingertip is not None) and (right_fingertip is not None):
                    between_fingertips = (left_fingertip + right_fingertip) / 2
                    print('3D point between fingertips =', between_fingertips)

                    visualize_points(image, fingertips, self.camera_intrinsics, colors=[(0, 255, 0), (0, 255, 0)], show_depth=show_depth)
                    if grip_force is not None: 
                        visualize_grip_force(image, grip_force, fingertips, self.camera_intrinsics, color=(0, 255, 0))
                    if applied_force is not None: 
                        visualize_forces(image, between_fingertips, applied_force, self.camera_intrinsics, color=(0, 255, 255))

        prompt = prediction.get('prompt', None)
        if prompt is not None: 
            visualize_prompt(image, prompt)

    
    def show_prediction_images(self, prediction): 
                
        if prediction is not None:
            grip_center_image = prediction['grip_center_image']
            grip_distance_image = prediction['grip_distance_image']

            if grip_center_image is not None: 
                min_confidence = np.min(grip_center_image)
                max_confidence = np.max(grip_center_image)
                if (max_confidence - min_confidence) > 0.0: 
                    display_grip_center_image = ((np.copy(grip_center_image) - min_confidence) /
                                                 (max_confidence - min_confidence))
                    cv2.imshow('Grip Center Image', display_grip_center_image)

            if grip_distance_image is not None: 
                min_distance = np.min(grip_distance_image)
                max_distance = np.max(grip_distance_image)
                if (max_distance - min_distance) > 0.0: 
                    display_grip_distance_image = ((np.copy(grip_distance_image) - min_distance) /
                                                 (max_distance - min_distance))
                    cv2.imshow('Grip Distance Image', display_grip_distance_image)


    def prediction_without_images(self, prediction):
        # For some applications, the predicted images are not
        # necessary. This method returns a prediction dictionary
        # without images that can be more efficiently transferred
        # across processes and machines.

        if prediction is None:
            return prediction
        
        def safe_remove(v, l):
            if v in l:
                l.remove(v)
        
        keys = list(prediction.keys())
        val = 'grip_center_image'
        safe_remove(val, keys)
        val = 'grip_distance_image'
        safe_remove(val, keys)

        output = {k:v for k,v in prediction.items() if k in keys}
        
        return output
